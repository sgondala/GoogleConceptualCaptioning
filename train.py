from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import sys

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
import opts
import models
from dataloader import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper
from misc.utils import decode_sequence
from eval_utils import language_eval
import math

# print("Imported all")
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def eval_cider_and_append_values(predictions, writer, key, start_iteration, batch_size):
    # TODO - This hardcodes to coco val. Take care
    out = language_eval(None, predictions, None, {}, 'val')
    cider_array = np.array(out['CIDErArary'])
    
    for i in range(math.ceil(len(cider_array) * 1.0 /batch_size)):
        start_index = i*batch_size
        end_index = i*batch_size + batch_size
        cider_avg = cider_array[start_index:end_index].mean()
        add_summary_value(writer, key, cider_avg, start_iteration + i)
    
    return i + start_iteration

def train(opt):
    # Deal with feature things before anything
    print("Opt input", opt)
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    if opt.vse_model == 'fc':
        opt.use_fc = True
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5
    
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                print(vars(saved_model_opt)[checkme], vars(opt)[checkme])
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    infos['iter'] = 0
    infos['epoch'] = 0
    infos['iterators'] = loader.iterators
    infos['split_ix'] = loader.split_ix
    infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    print(model.parameters)
    
    model_greedy = None
    initial_greedy_model_weights = []
    if True:
        model_greedy = models.setup(opt).cuda()
        model_greedy.eval()
        initial_greedy_model_weights = list(model_greedy.parameters())
        for param in model_greedy.parameters():
            param.requires_grad = False

    dp_model = torch.nn.DataParallel(model)
    

    vocab = opt.vocab
    # del opt.vocab

    cider_model = None
    cider_dataset = None

    open_gpt_tokenizer = None
    open_gpt_model = None
    unigram_prob_dict = None

    glove_embedding = None
    glove_word_to_ix = None
    ground_truth_object_annotations = None

    initial_cider_model_weights = []
    final_cider_model_weights = []

    if opt.self_critical_after != -1 and not opt.use_ref_caps:
        # CIDEr
        if opt.use_cider:
            print("Using cider")
            from vilbert.vilbert import BertConfig
            from vilbert.vilbert import VILBertForVLTasks

            from CiderDataset import CiderDataset

            from pytorch_pretrained_bert.tokenization import BertTokenizer

            config = BertConfig.from_json_file(opt.config_file)
            cider_model = VILBertForVLTasks.from_pretrained(opt.cider_model, config, num_labels=1, default_gpu=True)
            cider_model.cuda()
            cider_model.eval()
            for param in cider_model.parameters():
                param.requires_grad = False

            initial_cider_model_weights = list(cider_model.parameters())

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            cider_dataset = CiderDataset(None, opt.input_fc_dir, tokenizer)

        # SLOR
        if opt.use_slor:
            print("Using slor")
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            open_gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            open_gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
            open_gpt_model.cuda()
            open_gpt_model.eval()
            for param in open_gpt_model.parameters():
                param.requires_grad = False

            unigram_prob_dict = json.load(open(opt.unigram_prob_file, 'r'))

        # VIFIDEL
        if opt.use_vifidel:
            print("Using vifidel")
            glove_embedding = nn.Embedding.from_pretrained(torch.load(opt.glove_vectors), freeze=True)
            glove_word_to_ix = json.load(open(opt.glove_word_to_ix, 'r'))
            ground_truth_object_annotations = json.load(open(opt.ground_truth_object_annotations, 'r'))

    lw_model = LossWrapper(model, opt, vocab, cider_dataset, cider_model, open_gpt_model, open_gpt_tokenizer, unigram_prob_dict, glove_embedding, glove_word_to_ix, ground_truth_object_annotations, model_greedy).cuda()
    
    dp_lw_model = torch.nn.DataParallel(lw_model)

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    if opt.noamopt:
        assert opt.caption_model == 'transformer', 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))


    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)


    gen_captions_all = {}
    greedy_captions_all = {}

    greedy_captions_since_last_checkpoint = []
    gen_captions_since_last_checkpoint = []

    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate  ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                # If start self critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    if opt.use_ref_caps:
                        init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False
                if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
                    drop_worst_flag = True
                else:
                    drop_worst_flag = False

                epoch_done = False
                    
            start = time.time()
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            image_ids = data['image_ids']
            
            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag, image_ids)

            if not drop_worst_flag:
                loss = model_out['loss'].mean()
            else:
                loss = model_out['loss']
                loss, idx = torch.topk(loss, k=int(loss.shape[0] * (1-opt.drop_worst_rate)), largest=False)
                loss = loss.mean()

                idx = set(idx.tolist())
                for ii, s in enumerate(utils.decode_sequence(loader.get_vocab(), labels.cpu()[:, 1:])):
                    if ii not in idx:
                        print(s)

            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()
            if struc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)
                    add_summary_value(tb_summary_writer, 'avg_greedy_cider', model_out.get('average_greedy_cider', 0), iteration)
                    add_summary_value(tb_summary_writer, 'avg_gen_cider', model_out.get('average_gen_cider', 0), iteration)
                    add_summary_value(tb_summary_writer, 'avg_greedy_slor', model_out.get('average_greedy_slor', 0), iteration)
                    add_summary_value(tb_summary_writer, 'avg_gen_slor', model_out.get('average_gen_slor', 0), iteration)
                    add_summary_value(tb_summary_writer, 'avg_greedy_vifidel', model_out.get('average_greedy_vifidel', 0), iteration)
                    add_summary_value(tb_summary_writer, 'avg_gen_vifidel', model_out.get('average_gen_vifidel', 0), iteration)
                    add_summary_value(tb_summary_writer, 'average_gen_cider_ground_truth', model_out.get('average_gen_cider_ground_truth', 0), iteration)
                    add_summary_value(tb_summary_writer, 'average_greedy_cider_ground_truth', model_out.get('average_greedy_cider_ground_truth', 0), iteration)

                    greedy_captions_since_last_checkpoint += model_out['greedy_captions']
                    gen_captions_since_last_checkpoint += model_out['gen_captions']

                    if opt.save_all_train_captions:
                        gen_captions_all[iteration] = model_out['gen_captions']
                        greedy_captions_all[iteration] = model_out['greedy_captions']

                elif struc_flag:
                    add_summary_value(tb_summary_writer, 'lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    add_summary_value(tb_summary_writer, 'struc_loss', model_out['struc_loss'].mean().item(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            
            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0):
                print("Calculating validation score")
                
                eval_kwargs = {'split': 'val',
                                'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                # Rechecking on train data too
                eval_kwargs_train = {'split': 'train',
                                'dataset': opt.input_json}
                eval_kwargs_train.update(vars(opt))
                _, _, train_lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs_train)
                
                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                
                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                
                if train_lang_stats is not None:
                    for k, v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k + '_train', v, iteration)

                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                if iteration != 1:
                    # Calculate actual cider values of previous iterations
                    # Doing this here to take advantage of batching
                    # assert len(greedy_captions_since_last_checkpoint) = opt.save_checkpoint_every * opt.batch_size
                    end_iteration_val = eval_cider_and_append_values(greedy_captions_since_last_checkpoint, tb_summary_writer, 'greedy_generated_captions_actual_cider_scores', iteration - opt.save_checkpoint_every, opt.batch_size)
                    assert end_iteration_val == iteration

                    # assert len(gen_captions_since_last_checkpoint) = opt.save_checkpoint_every * opt.batch_size
                    end_iteration_val = eval_cider_and_append_values(gen_captions_since_last_checkpoint, tb_summary_writer, 'gen_generated_captions_actual_cider_scores', iteration - opt.save_checkpoint_every, opt.batch_size)
                    assert end_iteration_val == iteration

                greedy_captions_since_last_checkpoint = []
                gen_captions_since_last_checkpoint = []

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    save_checkpoint(model, infos, optimizer, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, append='best')

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
        
        if cider_model is not None:
            final_cider_model_weights = list(cider_model.parameters())
            assert initial_cider_model_weights == final_cider_model_weights
        
        if model_greedy is not None:
            final_greedy_model_weights = list(model_greedy.parameters())
            assert initial_greedy_model_weights == final_greedy_model_weights

        if opt.save_all_train_captions:
            final_dict = {}
            final_dict['gen_captions'] = gen_captions_all
            final_dict['greedy_captions'] = greedy_captions_all
            json.dump(final_dict, open(opt.checkpoint_path + '/captions_all.json', 'w'))

    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

opt = opts.parse_opt()
train(opt)