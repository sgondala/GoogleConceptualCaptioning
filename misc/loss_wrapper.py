import torch
from torch.autograd import Variable
import numpy as np
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.utils import decode_sequence_to_dict, clip_gradient
from misc.slor_scoring import get_slor_rewards
from misc.cider_scoring import get_self_critical_cider_reward_using_model
from misc.vifidel_scoring import get_vifidel_rewards
from eval_utils import language_eval_for_coco

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt, ix_to_word=None, cider_dataset=None, cider_model=None, language_model=None, language_model_tokenizer = None, unigram_prob_dict=None, glove_embedding=None, glove_word_to_ix=None, ground_truth_object_annotations=None, model_greedy=None, is_classification_cider_model=False, classification_threshold = 0.999):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.model_greedy = model_greedy
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion() if opt.do_not_use_ppo else utils.PPOCriterion(opt.ppo_clip_param)
        self.retrieval_reward_weight = 0
        self.ix_to_word = ix_to_word
        self.cider_dataset = cider_dataset
        self.cider_model = cider_model
        self.language_model = language_model
        self.language_model_tokenizer = language_model_tokenizer
        self.unigram_prob_dict = unigram_prob_dict
        self.glove_embedding = glove_embedding
        self.glove_word_to_ix = glove_word_to_ix
        self.ground_truth_object_annotations = ground_truth_object_annotations
        self.is_classification_cider_model = is_classification_cider_model
        self.classification_threshold = classification_threshold
        self.optimizer = None

    def post_process(self, captions_list):
        ret_list = []
        for entry in captions_list:
            image_id = entry['image_id'].item()
            caption = entry['caption']
            new_dict = {}
            new_dict['image_id'] = image_id
            new_dict['caption'] = caption
            ret_list.append(new_dict)
        return ret_list
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag, image_ids):

        assert self.optimizer is not None
        self.optimizer.zero_grad()

        out = {}
        out['gen_captions'] = []
        out['greedy_captions'] = []

        reduction = 'none' if drop_worst_flag else 'mean'

        if not self.opt.do_not_use_ppo and sc_flag: # PPO self-critical update
            # PPO code taken from https://github.com/clu8/self-critical-ppo

            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model_greedy(fc_feats, att_feats, att_masks, mode='sample')

            self.model.train()
            gen_result, old_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')

            # All sentences{}

            # For every caption
                # Say caption i
                # For every token
                    # Current sentence = [apprend all tokens till now]
                        # Generate sentences using beam search? - Fast
                            # All_sentences[i] += [these to sentences]
            
            # All sentences [i] = 10 * 20 sentences
            # If batch size = n
            # All sentences = n * 10 * 20
            # Collate all into a singel array = [] - 200n
            # Calcualte cider for that array 
            # We get 200n cider scores
            # Now you can just use indexing to calculate cider for cider[i,j] - Assigning right

            # The above function returns prob of *all tokens* at each step
            # We need to pick the probabiliity of selected token
            # Other things to take care - log prob - Line 130 new_logprobs[:, 1:] = new_logprobs[:, 1:] * Variable((gen_result[:, :-1] > 0).float(), requires_grad=False)
            # 

            old_logprobs = old_logprobs.gather(2, gen_result.unsqueeze(2)).squeeze(2)
            assert old_logprobs.shape == gen_result.shape

            old_logprobs = old_logprobs.detach()
            old_logprobs[:, 1:] = old_logprobs[:, 1:] * Variable((gen_result[:, :-1] > 0).float(), requires_grad=False)
            old_logprobs_agg = old_logprobs.sum(dim=1)
            
            gts = [gts[_] for _ in gt_indices.tolist()]
            greedy_captions = decode_sequence_to_dict(self.ix_to_word, greedy_res, image_ids)
            gen_captions = decode_sequence_to_dict(self.ix_to_word, gen_result, image_ids)

            out['gen_captions'] = self.post_process(gen_captions)
            out['greedy_captions'] = self.post_process(greedy_captions)

            length_of_output = gen_result.shape[1]

            reward = np.zeros((fc_feats.shape[0], length_of_output))
            score = None 

            if self.opt.use_ref_caps:
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
                out['reward'] = reward[:,0].mean()
                # cider_array_greedy = np.array(language_eval_for_coco(out['greedy_captions'], self.opt.id_language_eval)['CIDErArary'])
                # cider_array_gen = np.array(language_eval_for_coco(out['gen_captions'], self.opt.id_language_eval)['CIDErArary'])
                # score = cider_array_gen - cider_array_greedy
                # out['reward'] = score
                # reward = np.repeat(score[:, np.newaxis], length_of_output, 1)
            else:
                if self.opt.use_cider:
                    cider_reward, average_greedy_cider, average_gen_cider = get_self_critical_cider_reward_using_model(self.cider_dataset, self.cider_model, greedy_captions, gen_captions, self.opt, length_of_output, self.is_classification_cider_model, self.classification_threshold)
                    assert cider_reward.shape == reward.shape
                    reward += cider_reward
                    out['average_greedy_cider'] = average_greedy_cider
                    out['average_gen_cider'] = average_gen_cider
                    out['reward'] = reward[:,0].mean()
                else:
                    assert False, 'Use cider atleast'

            score = reward[:, 0]

            for ppo_iter in range(self.opt.ppo_iters):
                new_logprobs = self.model.get_seq_logprobs(fc_feats, att_feats, att_masks, gen_result)
                new_logprobs = new_logprobs.gather(2, gen_result.unsqueeze(2)).squeeze(2)
                new_logprobs[:, 1:] = new_logprobs[:, 1:] * Variable((gen_result[:, :-1] > 0).float(), requires_grad=False)

                new_logprobs_agg = new_logprobs.sum(dim=1)
                loss = self.rl_crit(old_logprobs_agg, new_logprobs_agg, Variable(torch.from_numpy(score).float().cuda(), requires_grad=False))

                self.optimizer.zero_grad()
                if ppo_iter < self.opt.ppo_iters - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()

        elif not sc_flag:
            # print("Using CE Loss")
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:], reduction=reduction)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_gradient(self.optimizer, self.opt.grad_clip)
            self.optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
        '''
        else:
            assert False
            # SC reinforce
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model_greedy(fc_feats, att_feats, att_masks, mode='sample')

            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')

            # Assuming 1 seq per image
            # We have token and it's log probability - not previous or next step's
            # Gen result shape  torch.Size([batch_size, 20 (length of caption)])
            # Sample logprobs shape  torch.Size([batch_size, 20, 16304 (size of vocab)])

            gts = [gts[_] for _ in gt_indices.tolist()]
            
            greedy_captions = decode_sequence_to_dict(self.ix_to_word, greedy_res, image_ids)
            gen_captions = decode_sequence_to_dict(self.ix_to_word, gen_result, image_ids)
            length_of_output = gen_result.shape[1]

            reward = np.zeros((len(greedy_captions), length_of_output))
            if self.opt.use_ref_caps:
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            else:
                if self.opt.use_cider:
                    cider_reward, average_greedy_cider, average_gen_cider = get_self_critical_cider_reward_using_model(self.cider_dataset, self.cider_model, greedy_captions, gen_captions, self.opt, length_of_output, self.is_classification_cider_model, self.classification_threshold)
                    assert cider_reward.shape == reward.shape
                    reward += cider_reward
                    out['average_greedy_cider'] = average_greedy_cider
                    out['average_gen_cider'] = average_gen_cider
                if self.opt.use_slor:
                    slor_reward, average_greedy_slor, average_gen_slor = get_slor_rewards(greedy_captions, gen_captions, self.unigram_prob_dict, self.language_model_tokenizer, self.language_model, length_of_output)
                    assert slor_reward.shape == reward.shape
                    reward += slor_reward
                    out['average_greedy_slor'] = average_greedy_slor
                    out['average_gen_slor'] = average_gen_slor
                if self.opt.use_vifidel:
                    vifidel_reward, average_greedy_vifidel, average_gen_vifidel = get_vifidel_rewards(greedy_captions, gen_captions, self.ground_truth_object_annotations, self.glove_embedding, self.glove_word_to_ix, length_of_output)
                    assert vifidel_reward.shape == reward.shape
                    reward += vifidel_reward
                    out['average_greedy_vifidel'] = average_greedy_vifidel
                    out['average_gen_vifidel'] = average_gen_vifidel

            reward = torch.from_numpy(reward).float().to(gen_result.device)
            out['reward'] = reward[:,0].mean()

            # Needed to get groundtruth cider
            out['gen_captions'] = self.post_process(gen_captions)
            out['greedy_captions'] = self.post_process(greedy_captions)

            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_gradient(self.optimizer, self.opt.grad_clip)
            self.optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
        '''
        out['loss'] = loss
        return out