import torch
import numpy as np
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.utils import decode_sequence_to_dict
from misc.slor_scoring import get_slor_rewards
from misc.cider_scoring import get_self_critical_cider_reward_using_model
from misc.vifidel_scoring import get_vifidel_rewards

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt, ix_to_word=None, cider_dataset=None, cider_model=None, language_model=None, language_model_tokenizer = None, unigram_prob_dict=None, glove_embedding=None, glove_word_to_ix=None, ground_truth_object_annotations=None):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            assert False
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.struc_crit = utils.StructureLosses(opt)
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

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag, image_ids):
        out = {}
        reduction = 'none' if drop_worst_flag else 'mean'

        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:], reduction=reduction)        
        else:
            print("Performing self critical training")
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')

            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            
            greedy_captions = decode_sequence_to_dict(self.ix_to_word, greedy_res, image_ids)
            gen_captions = decode_sequence_to_dict(self.ix_to_word, gen_result, image_ids)
            length_of_output = gen_result.shape[1]

            reward = np.zeros((len(greedy_captions), length_of_output))
            if self.opt.use_ref_caps:
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            else:
                if self.opt.use_cider:
                    cider_reward, average_greedy_cider, average_gen_cider = get_self_critical_cider_reward_using_model(self.cider_dataset, self.cider_model, greedy_captions, gen_captions, self.opt, length_of_output)
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

            if self.opt.save_all_train_captions is not None:
                out['gen_captions'] = gen_captions
                out['greedy_captions'] = greedy_captions
	
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)

        out['loss'] = loss
        return out
