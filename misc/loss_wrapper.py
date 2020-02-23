import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward, get_self_critical_reward_using_model
from misc.utils import decode_sequence_to_dict

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt, ix_to_word=None, cider_dataset=None, cider_model=None):
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

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag, image_ids):
        opt = self.opt        
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

            if self.opt.use_model_for_sc_train == 0:
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            else:
                reward = get_self_critical_reward_using_model(self.cider_dataset, self.cider_model, greedy_captions, gen_captions, opt, gen_result.shape[1])
            
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            out['reward'] = reward[:,0].mean()
	
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)

        out['loss'] = loss
        return out
