from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

from .rewards import get_scores
import torch.nn.functional as F

import six
from six.moves import cPickle

bad_endings = ['with','in','on','of','a','at','to','for','an','this','his','her','that']
bad_endings += ['the']

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def if_use_feat(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_att, use_fc = False, True
    elif caption_model == 'language_model':
        use_att, use_fc = False, False
    elif caption_model == 'topdown':
        use_fc, use_att = True, True
    else:
        use_att, use_fc = True, False
    return use_fc, use_att

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N,L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class StructureLosses(nn.Module):
    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts, reduction='mean'):
        """
        Input is either logits or log softmax
        """
        batch_size = input.size(0)# batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.seq_per_img * self.opt.structure_sample_n, seq_per_img

        mask = (seq>0).float()
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)
        
        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        if self.opt.entropy_reward_weight > 0:
            entropy = - (F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print('entropy', entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        # rescale cost to [0,1]
        costs = - scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin': 
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == 'seqnll':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)
        elif self.loss_type == 'risk':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            assert reduction=='mean'
            output = (F.softmax(input.exp()) * costs).sum(1)

            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())            

        elif self.loss_type == 'max_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()
            assert reduction=='mean'

            # sanity
            # avg_scores = input + costs
            # scores_with_high_target = avg_scores.clone()
            # scores_with_high_target.scatter_(1, costs.min(1)[1].view(-1, 1), 1e10)

            # target_and_offender_index = scores_with_high_target.sort(1, True)[1][:, 0:2]
            # avg_scores = avg_scores.gather(1, target_and_offender_index)
            # target_index = avg_scores.new_zeros(avg_scores.size(0), dtype=torch.long)
            # loss = F.multi_margin_loss(avg_scores, target_index, size_average=True, margin=0)
            # print(loss.item() * 2, output.item()) 

        elif self.loss_type == 'multi_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()
            assert reduction=='mean'

            # sanity
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == 'softmax_margin':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'real_softmax_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'policy_gradient':
            # None:
            # This is not standard pg, because the baseline is dependant on the reward
            # This is eccenstially a rescaled reward. See how it works.
            # output = input * mask * (costs - costs.mean(1, keepdim=True)).view(-1, 1) not working
            # output = input * mask * ((costs - costs.mean(1, keepdim=True))/(costs.std(1, keepdim=True)+1e-7)).view(-1, 1)
            # output = input * mask * costs.view(-1, 1)
            # output = input * mask * (costs.view(-1, 1)-0.5)
            # costs = -scores
            # output = input * mask * ((costs - costs.mean())/costs.std()).view(-1, 1)
            # output = - input * mask * scores.view(-1, 1)
            # output = - input * mask * (scores - scores.min(1, keepdim=True)[0]).view(-1, 1)
            output = - input * mask * (scores - scores.median(1, keepdim=True)[0]).view(-1, 1)
            if reduction == 'none':
                output = output.sum(1) / mask.sum(1)
            elif reduction == 'mean':
                output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None
        
    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        output = self.criterion(input, true_dist).sum(1) * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x,y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x,y: length_wu(x,y,alpha)
    if pen_type == 'avg':
        return lambda x,y: length_average(x,y,alpha)

def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)
        
    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr) # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case
    
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
        
def get_std_opt(model, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(model.model.tgt_embed[0].d_model, factor, warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    

def post_processing_for_conceptual(s):
    print('Test before use!!!!!!!!!!!!!!!!!')
    if s.endswith('stock photo #'):
        s = s[:s.find('stock photo')]
    return s
    
