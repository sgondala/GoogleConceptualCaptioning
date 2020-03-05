
import numpy as np
import torch
from nltk import word_tokenize

def get_log_probability(batch_caption, language_model_tokenizer, language_model):
    batch_tokens = [language_model_tokenizer.convert_tokens_to_ids(
        language_model_tokenizer.tokenize(x, add_prefix_space=True))
        for x in batch_caption]

    # add BOS and EOS
    tokenize_input = [[language_model_tokenizer.bos_token_id] + x + [language_model_tokenizer.eos_token_id]
            for x in batch_tokens]
    mask = []
    for sentence in tokenize_input:
        curr_mask = [1]*len(sentence) + [0] * (20-len(sentence))
        mask.append(curr_mask)
    mask = torch.FloatTensor(mask)
    for i in range(len(tokenize_input)):
        tokenize_input[i] = tokenize_input[i][: 20]
        tokenize_input[i].extend([language_model_tokenizer.eos_token_id] * (20 - len(tokenize_input[i]))) 
    tokenize_input = torch.LongTensor(tokenize_input)
    with torch.no_grad():
        outputs = language_model(tokenize_input, labels=tokenize_input, attention_mask = mask)
        _, logits = outputs[:2]
        
    batch_lp = []
    for index in range(len(tokenize_input)):
        lp = 0.0
        for j in range(len(tokenize_input[index])):
            if mask[index][j]:
                masked_index = j
                predicted_score = logits[index][masked_index]
                predicted_prob = torch.nn.functional.softmax(predicted_score)
                lp += np.log(predicted_prob[language_model_tokenizer.convert_tokens_to_ids([tokenize_input[index][j]])[0]])
        batch_lp.append(lp)
    return batch_lp

def get_slor_score(unigram_prob_dict, logprob, caption):
    # TODO: Update with batch SLOR
    uni = 0.0
    caption = caption.strip('.')
    for w in word_tokenize(caption):
        if not w.lower()[-1].isalpha():
            w = w.lower()[:-1]
        if w.lower() not in unigram_prob_dict.keys():
            return -1
        uni += np.log(unigram_prob_dict[w.lower()])
    n = len(caption.split())
    return ((logprob - uni) / n)

def clip_slor_scores(slor_scores):
    # Convert np array from -3 to 4 to 0 and 2
    scores = np.minimum(slor_scores, 4)
    scores = np.maximum(scores, -3)
    scores += 3
    scores = scores / 7.0
    return scores

def get_slor_rewards(greedy_captions, gen_captions, unigram_prob_dict, language_model_tokenizer, language_model, length_of_output):
    greedy_captions_list = [entry['caption'] for entry in greedy_captions]
    gen_captions_list = [entry['caption'] for entry in gen_captions]
    greedy_captions_log_probabilities = get_log_probability(greedy_captions_list, language_model_tokenizer, language_model)
    gen_captions_log_probabilities = get_log_probability(gen_captions_list, language_model_tokenizer, language_model)
    greedy_slor_scores = clip_slor_scores(get_slor_score(unigram_prob_dict, greedy_captions_log_probabilities, greedy_captions_list))
    gen_slor_scores = clip_slor_scores(get_slor_score(unigram_prob_dict, gen_captions_log_probabilities, gen_captions_list))
    assert len(greedy_slor_scores) == len(gen_slor_scores)
    scores = gen_slor_scores - greedy_slor_scores
    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards