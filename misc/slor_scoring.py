import torch
from nltk import word_tokenize
import numpy as np

PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}" , ".", "?", "!", ",", ":", "-", "--", "...", ";", '$']

def get_log_probability(tokenizer, model, caption):
    tokenize_input = tokenizer.tokenize(caption)
    # 50256 is the token_id for <|endoftext|>
    tensor_input = torch.tensor([[tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenize_input) + [tokenizer.eos_token_id]])
    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        _, logits = outputs[:2]
    lp = 0.0
    for i in range(len(tokenize_input)):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.nn.functional.softmax(predicted_score)
        lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[i]])[0]])
    return float(lp)

def get_slor_score(unigram_prob_dict, logprob, caption):
    caption_tokens = word_tokenize(caption)
    caption_tokens = [ct.lower() for ct in caption_tokens if ct not in PUNCTUATIONS]

    curr_uni = 0.0
    for token in caption_tokens:
        token_prob = 1e-10
        if token in unigram_prob_dict:
            token_prob = unigram_prob_dict[token]
        curr_uni += np.log(token_prob)
    n = len(caption.split())
    return ((logprob - curr_uni) / n)

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
    
    greedy_captions_log_probabilities = [get_log_probability(language_model_tokenizer, language_model, caption) for caption in greedy_captions_list]
    gen_captions_log_probabilities = [get_log_probability(language_model_tokenizer, language_model, caption) for caption in gen_captions_list]
    
    greedy_slor_scores = [get_slor_score(unigram_prob_dict, greedy_captions_log_probabilities[i], greedy_captions_list[i]) for i in range(len(greedy_captions_list))]
    gen_slor_scores = [get_slor_score(unigram_prob_dict, gen_captions_log_probabilities[i], gen_captions_list[i]) for i in range(len(greedy_captions_list))]

    greedy_slor_scores = clip_slor_scores(np.array(greedy_slor_scores))
    gen_slor_scores = clip_slor_scores(np.array(gen_slor_scores))
    
    assert len(greedy_slor_scores) == len(gen_slor_scores)
    
    scores = gen_slor_scores - greedy_slor_scores
    rewards = np.repeat(scores[:, np.newaxis], length_of_output, 1)
    return rewards