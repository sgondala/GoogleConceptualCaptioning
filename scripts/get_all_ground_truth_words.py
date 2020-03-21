from nltk import word_tokenize
import json

a = json.load(open('coco_ground_truth_annotations.json', 'r'))
all_ground_truths = [entry['category'] for entry in a['images']]
all_words = set()
for ground_truth in all_ground_truths:
    for word in word_tokenize(ground_truth):
        all_words.add(word)