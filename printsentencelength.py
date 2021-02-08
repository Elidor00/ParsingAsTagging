#!/usr/bin/env python

# to print sentence length in UD:
#   ./printsentencelength.py --metadata ./data/UD_2.6/it/it_isdt-ud-train.conllu
#   ./printsentencelength.py --metadata data.conllu

import argparse
import matplotlib.pyplot as plt
from numpy import mean
from conll import iter_conll

parser = argparse.ArgumentParser()
parser.add_argument('datafile')
parser.add_argument('--metadata', action='store_true', help='specify if file contains metadata or compound words')
args = parser.parse_args()


def freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return d


def calculate_stats(freq_dict: dict) -> dict:
    """
    freq: {sentence_len : number of sentence with that length}
    """
    return {"tot_sentences": sum(freq_dict.values()), "max_sentence_len": max(freq_dict.keys()),
            "mean_sentence_len": mean([int(key) for key, _ in freq_dict.items()])}


sentence_length = []
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    sentence_length.append(len(sentence))

frequency = freq(sentence_length)
print(frequency)  # sentence length: number of sentences with that length

if "train" in args.datafile:
    name = "TRAIN"
elif "dev" in args.datafile:
    name = "DEV"
else:
    name = "TEST"

fig, ax = plt.subplots()
plt.bar(frequency.keys(), frequency.values(), log=True, color='g')
plt.title('DISTRIBUTION OF SENTENCE LENGTH - ' + name)
plt.xlabel('Sentence length')
plt.ylabel('Frequency')
stats = calculate_stats(frequency)
print(stats)
# labels = list(stats.keys())
# print(labels)
# handles = [plt.Rectangle((0, 0), 1, 1, stats[label]) for label in labels]
# plt.legend(handles, labels)
ax.legend()
plt.show()
# plt.savefig(name + ".png", dpi=200, bbox_inches='tight')
