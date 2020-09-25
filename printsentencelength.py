#!/usr/bin/env python

# to print sentence length in UD:
#   ./printsentencelength.py --metadata ./data/UD_2.6/it/it_isdt-ud-train.conllu
#   ./printsentencelength.py --metadata data.conllu

import argparse
import matplotlib.pyplot as plt
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


sentence_length = []
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    sentence_length.append(len(sentence))

frequency = freq(sentence_length)
print(frequency)  # sentence length: number of sentences with that length
print("Max sentence length in file: ", max(sentence_length))
# x (sentence length), y (number of sentences with that length)
plt.bar(frequency.keys(), frequency.values(), width=0.5, log=True, color='g')
plt.show()

