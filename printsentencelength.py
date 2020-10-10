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

# TODO: the idea could be:
#        - get the total number of sentences
#        - divide by 3
#        - add sentences to a range until the number of added sentences is >= to the number of sentences / 3
# def create_length_range(r1, r2, freq_dict):
#     first_range = 0
#     second_range = 0
#     third_range = 0
#     for k in freq_dict:
#         if k in r1:
#             first_range += freq_dict[k]
#         elif k in r2:
#             second_range += freq_dict[k]
#         else:
#             third_range += freq_dict[k]
#     return first_range, second_range, third_range


sentence_length = []
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    sentence_length.append(len(sentence))

frequency = freq(sentence_length)
print(frequency)  # sentence length: number of sentences with that length
total_sentences = sum(frequency.values())
print("Total sentences of every length in dataset: ", total_sentences)
print("Max sentence length in file: ", max(sentence_length))
# x (sentence length), y (number of sentences with that length)
plt.bar(frequency.keys(), frequency.values(), width=0.5, log=True, color='g')
# range0_13, range14_25, range30_81 = create_length_range(range(0, 14), range(14, 26), frequency)
# assert range0_13 + range14_25 + range30_81 == total_sentences
# print(range0_13, range14_25, range30_81)
plt.show()
