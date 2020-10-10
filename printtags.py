#!/usr/bin/env python

# base use
#   python printtags.py data/UD_2.6/it/it_isdt-ud-test.conllu

# to count tag combinations:
#   ./printtags.py ~/data/pat/train.conllu | sort | uniq -c | sort -n

import argparse
from conll import iter_conll, write_conll_single

parser = argparse.ArgumentParser()
parser.add_argument('datafile')
parser.add_argument('--no-deprel', dest='deprel', action='store_false', help="don't print dependencies")
parser.add_argument('--no-pos', dest='pos', action='store_false', help="don't print relative positions")
parser.add_argument('--metadata', action='store_true', help='specify if file contains metadata or compound words')
args = parser.parse_args()

rel_pos = []
punct = 0
out_of_range = 0
left_threshold = -50
right_threshold = 50
first_range_sentences = []
second_range_sentences = []
third_range_sentences = []
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    # for each sentence (max sentence length in dev data is equal to 81)
    # calculates the percentage of punctuation marks
    # the ranges are chosen in a empirical way (see printsentencelength.py)
    if len(sentence) in range(0, 14):
        first_range_sentences.append(sentence)
    elif len(sentence) in range(14, 26):
        second_range_sentences.append(sentence)
    else:
        third_range_sentences.append(sentence)

    for entry in sentence:
        if entry.id > 0:
            result = []
            if args.deprel:
                result.append(entry.deprel)
                if entry.deprel == "punct":
                    punct += 1
            if args.pos:
                result.append(str(entry.pos))
                rel_pos.append(entry.pos)
                if entry.pos <= left_threshold or entry.pos >= right_threshold:  # empirical range [-50, 50]
                    out_of_range += 1
            print(' '.join(result))
# print("Max relative position: ", max(rel_pos, key=abs))
print("N. relative position out of empirical range: ", out_of_range)
print("N. of punct deprel tag: ", punct)

tot_0_9 = 0
tot_10_24 = 0
tot_25 = 0
for sentence in first_range_sentences:
    p = 0
    for entry in sentence:
        if entry.id > 0:
            if entry.deprel == "punct":
                p += 1
    print("p: ", p)
    print("len: ", len(sentence))
    print(int((p / len(sentence))*100))
    if int((p / len(sentence))*100) in range(0, 10):
        tot_0_9 += 1
        write_conll_single("TMP0_9", sentence)
    elif int((p / len(sentence))*100) in range(10, 25):
        tot_10_24 += 1
        write_conll_single("TMP10_24", sentence)
    else:
        tot_25 += 1
        write_conll_single("TMP25", sentence)
print("0 - 9: ", tot_0_9)
print("10 - 24: ", tot_10_24)
print(" > 25: ", tot_25)



