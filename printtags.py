#!/usr/bin/env python

# base use
#   python printtags.py data/UD_2.6/it/it_isdt-ud-test.conllu

# to count tag combinations:
#   ./printtags.py ~/data/pat/train.conllu | sort | uniq -c | sort -n

import argparse
from conll import iter_conll

parser = argparse.ArgumentParser()
parser.add_argument('datafile')
parser.add_argument('--no-deprel', dest='deprel', action='store_false', help="don't print dependencies")
parser.add_argument('--no-pos', dest='pos', action='store_false', help="don't print relative positions")
parser.add_argument('--metadata', action='store_true', help='specify if file contains metadata or compound words')
args = parser.parse_args()

total = 0
rel_pos = []
out_of_range = 0
punct = 0
left_threshold = -50
right_threshold = 50
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    for entry in sentence:
        if entry.id > 0:
            total += 1
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
print("Total number of token: ", total)
print("N. relative position out of empirical range: ", out_of_range, " = ", round((out_of_range / total) * 100, 2), "%")
print("N. of punct deprel tag: ", punct)
