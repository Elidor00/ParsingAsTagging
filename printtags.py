#!/usr/bin/env python

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

rel_pos = []
for sentence in iter_conll(args.datafile, args.metadata, verbose=False):
    for entry in sentence:
        if entry.id > 0:
            result = []
            if args.deprel:
                result.append(entry.deprel)
            if args.pos:
                result.append(str(entry.pos))
                rel_pos.append(entry.pos)
            print(' '.join(result))
print("Max relative position: ", max(rel_pos, key=abs))
