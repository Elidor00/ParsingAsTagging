#!/usr/bin/env python

# to print sentence length in UD:
#   ./printsentencelength.py --metadata ./data/UD_2.6/it/it_isdt-ud-train.conllu
#   ./printsentencelength.py --metadata data.conllu

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

frequencies = freq(sentence_length)
print(frequencies)  # sentence length: number of sentences with that length

name = "Frequency_"

if "train" in args.datafile:
    name += "TRAIN_SET"
elif "dev" in args.datafile:
    name += "DEV_SET"
else:
    name += "TEST_SET"

stats = calculate_stats(frequencies)
# print(stats)

freq_plot = sns.histplot(x=frequencies.keys(), weights=frequencies.values(), discrete=True,
                  kde=True, kde_kws={'bw_adjust': 0.2}, line_kws={'linewidth': 3}, stat="frequency")
freq_plot.margins(x=0.01)
freq_plot.set_title('DISTRIBUTION OF SENTENCE LENGTHS - ' + name.split("_")[1] + " SET")
plt.xlabel('Sentence length')
plt.show()
fig = freq_plot.get_figure()

if not os.path.exists("./img"):
    os.mkdir("./img")

fig.savefig(os.path.join("./img", name + ".png"), bbox_inches='tight')
