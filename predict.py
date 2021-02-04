#!/usr/bin/env python

import argparse
import pickle
import torch
import random
import numpy as np
import importlib

from bert_features import Bert
from conll import read_conll, eval_conll, parse_conll
from modelDict import model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', help='serialized model')
parser.add_argument('test')
parser.add_argument('--test_metadata', action='store_true', help='specify if test file contains metadata or compound words')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--no-cycles', action='store_true', help='no cycles flag')
parser.add_argument('--no-cycles-strategy', default="optimal", help='what strategy to use for ensuring no cycles in output. Either greedy or optimal')
parser.add_argument('--print-nr-of-cycles', action='store_true', help='print percentage of cycles in the output')
parser.add_argument('--which-cuda', type=int, default=0, help='which cuda to use')

# Choose model
parser.add_argument('--choose-model', type=int, default=0, help='0 original model, 1 model without biLSTM and hidden layer, 2 model without only biLSTM')

args = parser.parse_args()

with open(f'{args.model}.pickle', 'rb') as f:
    params = pickle.load(f)

# Set random
random.seed(params[0].random_seed)
torch.manual_seed(params[0].random_seed)
np.random.seed(params[0].random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(params[0].random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f'loading model from {args.model}')
device = torch.device(f'cuda:{args.which_cuda}' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

print(f'loading test dataset from {args.test}')
test = read_conll(args.test, args.test_metadata, lower_case=not params[0].bert_multilingual_cased)

# Add .bert attribute to ConnlEntry if -bert flag is set @see ConllEntry
if params[0].bert:
    bert = Bert(params[0].bert_layers, params[0].bert_max_seq_length, params[0].bert_batch_size,
                params[0].bert_multilingual_cased, args.which_cuda)
    bert.extract_bert_features(test)
print(params[0])
print(args)
print('parsing test dataset')

model_type = model[args.choose_model]
model_module = importlib.import_module(model_type)
pat = model_module.Pat.load(model_module, args.model, device).to(device)
print("Model:")
print(pat)
pat.mode = 'evaluation'
pat.no_cycles = args.no_cycles
pat.no_cycles_strategy = args.no_cycles_strategy
pat.print_nr_of_cycles = args.print_nr_of_cycles
pat = pat.eval()
with torch.no_grad():
    parse_conll(pat, test, args.batch_size, clear=True)  # Clear head, deprel and pos


print('evaluating parsing results')
eval_conll(test, args.test, args.test_metadata)
