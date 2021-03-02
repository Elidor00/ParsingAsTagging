import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import first

'''
BaseModel inherited from all other models
'''


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.i = 0

    def load(self, name, device):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            print(params)
            params[0].which_cuda = device.index
            print("name: ", name)
            pat = self.Pat(*params)
            pat.load_state_dict(torch.load(f'{name}.model', map_location=device), strict=True)
        return pat

    def save(self, name):
        torch.save(self.state_dict(), f'{name}.model')
        with open(f'{name}.pickle', 'wb') as f:
            params = (self.args, self.word_vocab, self.tag_vocab, self.pos_vocab, self.deprel_vocab, self.char_vocab)
            pickle.dump(params, f)

    # receives an array of Conll objects
    # returns:
    # 1. a matrix word indexes padded
    # 2. a matrix of tag indexes padded
    # 3. an array with sentence lengths
    def sentence2tok_tags(self, sentences):
        w = [[e.norm for e in sentence] for sentence in sentences]
        w, x_lengths = self.prepare(w, self.word_vocab)
        t = [[e.get_partofspeech_tag(self.partofspeech_type) for e in sentence] for sentence in sentences]
        t, _ = self.prepare(t, self.tag_vocab)
        return w, t, x_lengths

    def train_conll(self, sentences):
        # get targets from sentences
        y1 = [[str(e.pos) for e in sentence] for sentence in sentences]
        y1, _ = self.prepare(y1, self.pos_vocab)

        y2 = [[str(e.deprel) for e in sentence] for sentence in sentences]
        y2, _ = self.prepare(y2, self.deprel_vocab)

        # flatten array
        y1 = y1.contiguous().view(-1)
        y2 = y2.contiguous().view(-1)

        # get preds from network
        y_pred1, y_pred2 = self.forward(sentences)
        # flatten preds
        y_pred1 = y_pred1.contiguous().view(-1, len(self.pos_vocab))

        y_pred2 = y_pred2.contiguous().view(-1, len(self.deprel_vocab))
        loss1 = F.cross_entropy(y_pred1, y1, ignore_index=self.pos_vocab.pad)
        loss2 = F.cross_entropy(y_pred2, y2, ignore_index=self.deprel_vocab.pad)

        return loss1 + self.loss_weight_factor * loss2

    # graph -> graph with no cycles
    # head_values -> a topk result applied on a 1d tensor. pair of original values and indices
    # (therefore it is sorted in descendent order)
    # Returns the index of first value from head_values which when added to graph doesn't add a cycle,
    # entry_id and pos_value
    def first_not_cycle(self, graph, head_values, entry_id, sentence_length):
        debug = []
        _, indices = head_values
        for value in indices:
            word = self.pos_vocab[int(value)]
            if word != '<pad>':
                copy_graph = graph.copy()
                pos_value = 0 if word == '<unk>' else int(word)
                if 0 <= entry_id + pos_value < sentence_length:
                    copy_graph.add_edge(entry_id + pos_value if pos_value != 0 else 0, entry_id)
                    if len(list(nx.simple_cycles(copy_graph))) == 0:
                        return entry_id + pos_value if pos_value != 0 else 0, entry_id, pos_value
                debug.append(
                    f"Tried to add  {entry_id + pos_value}, but it adds a cycle or is outside {entry_id + pos_value}"
                )

        # Not possible to add an edge without creating a cycle. This should not be possible
        print(graph.edges())
        print(graph.nodes)
        print(head_values)
        print(entry_id)
        print(debug)
        raise ValueError(
            "Not possible to add an edge without creating a cycle. "
            "This should not be possible, because E=V+1 (for this particular problem) "
            "and there are V^2 possible edges to add"
        )

    # Receives a matrix which is similar to an adjacency matrix, but the weights represent the probability.
    # Take the one that is most likely to be the root first
    # Returns a dictionary with the most likely tree
    def optimal_no_cycles(self, sentence, y):

        G = nx.DiGraph()
        sentence_length = len(sentence)

        for j, entry in enumerate(sentence):
            # Construct graph, ignoring pads
            # if j != root[0] and entry.id != 0: # Do not add root or fake-root yet.
            # Add every node with no root at the end
            if entry.id != 0:
                for k, probability in enumerate(y[j]):
                    word = self.pos_vocab[k]
                    if word != '<pad>':  # Skip pad
                        pos = int(word) if word != '<unk>' else 0
                        if 0 <= entry.id + pos < sentence_length:  # make sure is between limits
                            G.add_edge(entry.id + pos if pos != 0 else 0, entry.id, weight=probability)

        edmond = nx.algorithms.tree.branchings.Edmonds(G)
        result = list(edmond.find_optimum(style='arborescence', kind='max').edges())
        result = [(x[1], x[0]) for x in result]
        result = dict(result)

        return result

    # Produce the results
    # Results can be with cycles or not.
    # Two strategies for removing cycles: greedy and optimal
    def parse_conll(self, sentences):
        y1, y2 = self.forward(sentences)
        max = y1.shape[2]
        self.i = 0
        # self.no_cycles is dynamically added in predict.py. self.mode is dynamically added in predict.py or train.py
        if self.mode == 'evaluation' and self.no_cycles:
            # For each word in each line ([i][j]), use y1[i][j].topk(<length_of_tensor>)
            # which returns a pair of two tensors: sorted values and indices in original array
            y2 = torch.argmax(y2, dim=2)
            # if running on cuda, copy y from gpu memory to cpu
            if next(self.parameters()).is_cuda:
                y1 = y1.cpu()
                y2 = y2.cpu()
            y2 = y2.numpy()
            if self.no_cycles_strategy == 'greedy':
                for i, sentence in enumerate(sentences):
                    sentence_length = len(sentence)
                    G = nx.DiGraph()
                    result = defaultdict(list)
                    # iterate in descending order, from the word for which there is the most confident output
                    for index_in_array in torch.max(y1[i][:sentence_length], 1)[0].sort(descending=True)[1]:
                        entry = sentence[index_in_array]
                        values = y1[i][index_in_array].topk(max)
                        if entry.id != 0:
                            edge_no_cycle = self.first_not_cycle(G, values, entry.id, sentence_length)
                            result[int(index_in_array)] = list(edge_no_cycle)
                            G.add_edge(edge_no_cycle[0], edge_no_cycle[1])
                        else:
                            result[int(index_in_array)] = ['_', '_', '_']  # head of entry with id = 0 is 0 (itself)

                    for j, entry in enumerate(sentence):
                        deprel = self.deprel_vocab[int(y2[i, j])]
                        entry.head = result[j][0]
                        entry.pos = result[j][2]
                        entry.deprel = deprel
            elif self.no_cycles_strategy == 'optimal':  # Uses Liu-Chen-Edmonds algorithm
                for i, sentence in enumerate(sentences):
                    torch.set_printoptions(threshold=5000)
                    result = self.optimal_no_cycles(sentence, y1[i])
                    for j, entry in enumerate(sentence):
                        deprel = self.deprel_vocab[int(y2[i, j])]
                        if entry.id not in result:
                            entry.head = entry.id

                        else:
                            entry.head = result[entry.id]
                        entry.pos = '_'
                        entry.deprel = deprel

        else:

            # After it was checked that ind is not a pad
            def get_pos(ind):
                word_of_ind = self.pos_vocab[int(ind)]
                if word_of_ind != '<pad>':
                    pos_of_ind = 0 if word_of_ind == '<unk>' else int(word_of_ind)
                    return pos_of_ind

                raise ValueError("No index is valid. Maybe there is a mistake somewhere else?")

            y2 = torch.argmax(y2, dim=2)

            # if running on cuda, copy y from gpu memory to cpu
            if next(self.parameters()).is_cuda:
                y1 = y1.cpu()
                y2 = y2.cpu()

            y2 = y2.numpy()
            for i, sentence in enumerate(sentences):
                sentence_length = len(sentence)
                for j, entry in enumerate(sentence):
                    # Skip over 'fake' root
                    if entry.id != 0:
                        # Indices from biggest to smallest (total = k) from y1[i][j] (current entry)
                        y1_topk = y1[i][j].topk(max)[1]
                        # Take first index (descending order) that is not '<pad>' and inside sentence
                        index = first(y1_topk, lambda ind: self.pos_vocab[int(ind)] != '<pad>'
                                                           and (0 <= (entry.id + get_pos(ind)) < sentence_length))

                        pos = get_pos(index)
                        entry.head = entry.id + pos if pos != 0 else 0
                        entry.pos = pos
                    else:
                        entry.pos = 0
                        entry.head = 0

                    deprel = self.deprel_vocab[int(y2[i, j])]
                    entry.deprel = deprel

        # Printing the number of cycles. Only on evaluation
        if self.mode == 'evaluation':
            if self.print_nr_of_cycles:
                for sentence in sentences:
                    G = nx.DiGraph()
                    for entry in sentence:
                        if entry.id != 0:  # Might be that root.head is equal to root.id (fake root, with id=0)
                            G.add_edge(entry.head, entry.id)
                    if len(list(nx.simple_cycles(G))) > 0:
                        self.nr_of_cycles += 1

    def prepare(self, sentences, vocab):
        x = [torch.tensor([vocab[w] for w in sentence]).to(self.device) for sentence in sentences]
        x_lengths = np.array([len(sentence) for sentence in x])
        padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        return padded_x, x_lengths

    # def get_elmo_embeddings(self, orig_w):
    #    # elmo uses non-normalized words
    #    # w = [[e.form for e in sentence] for sentence in sentences]
    #    # elmo from batch/sentence to batch/character_ids
    #    elmo_ids = batch_to_ids(orig_w).to(self.device)
    #    # get elmo_embeddings
    #    elmo_embeds = self.elmo(elmo_ids)
    #    # elmo_embeds is a dict with elmo_representations and mask
    #    # first dimention contains the output of each elmo layer (in this project 1 layer)
    #    elmo_embeds = elmo_embeds['elmo_representations'][0]
    #    # (batch_size, seq_len, embedding_dim)
    #    return elmo_embeds

    def get_polyglot_embedding(self, word):
        # print("Got word: ", word, "\n")
        if word.isdigit():
            processed = self.polyglot_digit_transformer.sub('#', word)
            if processed in self.polyglot_dictionary:
                return self.polyglot_dictionary[self.polyglot_digit_transformer.sub('#', word)]
            else:
                return self.polyglot_dictionary['<UNK>']
        elif word not in self.polyglot_dictionary:
            return self.polyglot_dictionary['<UNK>']
        else:
            return self.polyglot_dictionary[word]

    def get_polyglot_embeddings(self, orig_w):
        return [[torch.tensor(self.get_polyglot_embedding(word)) for word in sentence] for sentence in orig_w]

    def forward(self, sentences):
        pass
