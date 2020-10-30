import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence

from packed_lstm import PackedLSTM
from utils import tensor_unsort


class CharacterModel(nn.Module):
    def __init__(self, args, word_vocab, sentences, pad=False, bidirectional=False, attention=False):
        super().__init__()
        self.args = args
        self.pad = pad
        self.word_vocab = word_vocab
        self.sentences = sentences
        self.num_dir = 2 if bidirectional else 1
        self.attn = attention


        # char embeddings
        self.char_emb = nn.Embedding(len(self.word_vocab), 100, padding_idx=0)
        if self.attn:
            self.char_attn = nn.Linear(self.num_dir * 400, 1, bias=False)
            self.char_attn.weight.data.zero_()

        # modules
        # self.charlstm = nn.LSTM(100, 400,
        #                         1, batch_first=True,
        #                         dropout=0 if 1 == 1 else None,
        #                         bidirectional=bidirectional)
        self.charlstm = PackedLSTM(100, 400, 1, batch_first=True,
                dropout=0, rec_dropout=0, bidirectional=False)

        self.charlstm_h_init = nn.Parameter(
            torch.zeros(self.num_dir * 1, 1, 400))
        self.charlstm_c_init = nn.Parameter(
            torch.zeros(self.num_dir * 1, 1, 400))

        self.dropout = nn.Dropout(0)

    def forward(self, word_vocab, sentences):
        wordlens = [len(s) for s in sentences]
        embs = self.dropout(self.char_emb(word_vocab))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, wordlens, batch_first=True)
        output = self.charlstm(embs, wordlens, hx=(
            self.charlstm_h_init.expand(self.num_dir * 1, batch_size,
                                        400).contiguous(),
            self.charlstm_c_init.expand(self.num_dir * 1, batch_size,
                                        400).contiguous()))

        # apply attention, otherwise take final states
        if self.attn:
            char_reps = output[0]
            weights = torch.sigmoid(self.char_attn(self.dropout(char_reps.data)))
            char_reps = PackedSequence(char_reps.data * weights, char_reps.batch_sizes)
            char_reps, _ = pad_packed_sequence(char_reps, batch_first=True)
            res = char_reps.sum(1)
        else:
            h, c = output[1]
            res = h[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        # recover character order and word separation
        sentlens = [len(s) for s in sentences]
        word_orig_idx = [word.index(word) for word in self.word_vocab]

        res = tensor_unsort(res, word_orig_idx[:batch_size])
        word_orig_idx = word_orig_idx[batch_size:]
        res = pack_sequence(res.split(sentlens[0]))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]

        return res
