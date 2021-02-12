import pickle
import re

from torch import nn
from torch.nn import functional as F

from baseModel import BaseModel
from bert_features import from_tensor_list_to_one_tensor
from char_embeddings import CharEmbeddings, CNNCharEmbeddings
from embeddings import *
from positional_embeddings import PositionalEmbeddings
from positional_encoding import PositionalEncoding

'''
Model for UmBERTo features extraction to comparing with fine tuning.
This model consists of:
- bert embeddings (feature extraction)
- bilstm
- 2 linear layer for classification 
'''


class Pat(BaseModel):

    def __init__(self, args, word_vocab, tag_vocab, pos_vocab, deprel_vocab, char_vocab):
        super().__init__()

        self.args = args
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.pos_vocab = pos_vocab
        self.deprel_vocab = deprel_vocab
        self.char_vocab = char_vocab

        self.device = torch.device(f'cuda:{args.which_cuda}' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        self.glove_emb = args.glove_emb
        self.word_emb_size = args.word_emb_size
        self.tag_emb_size = args.tag_emb_size
        self.bilstm_hidden_size = args.bilstm_hidden_size
        self.bilstm_num_layers = args.bilstm_num_layers

        self.mlp_output_size = args.mlp_output_size

        self.bilstm_input_size = 0  # self.word_emb_size  # + self.tag_emb_size
        # char embeddings
        self.char_emb = args.char_emb
        self.char_emb_hidden_size = args.char_emb_hidden_size
        self.char_emb_size = args.char_emb_size
        # elmo
        # self.elmo_opts = args.elmo_opts
        # self.elmo_weights = args.elmo_weights

        # position embeddings
        self.position_emb_max_pos = len(self.word_vocab)
        self.position_emb = args.position_emb
        self.position_emb_size = args.position_emb_size

        # position encoding
        self.position_enc_max_pos = args.word_emb_size  # word emb size = 100
        self.position_enc = args.position_enc
        # self.position_enc_size = args.position_enc_size

        # bert
        self.bert = args.bert
        self.bert_hidden_size = args.bert_hidden_size
        # polyglot embeddings
        self.polyglot = args.polyglot
        self.polyglot_size = 64  # pretrained model has standard length 64 (no other variants)
        # cnn char encoding
        self.cnn_ce = args.cnn_ce
        self.cnn_embeddings_size = args.cnn_embeddings_size
        self.cnn_ce_kernel_size = args.cnn_ce_kernel_size
        self.cnn_ce_out_channels = args.cnn_ce_out_channels

        # Use head for predicting label
        self.use_head = args.use_head

        # dropout
        self.dropout = nn.Dropout(p=args.dropout)
        self.bilstm_dropout = args.bilstm_dropout

        self.mlp_hidden_size = args.mlp_hidden_size

        self.partofspeech_type = args.part_of_speech

        self.nr_of_cycles = 0

        self.loss_weight_factor = args.loss_weight_factor

        if self.polyglot:
            # Load embeddings
            words, embeddings = pickle.load(open(self.polyglot, 'rb'), encoding='latin1')

            # build a dictionary for fast access
            self.polyglot_dictionary = {}
            for i, w in enumerate(words):
                self.polyglot_dictionary[w] = embeddings[i]

            # Digits are replaced with # in this embedding
            self.polyglot_digit_transformer = re.compile('[0-9]', re.UNICODE)

            # Increase input size accordingly
            self.bilstm_input_size += self.polyglot_size

        if self.bert:
            self.bilstm_input_size = self.bilstm_input_size + self.bert_hidden_size

        if self.position_emb:
            self.bilstm_input_size = self.bilstm_input_size + self.position_emb_size

        # sum output of char embedding to bilstm
        # it is *2 bec a bilstm is used
        if self.char_emb:
            self.bilstm_input_size = self.bilstm_input_size + self.char_emb_hidden_size * 2

        if self.cnn_ce:
            self.bilstm_input_size += self.cnn_ce_out_channels

        # if elmo files are set
        # if self.elmo_opts:
        #    print('using elmo')
        #    self.elmo = Elmo(
        #        self.elmo_opts,
        #        self.elmo_weights,
        #        num_output_representations=1,
        #        dropout=0
        #    ).to(self.device)
        #    # increace size of embedding with elmo's size
        #    self.bilstm_input_size = self.bilstm_input_size + self.elmo.get_output_dim()

        '''
        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.word_vocab),
            embedding_dim=self.word_emb_size,
            padding_idx=self.word_vocab.pad,
        )
        '''

        if self.position_enc:
            self.positional_encoding = PositionalEncoding(
                d_model=self.position_enc_max_pos
            ).to(self.device)

        if self.position_emb:
            self.positional_embedding = PositionalEmbeddings(
                emb_size=self.position_emb_size,
                max_position=self.position_emb_max_pos,
                pad_index=self.word_vocab.pad
            ).to(self.device)

        if self.char_emb:
            self.char_embedding = CharEmbeddings(
                char_vocab=self.char_vocab,
                embedding_dim=self.char_emb_size,
                hidden_size=self.char_emb_hidden_size,
                num_layers=1,
                attention=True,
                bidirectional=True
            ).to(self.device)

        if self.cnn_ce:
            self.cnn_char_embedding = CNNCharEmbeddings(
                char_vocab=self.char_vocab,
                cnn_embeddings_size=self.cnn_embeddings_size,
                cnn_ce_kernel_size=self.cnn_ce_kernel_size,
                cnn_ce_out_channels=self.cnn_ce_out_channels,
                which_cuda=args.which_cuda
            ).to(self.device)

        # if glove is defined
        if self.glove_emb:
            # load glove
            glove = load_glove(self.glove_emb)
            # convert to matrix
            glove_weights = from_vocab_to_weight_matrix(self.word_vocab, glove)
            # load matrix into embeds
            self.word_embedding.load_state_dict({'weight': glove_weights})

        '''
        self.tag_embedding = nn.Embedding(
            num_embeddings=len(self.tag_vocab),
            embedding_dim=self.tag_emb_size,  # vector dimension for tag embedding
            padding_idx=self.tag_vocab.pad,
        )
        '''

        self.bilstm = nn.LSTM(
            input_size=self.bilstm_input_size,
            hidden_size=self.bilstm_hidden_size,
            num_layers=self.bilstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.bilstm_dropout
        )

        '''
        self.bilstm_to_hidden1 = nn.Linear(
            in_features=self.bilstm_hidden_size * 2,
            out_features=self.mlp_hidden_size,
        )

        self.hidden1_to_hidden2 = nn.Linear(
            in_features=self.mlp_hidden_size,
            out_features=self.mlp_output_size,
        )
        '''

        self.hidden2_to_pos = nn.Linear(
            in_features=self.bilstm_hidden_size * 2,
            out_features=len(self.pos_vocab),
        )

        self.hidden2_to_dep = nn.Linear(
            in_features=self.bilstm_hidden_size * 4 if self.use_head else self.bilstm_hidden_size * 2,
            # Depending on whether the head is used or not
            out_features=len(self.deprel_vocab),
        )

        # init embedding weights only if glove is not defined
        # if self.glove_emb is None:
        #   nn.init.xavier_normal_(self.word_embedding.weight)
        # nn.init.xavier_normal_(self.tag_embedding.weight)
        # nn.init.xavier_normal_(self.bilstm_to_hidden1.weight)
        # nn.init.xavier_normal_(self.hidden1_to_hidden2.weight)
        nn.init.xavier_normal_(self.hidden2_to_pos.weight)
        nn.init.xavier_normal_(self.hidden2_to_dep.weight)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sentences):
        orig_w = [[e.form for e in sentence] for sentence in sentences]  # all token from a given sentence
        # print("token: " + str(orig_w))
        w, t, x_lengths = self.sentence2tok_tags(sentences)

        batch_size, seq_len = w.size()
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        # we = self.word_embedding(w)
        # t = self.tag_embedding(t)

        if self.bert:
            # get bert features from model
            bert_features_list = [[e.bert for e in sentence] for sentence in sentences]
            # convert list to one tensor
            bert_features_tensor = from_tensor_list_to_one_tensor(bert_features_list, self.bert_hidden_size).to(
                self.device)
            # concat the tensor with the the rest of the word embeddings
            # we = torch.cat((bert_features_tensor, we), 2)
            x = bert_features_tensor

        if self.position_emb:
            # get positional embeddings
            # print(w.min(), w.max()) # args.position_emb_size = w.max()
            position = self.positional_embedding(w)
            # concat positional embeddings with word embeddings

            # we = torch.cat((position, we), 2)  # raise index error -> print(w.min(), w.max())

        # concat tags embeddings and word embeddings
        # x = torch.cat((we, t), 2)

        # if self.elmo_opts:
        #    elmo_embeds = self.get_elmo_embeddings(orig_w)
        #    x = torch.cat([x, elmo_embeds], 2)

        if self.char_emb:
            c = self.char_embedding(orig_w)  # orig_w batch (list) of sentences
            x = torch.cat([x, c], 2)

        if self.cnn_ce:
            c = self.cnn_char_embedding(orig_w)
            x = torch.cat([x, c], 2)

        if self.polyglot:
            polyglot_features_list = self.get_polyglot_embeddings(orig_w)
            polyglot_features_tensor = from_tensor_list_to_one_tensor(polyglot_features_list, self.polyglot_size) \
                .to(self.device)
            x = torch.cat([x, polyglot_features_tensor], 2)

        # Error in local because there is no BERT (no space in my GPU)
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, n_lstm_units)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x, _ = self.bilstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # (batch_size, seq_len, n_lstm_units) -> (batch_size * seq_len, n_lstm_units)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        y1 = self.hidden2_to_pos(x)
        if self.mode == 'training':
            if self.use_head:
                heads = [[e.head for e in sentence] for sentence in sentences]
                maximum = max([len(z) for z in heads])
                heads = [z + (maximum - len(z)) * [0] for z in heads]  # pads
                heads = torch.tensor(heads)

                heads = heads.view(1, -1)[0].to(self.device)

                # each offset is of length (seq_len) in the end
                # Creates offsets: [0,0,0..,0], [seq_length, seq_length, .., seq_length], [2*seq_length, 2*seq_length, .., 2*seq_length] etc that are used for fast access in a tensor of shape (batch_size * seq_length, pos_hidden_size)
                offsets = (torch.arange(batch_size).repeat(seq_len).view(seq_len, batch_size).transpose(0, 1) * seq_len).contiguous().view( 1, -1)[0]\
                    .to(self.device)
                indices = heads + offsets

                heads = x[indices]

        elif self.mode == 'evaluation':
            if self.use_head:
                # (batch_size * seq_len, n_tags)
                ids = [[e.id for e in sentence] for sentence in sentences]
                maximum = max([len(z) for z in ids])
                ids = [z + (maximum - len(z)) * [0] for z in ids]
                ids = torch.tensor(ids).to(self.device)
                ids = ids.view(1, -1)[0]

                heads = torch.zeros(ids.shape[0]).long().to(self.device)

                maxes = torch.argmax(y1, dim=1)

                offsets = (torch.arange(batch_size).repeat(seq_len).view(seq_len, batch_size).transpose(0, 1) * seq_len).contiguous().view(1, -1)[0]\
                    .to(self.device)
                for i in range(heads.shape[0]):
                    if ids[i] != 0:
                        word = self.pos_vocab[int(maxes[i])]
                        pos = 0 if word == '<unk>' else int(word)
                        heads[i] = (
                            0 if pos + ids[i] > maximum else torch.clamp(pos + ids[i], min=0)) if pos != 0 else 0
                    else:
                        heads[i] = 0
                indices = heads + offsets
                heads = x[indices]
        else:
            exit("Unknown mode")

        if self.use_head:
            x = torch.cat([x, heads], 1)

        y2 = self.hidden2_to_dep(x)

        if self.mode == 'evaluation':
            y1 = F.softmax(y1, dim=1)
            y2 = F.softmax(y2, dim=1)

        # (batch_size * seq_len, n_lstm_units) -> (batch_size, seq_len, n_tags)
        y1 = y1.view(batch_size, seq_len, len(self.pos_vocab))
        y2 = y2.view(batch_size, seq_len, len(self.deprel_vocab))

        return y1, y2
