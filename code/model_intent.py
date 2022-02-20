import torch
import torch.nn as nn
import numpy as np
import math
# import model_encoder as encoder
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class att_Generator(nn.Module):
    def __init__(self,config, d_emb, lstm_output_dim ,att_lstm_hsize, embedding=None):
        super(att_Generator, self).__init__()
        self.lstm_hidden_size = lstm_output_dim
        self.word_emb_size = d_emb
        self.lstm_nlayers = 1
        self.d_a = 10
        self.r = 1
        self.cuda_id = 0
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.atttolstm =  config['atttolstm']
        if self.atttolstm == True:
            self.att_lstm_input = 3
        else:
            self.att_lstm_input = 2
        self.att_lstm_hsize = att_lstm_hsize
        self.att_lstm_nlayers = 1
        self.dropout_weight = 0 #0.7 #0
        self.dropout = nn.Dropout(self.dropout_weight)

        self.bilstm = nn.LSTM(self.word_emb_size, self.lstm_hidden_size,
                              self.lstm_nlayers, bidirectional=True, batch_first=True)

        self.att_bilstm = nn.LSTM(self.att_lstm_input, self.att_lstm_hsize,
                              self.att_lstm_nlayers, bidirectional=True, batch_first=True)
        self.att_line=nn.Linear(self.att_lstm_hsize*2, 1, bias=False)

        self.ws1 = nn.Linear(self.lstm_hidden_size * 2, self.d_a, bias=False)
        self.ws2 = nn.Linear(self.d_a, self.r, bias=False)

        self.watt = nn.Linear(2, 1, bias=False)

        if torch.cuda.is_available():
            self.ws1 = self.ws1.cuda(self.cuda_id)
            self.ws2 = self.ws2.cuda(self.cuda_id)
            self.watt = self.watt.cuda(self.cuda_id)

        self.softmax = nn.Softmax(dim = 1)

        self.ifDS = config['ifDS']


    def sort_batch(self, batch_x, batch_len):
        batch_len_new = batch_len
        batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
        batch_x_new = batch_x[perm_idx]
        return batch_x_new, batch_len_new, perm_idx

    def forward(self,  outp,len, embedding ,train_iwf,train_st,ifattention = True):

        if ifattention == True:
            hbar = self.relu(self.ws1(outp))
            hbar = self.dropout(hbar)
            alphas = F.softmax(self.ws2(hbar), dim=1)

        train_iwf = train_iwf
        train_st = train_st


        if self.atttolstm == False:
            att2 = torch.stack([train_iwf,train_st], dim=2, out=None).squeeze(3)
        else:
            newzero = torch.zeros(train_iwf.size())
            newzero[:, :alphas.size()[1]] = alphas
            att2 = torch.stack([train_iwf,train_st,newzero], dim=2, out=None).squeeze(3)
        sort_att2,sort_att2_len,sort_att2_idx = self.sort_batch(att2, len)
        sort_att2_emb = sort_att2.transpose(0, 1)
        sort_att2_emb = pack_padded_sequence(sort_att2_emb, sort_att2_len)
        sort_att2_emb = self.att_bilstm(sort_att2_emb.cuda())[0]
        sort_att2_emb = pad_packed_sequence(sort_att2_emb)[0].transpose(0, 1).contiguous()

        ori, newidx = sort_att2_idx.sort(0)
        sort_att2_emb = sort_att2_emb[newidx]

        att2 = self.att_line(sort_att2_emb)
        att2 = self.softmax(att2)


        self.st = True
        if self.atttolstm == False:
            if ifattention == True:
                if self.st == False:
                    train_iwf = train_iwf[:,:att2.size()[1],:].cuda()
                    fina_att = torch.cat((train_iwf, att2), dim=2)
                    fina_att = self.watt(fina_att)
                    # fina_att = (train_iwf + att2) / 2
                elif self.ifDS == False:
                    fina_att = alphas
                else:
                    # fina_att = alphas
                    fina_att = torch.cat((alphas,att2),dim=2)
                    fina_att = self.watt(fina_att)
            else:
                fina_att = att2
        else:
            fina_att = att2
        final_emb = outp.transpose(1,2) @ fina_att
        final_emb = final_emb.transpose(1,2)


        return  final_emb

class lstm(nn.Module):
    def __init__(self, d_emb, lstm_output_dim , embedding=None):
        super(lstm, self).__init__()
        self.lstm_hidden_size = lstm_output_dim
        self.word_emb_size = d_emb
        self.lstm_nlayers = 3
        self.d_a = 10
        self.r = 1
        self.cuda_id = 0
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.att_lstm_input = 2
        self.att_lstm_hsize = 1
        self.att_lstm_nlayers = 1

        self.bilstm = nn.LSTM(self.word_emb_size, self.lstm_hidden_size,
                              self.lstm_nlayers, bidirectional=True, batch_first=True)

        self.att_bilstm = nn.LSTM(self.att_lstm_input, self.att_lstm_hsize,
                              self.att_lstm_nlayers, bidirectional=True, batch_first=True)
        self.att_line=nn.Linear(self.att_lstm_hsize*2, 1, bias=False)

        self.ws1 = nn.Linear(self.lstm_hidden_size * 2, self.d_a, bias=False)
        self.ws2 = nn.Linear(self.d_a, self.r, bias=False)

        if torch.cuda.is_available():
            self.ws1 = self.ws1.cuda(self.cuda_id)
            self.ws2 = self.ws2.cuda(self.cuda_id)


    def sort_batch(self, batch_x, batch_len):
        batch_len_new = batch_len
        batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
        batch_x_new = batch_x[perm_idx]
        return batch_x_new, batch_len_new, perm_idx

    def forward(self, pre_emb ,len):
        sort_emb, sort_len, idx = self.sort_batch(pre_emb, len)

        sort_emb = sort_emb.transpose(0, 1)
        emb = pack_padded_sequence(sort_emb, sort_len.cpu())
        emb = self.bilstm(emb.cuda())[0]
        outp = pad_packed_sequence(emb)[0].transpose(0, 1).contiguous()

        ori, newidx = idx.sort(0)
        outp = outp[newidx]

        return outp

class Generator_stov(nn.Module):
    def __init__(self, config,d_emb,lstm_output_dim,embedding=None):
        super(Generator_stov,self).__init__()
        self.input = d_emb
        self.hsize = config['hsize']
        self.output = lstm_output_dim
        self.drop_weight = 0.3
        self.cuda_id = 0

        self.dense1 = nn.Linear(self.input, self.hsize, bias=False)
        self.dense3 = nn.Linear(self.hsize, self.output*2, bias=False)
        self.drop = nn.Dropout(self.drop_weight)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input):

        emb = self.tanh(self.dense1(input.cuda(self.cuda_id)))
        emb = self.drop(emb)
        emb = self.tanh(self.dense3(emb))
        return emb



