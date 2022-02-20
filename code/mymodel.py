import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from torch import nn,optim,autograd

import torch.nn.functional as F
import torch
# from config import *
import model_intent as model_out
# from tsne import tsne_class, tsne_outlier
from tool import parse_sklearn_log
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt

import random
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def get_meta_data(device,data,config):
    int_num = config['meta_test_num']
    seenclass = data['seen_class']
    random.shuffle(seenclass)
    data['meta_test_class']=seenclass[:int_num]
    data['meta_train_class']=[x for x in seenclass if x not in data['meta_test_class']]
    data['meta_test_class_list']=[]
    data['meta_train_class_list']=[]
    data['meta_test_ind']=[]
    data['meta_train_ind']=[]
    iwf_test = True
    meta_refine = False   
    meta_trinte = 0.5  #0.3
    flag = 0
    for i in data['meta_test_class']:
        data['meta_test_class_list'].append(data['sc_dict'][i[0]])
        d = (data['y_tr'] == data['sc_dict'][i[0]]).nonzero()
        index = [i for i in range(d.size()[0])]
        random.shuffle(index)
        d = d[index]
        if(flag == 0):
            data['meta_test_sample_id']=d
            flag = 1
        else:
            data['meta_test_sample_id'] = torch.cat((data['meta_test_sample_id'],d),0)

    if meta_refine == True:
        flag = 0
        for i in data['meta_train_class']:
            data['meta_train_class_list'].append(data['sc_dict'][i[0]])
            d = (data['y_tr'] == data['sc_dict'][i[0]]).nonzero()
            index = [i for i in range(d.size()[0])]
            random.shuffle(index)
            d = d[index]
            m = d[:int(meta_trinte*d.size()[0])]
            d = d[int(meta_trinte*d.size()[0]):]
            data['meta_test_sample_id'] = torch.cat((data['meta_test_sample_id'], m), 0)
            if (flag == 0):
                data['meta_train_sample_id'] = d
                flag = 1
            else:
                data['meta_train_sample_id'] = torch.cat((data['meta_train_sample_id'], d), 0)

    else:
        flag = 0
        for i in data['meta_train_class']:
            data['meta_train_class_list'].append(data['sc_dict'][i[0]])
            d = (data['y_tr'] == data['sc_dict'][i[0]]).nonzero()
            index = [i for i in range(d.size()[0])]
            random.shuffle(index)
            d = d[index]
            if(flag == 0):
                data['meta_train_sample_id'] = d
                flag = 1
            else:
                data['meta_train_sample_id'] = torch.cat((data['meta_train_sample_id'],d),0)
    index = [i for i in range(data['meta_train_sample_id'].size()[0])]
    random.shuffle(index)
    data['meta_train_sample_id'] = data['meta_train_sample_id'][index]
    index = [i for i in range(data['meta_test_sample_id'].size()[0])]
    random.shuffle(index)
    data['meta_test_sample_id'] = data['meta_test_sample_id'][index]

    data['meta_test_sample'] = data['x_tr'][data['meta_test_sample_id']].squeeze(dim=1)
    data['meta_train_sample'] = data['x_tr'][data['meta_train_sample_id']].squeeze(dim=1)
    data['y_ind'] = torch.zeros(data['x_tr'].size()[0], len(seenclass)).scatter_(1, data['y_tr'].unsqueeze(1), 1)
    data['meta_test_ind'] = data['y_ind'][data['meta_test_sample_id']].squeeze(dim=1)
    data['meta_train_ind'] = data['y_ind'][data['meta_train_sample_id']].squeeze(dim=1)
    data['meta_train_label'] = data['y_tr'][data['meta_train_sample_id']].squeeze(dim=1)
    data['meta_test_label'] = data['y_tr'][data['meta_test_sample_id']].squeeze(dim=1)
    data['meta_train_len'] = torch.tensor(data['s_len'][data['meta_train_sample_id']]).squeeze(dim=1)
    data['meta_test_len'] = torch.tensor(data['s_len'][data['meta_test_sample_id']]).squeeze(dim=1)
    print("get meta batch")
    bertsize=30000

    data['n_t'] = {}
    length = data['meta_train_sample'].size()[0]
    for i in range(0, length):
        idx, counts = np.unique(data['meta_train_sample'][i], return_counts=True)
        if data['meta_train_label'][i].item() not in data['n_t']:
            if config['ctxEmb']=='bert':
                data['n_t'][data['meta_train_label'][i].item()] = np.zeros(bertsize, dtype=np.float32)
            else:
                data['n_t'][data['meta_train_label'][i].item()] = np.zeros(config['n_vocab'], dtype=np.float32)
        data['n_t'][data['meta_train_label'][i].item()][idx] += counts

    length = data['meta_test_sample'].size()[0]
    for i in range(0, length):
        idx, counts = np.unique(data['meta_test_sample'][i], return_counts=True)
        if data['meta_test_label'][i].item() not in data['n_t']:
            # print(data['meta_test_label'][i].item())
            if config['ctxEmb']=='bert':
                data['n_t'][data['meta_test_label'][i].item()] = np.zeros(bertsize, dtype=np.float32)
            else:
                data['n_t'][data['meta_test_label'][i].item()] = np.zeros(config['n_vocab'], dtype=np.float32)
        data['n_t'][data['meta_test_label'][i].item()][idx] += counts

    if iwf_test==True:
        length = data['x_te'].size()[0]
        for i in range(0, length):
            idx, counts = np.unique(data['x_te'][i], return_counts=True)
            if config['ctxEmb']=='bert':
                data['n_t'][data['y_te'][i].item()] = np.zeros(bertsize, dtype=np.float32)
            else:
                data['n_t'][data['y_te'][i].item()] = np.zeros(config['n_vocab'], dtype=np.float32)
            data['n_t'][data['y_te'][i].item()][idx] += counts

    # if classes is None:
    classes = np.unique(data['y_tr'])
    if config['ctxEmb'] == 'bert':
        n_tokens = np.zeros((len(classes), bertsize), dtype=np.float32)
    else:
        n_tokens = np.zeros((len(classes), config['n_vocab']), dtype=np.float32)

    for i, key in enumerate(classes):
        n_tokens[i, :] = data['n_t'][key]


    n_tokens_sum = np.sum(n_tokens, axis=0, keepdims=True)
    n_tokens_sum[0][0] = 0
    n_total = np.sum(n_tokens_sum)

    p_t = n_tokens_sum / n_total

    # compute iwf
    iwf = 1e-5 / (1e-5 + p_t)
    iwf = np.transpose(iwf)

    sample_emb = data['x_tr'].squeeze(dim=1)
    sample_label = data['y_ind'].squeeze(dim=1)
    lamda = 1

    ebd2 = torch.tensor(data['sc_vec'])
    train_I = torch.eye(ebd2.size()[0])
    data['sc_ind'] = torch.eye(ebd2.size()[0])
    num_use = torch.sum(torch.abs(sample_emb)> 0, dim=1)

    w = ebd2.t() \
        @ torch.inverse(ebd2 @ ebd2.t() + 1 * train_I) @ data['sc_ind']

    ebd_unseen = torch.tensor(data['uc_vec'])
    train_I_unseen = torch.eye(ebd_unseen.size()[0])
    data['uc_ind'] = torch.eye(ebd_unseen.size()[0])
    # num_use = torch.sum(torch.abs(sample_emb)> 0, dim=1)

    un_w = ebd_unseen.t() \
        @ torch.inverse(ebd_unseen @ ebd_unseen.t() + 1 * train_I_unseen) @ data['uc_ind']


    if config['ctxEmb'] == 'bert':
        w2=0
        ebd=0
    else:
        data['embedding'] = torch.Tensor(data['embedding'])
        vab =  torch.zeros(data['embedding'].size())
        vab[1:] = data['embedding'][1:]
        train_ebd = vab[sample_emb]
        train_I = torch.eye(data['x_tr'].size()[0])
        train_avg_sum = torch.sum(train_ebd, dim=1)
        num_use = num_use.unsqueeze(dim=1)
        ebd = train_avg_sum/torch.tensor(data['s_len']).unsqueeze(dim=1)
        w2 = ebd.t() \
            @ torch.inverse(ebd @ ebd.t() + 1 * train_I) @ sample_label
            # @ data['meta_train_ind']

    iwf[0] = 0
    train_iwf = torch.tensor(iwf[data['meta_train_sample']]).to(device)
    test_iwf = torch.tensor(iwf[data['meta_test_sample']]).to(device)
    realtest_iwf = torch.tensor(iwf[data['x_te']]).to(device)
    # train_st = st[data['meta_train_sample_id']]
    # test_st = st[data['meta_test_sample_id']]
    print("got")
    return train_iwf,test_iwf,train_iwf.size()[0],ebd,realtest_iwf,w,w2,un_w


def f_get_meta_data(device,data,config):
    int_num = config['meta_test_num']
    seenclass = data['seen_class']
    random.shuffle(seenclass)
    data['meta_train_class']=[x for x in seenclass]
    data['meta_train_class_list']=[]
    data['meta_train_ind']=[]
    iwf_test = False
    flag = 0
    for i in data['meta_train_class']:
        data['meta_train_class_list'].append(data['sc_dict'][i[0]])
        d = (data['y_tr'] == data['sc_dict'][i[0]]).nonzero()
        index = [i for i in range(d.size()[0])]
        random.shuffle(index)
        d = d[index]
        if(flag == 0):
            data['meta_train_sample_id'] = d
            flag = 1
        else:
            data['meta_train_sample_id'] = torch.cat((data['meta_train_sample_id'],d),0)
    index = [i for i in range(data['meta_train_sample_id'].size()[0])]
    random.shuffle(index)
    data['meta_train_sample_id'] = data['meta_train_sample_id'][index]

    data['meta_train_sample'] = data['x_tr'][data['meta_train_sample_id']].squeeze(dim=1)
    data['y_ind'] = torch.zeros(data['x_tr'].size()[0], len(seenclass)).scatter_(1, data['y_tr'].unsqueeze(1), 1)
    data['meta_train_ind'] = data['y_ind'][data['meta_train_sample_id']].squeeze(dim=1)
    data['meta_train_label'] = data['y_tr'][data['meta_train_sample_id']].squeeze(dim=1)
    data['meta_train_len'] = torch.tensor(data['s_len'][data['meta_train_sample_id']]).squeeze(dim=1)
    print("get meta batch")
    bertsize = 15000

    data['n_t'] = {}
    length = data['meta_train_sample'].size()[0]
    for i in range(0, length):
        idx, counts = np.unique(data['meta_train_sample'][i], return_counts=True)
        if data['meta_train_label'][i].item() not in data['n_t']:
            # print(data['meta_train_label'][i].item())
            if config['ctxEmb']=='bert':
                data['n_t'][data['meta_train_label'][i].item()] = np.zeros(bertsize, dtype=np.float32)
            else:
                data['n_t'][data['meta_train_label'][i].item()] = np.zeros(config['n_vocab'], dtype=np.float32)
        data['n_t'][data['meta_train_label'][i].item()][idx] += counts


    if iwf_test==True:
        length = data['x_te'].size()[0]
        for i in range(0, length):
            idx, counts = np.unique(data['x_te'][i], return_counts=True)
            if data['y_te'][i].item() not in data['n_t']:
                if config['ctxEmb'] == 'bert':
                    data['n_t'][data['y_te'][i].item()] = np.zeros(bertsize, dtype=np.float32)
                else:
                    data['n_t'][data['y_te'][i].item()] = np.zeros(config['n_vocab'], dtype=np.float32)
            data['n_t'][data['y_te'][i].item()][idx] += counts

    # if classes is None:
    classes = np.unique(data['y_tr'])
    if config['ctxEmb'] == 'bert':
        n_tokens = np.zeros((len(classes), bertsize), dtype=np.float32)
    else:
        n_tokens = np.zeros((len(classes), config['n_vocab']), dtype=np.float32)

    for i, key in enumerate(classes):
        n_tokens[i, :] = data['n_t'][key]


    n_tokens_sum = np.sum(n_tokens, axis=0, keepdims=True)
    n_tokens_sum[0][0] = 0
    n_total = np.sum(n_tokens_sum)

    p_t = n_tokens_sum / n_total

    # compute iwf
    iwf = 1e-5 / (1e-5 + p_t)
    iwf = np.transpose(iwf)

    sample_emb = data['x_tr'].squeeze(dim=1)
    sample_label = data['y_ind'].squeeze(dim=1)
    lamda = 1

    ebd2 = torch.tensor(data['sc_vec'])
    train_I = torch.eye(ebd2.size()[0])
    data['sc_ind'] = torch.eye(ebd2.size()[0])
    num_use = torch.sum(torch.abs(sample_emb)> 0, dim=1)

    w = ebd2.t() \
        @ torch.inverse(ebd2 @ ebd2.t() + 1 * train_I) @ data['sc_ind']

    ebd_unseen = torch.tensor(data['uc_vec'])
    train_I_unseen = torch.eye(ebd_unseen.size()[0])
    data['uc_ind'] = torch.eye(ebd_unseen.size()[0])
    # num_use = torch.sum(torch.abs(sample_emb)> 0, dim=1)

    un_w = ebd_unseen.t() \
        @ torch.inverse(ebd_unseen @ ebd_unseen.t() + 1 * train_I_unseen) @ data['uc_ind']

    data['embedding'] = torch.Tensor(data['embedding'])
    vab =  torch.zeros(data['embedding'].size())
    vab[1:] = data['embedding'][1:]
    train_ebd = vab[sample_emb]
    train_I = torch.eye(data['x_tr'].size()[0])
    # num_use = torch.sum(torch.abs(data['meta_train_sample'])> 0, dim=1)
    # vab =  torch.zeros(data['embedding'].size())
    # vab[1:] = data['embedding'][1:]
    # train_ebd = vab[data['meta_train_sample']]
    train_avg_sum = torch.sum(train_ebd, dim=1)
    num_use = num_use.unsqueeze(dim=1)
    ebd = train_avg_sum/torch.tensor(data['s_len']).unsqueeze(dim=1)
    w2 = ebd.t() \
        @ torch.inverse(ebd @ ebd.t() + 1 * train_I) @ sample_label

    iwf[0] = 0
    train_iwf = torch.tensor(iwf[data['meta_train_sample']]).to(device)
    realtest_iwf = torch.tensor(iwf[data['x_te']]).to(device)
    # train_st = st[data['meta_train_sample_id']]
    # test_st = st[data['meta_test_sample_id']]
    print("got")
    return train_iwf,train_iwf.size()[0],ebd,realtest_iwf,w,w2,un_w


def train_outlier_detection(data, config,
                            caption=''):

    print(">> training for embedding <<")

    device = config['device']
    batch_size = config['batch_size'] * (1 if config['dataset'] == 'SMP18' else 4)
    refine_batch_size = config['refine_batch_size'] * (1 if config['dataset'] == 'SMP18' else 4)
    embedding = data['embedding'] if config['text_represent'] == 'w2v' else None
    # class_token_ids = data['class_padded'].to(device)
    n_tr = data['y_tr'].shape[0]


    sample_dim = config['sample_dim']
    attention_hidden_dim = config['attention_hidden_dim']
    lstm_output_dim = config['lstm_output_dim']
    label_dim = config['label_dim']
    ifattention = config['ifattention']  #True

    #mixture attention
    att_generator = model_out.att_Generator(config,sample_dim, lstm_output_dim,attention_hidden_dim, embedding=embedding).to(device)
    #utterance embedding
    lstm  = model_out.lstm(sample_dim, lstm_output_dim, embedding=embedding).to(device)
    #map the label
    generator_stov = model_out.Generator_stov(config,label_dim, lstm_output_dim, embedding=embedding).to(device)




    #meta-train optim
    pre_generator_optim = torch.optim.Adam(list(lstm.parameters())+list(att_generator.parameters()) + list(generator_stov.parameters()) ,
                                 lr=config['learning_rate'])
    print(config['learning_rate'])
    #meta-adapt optim
    only_stov_optim = torch.optim.Adam( list(generator_stov.parameters()),
                                 lr= config['learning_rate2'] )#config['learning_rate'])
    #control lr
    ssize = config['stepsize']
    gamma = config['LRDGamma']
    StepLRDistance = torch.optim.lr_scheduler.StepLR(pre_generator_optim, step_size=ssize, gamma=gamma)
    #meta-adapt
    StepLRDRefine = torch.optim.lr_scheduler.StepLR(only_stov_optim, step_size=ssize, gamma=gamma)

    config['ifBN'] = False
    Norm = False
    CosTest = False  
    config['BN2'] = False
    config['f1'] = True
    stopflag = 0
    stopnum = 0
    if config['ifBN'] == True:
        BN = torch.nn.BatchNorm1d(sample_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
    if config['BN2'] == True:
        BN2 = torch.nn.BatchNorm1d(2*lstm_output_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)

    data['x_tr'] = torch.LongTensor(data['x_tr'])
    data['y_tr'] = torch.LongTensor(data['y_tr'])
    data['x_te'] = torch.LongTensor(data['x_te'])
    data['y_te'] = torch.LongTensor(data['y_te'])
    myembedding = torch.FloatTensor(embedding).to(device)
    if config['ctxEmb'] == 'bert':
        from transformers import BertTokenizer, BertModel
        word_embedding = BertModel.from_pretrained(config['bertname']).to(device)

        for name, param in word_embedding.named_parameters():
            param.requires_grad = False

    # training begin
    n_batch = (n_tr - 1) // batch_size + 1
    # train_iwf, refine_iwf, meta_n_tr, sentence_ebd = get_meta_data(data, config)
    for i_epoch in range(config['n_epoch']):
        att_generator.train()
        lstm.train()
        generator_stov.train()

        acc_avg = 0.0
        loss_avg = 0.0
        tr_emb = []

        if config['meta_test_num'] !=0 :
            train_iwf, refine_iwf, meta_n_tr,sentence_ebd ,real_test_iwf,stw,stw2,un_stw= get_meta_data(device,data, config)
        else:
            train_iwf,  meta_n_tr, sentence_ebd, real_test_iwf, stw,stw2, un_stw = f_get_meta_data(device, data,
                                                                                                       config)
        #train_iwf: meta_train_num * T * 1
        # refine_inf:  meta_test_num * T * 1
        # meta_n_tr: meta_train_num
        # sentence_ebd: seen_num * embeddingsize
        # real_test_iwf: real_test * T * 1

        n_batch =  (meta_n_tr - 1) // batch_size #+1   
        new_train_st = -1 * torch.ones(train_iwf.size())
        train_loss1 = []
        train_loss2 = []
        update_vec = torch.zeros(train_iwf.size())
        stw = stw.to(device)
        i_batch = 0
        # index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, meta_n_tr))
        for i_batch in range(n_batch):

            # make batch data
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, meta_n_tr))

            # get_idf(data,config)
            att_generator.train()
            lstm.train()
            # generator_vtos.eval()
            generator_stov.train()

            pre_generator_optim.zero_grad()

            batch_x = data['meta_train_sample'][index, :].to(device)
            if config['ifBN'] == True:
                batch_y_emb = BN( torch.tensor(data['sc_vec'][data['meta_train_label']][index, :]).to(device)) # batch_size * label_dim
            else:
                batch_y_emb = torch.tensor(data['sc_vec'][data['meta_train_label']][index, :]).to(
                    device)  # batch_size * label_dim

            batch_y = data['meta_train_ind'][index].to(device)

            batch_y_label = data['meta_train_ind'][index, :] #对应的onehot batch_size * label_dim
            batch_len = data['meta_train_len'][index]
            iwf = train_iwf[index]
            st = new_train_st[index]

            if config['ifBN'] == True:
                pre_emb = myembedding[batch_x].transpose(1,2)
                pre_emb = BN(pre_emb).transpose(1, 2)
            else:
                if config['ctxEmb'] == 'bert':
                    mask = torch.zeros(batch_x.size())
                    for i in range(batch_x.size()[0]):
                        mask[i][:batch_len[i]] = 1
                    output = word_embedding(input_ids=batch_x, attention_mask=mask.to(device))
                    pre_emb = output.last_hidden_state
                else:
                    pre_emb = myembedding[batch_x]

            #
            if config['stw_sample']==True:
                w = torch.abs(pre_emb @ stw2.to(device))
            else:
                w = torch.softmax(pre_emb @ un_stw.to(device),dim=2)
                # w = torch.abs(pre_emb @ stw.to(device))
                        # x = torch.cat([x, w_target.detach()], dim=-1)

          
            w = torch.softmax(pre_emb @ un_stw.to(device), dim=2)
            st = (-1) * torch.mul(w, torch.log(w + 1e-6)).sum(dim=2)

            st = 1.0 / st
            st = st.unsqueeze(dim=2)
            # st = w.max(dim=2, keepdim=True)[0]

            batch_x_emb = lstm(pre_emb , batch_len)
            emb = att_generator(batch_x_emb, batch_len, data['embedding'], iwf, st, ifattention).squeeze(1)

            #
            pre_sample = generator_stov(batch_y_emb)

            f = 0

            sc_label_v = generator_stov(torch.from_numpy(data['sc_vec']).to(device))

  
            distance = torch.zeros(emb.size()[0], sc_label_v.size()[0]).to(device)
            for i in range(distance.size()[0]):
                distance[i] = torch.norm((emb[i] - sc_label_v), p=2, dim=1)
            distance = F.log_softmax(-1 * distance, dim=1)
            p_e = (distance @ batch_y.t().to(device)).diag()
 
            loss_MCE = (-1)* p_e.mean() 


            #loss function
            b = config['d_hp']
            distance_loss = b * loss_MCE


            #Update parameter
            distance_loss.backward()
            pre_generator_optim.step()
            train_loss1.append(distance_loss)
            torch.cuda.empty_cache()


        #meta-adapting
        print("loss: %f" % ( torch.tensor(train_loss1).mean() ))
        print("here is refine,LR", StepLRDistance.get_lr())
        if config['meta_test_num'] !=0 :
            print("here is refine")
            n_batch =  (data['meta_test_sample'].size()[0] - 1) // refine_batch_size +1
            refine_n_tr = data['meta_test_sample'].size()[0]
            for i_batch in range(n_batch):
                index = range(i_batch * refine_batch_size, min((i_batch + 1) * refine_batch_size, refine_n_tr))
                batch_te_x = data['meta_test_sample'][index].to(device)
                batch_te_len = data['meta_test_len'][index]
                batch_y_emb = torch.tensor(data['sc_vec'][data['meta_test_label']][index, :]).to(device)
                iwf = refine_iwf[index]

                #embedding
                if config['ctxEmb'] == 'bert':
                    mask = torch.zeros(batch_te_x.size())
                    for i in range(batch_te_x.size()[0]):
                        mask[i][:batch_te_len[i]] = 1
                    pre_emb = word_embedding(input_ids=batch_te_x, attention_mask=mask.to(device)).last_hidden_state
                    # pre_emb = output.last_hidden_state
                else:
                    pre_emb = myembedding[batch_te_x]
                w = torch.softmax(pre_emb @ un_stw.to(device), dim=2)
                st = (-1) * torch.mul(w, torch.log(w + 1e-6)).sum(dim=2)
                st = 1.0 / st
                st = st.unsqueeze(dim=2)


                batch_x_emb = lstm(pre_emb , batch_te_len)
                emb = att_generator(batch_x_emb, batch_te_len, data['embedding'], iwf, st, ifattention).squeeze(1)

                label_tov =  generator_stov(torch.from_numpy(data['sc_vec']).to(device))
                distance = torch.zeros(emb.size()[0],label_tov.size()[0]).to(device)

                for i in range(distance.size()[0]):
                    a = torch.norm((emb[i] - label_tov), p=2,dim=1)
                    distance[i] = a


                #distance
                if config['ifsigmoid'] == True:
                    distance = F.sigmoid(distance)
                distance = F.softmax(-1 * distance, dim=1)
                sc_label_v = generator_stov(torch.from_numpy(data['sc_vec']).to(device))


                p = distance #(distance + p2) / 2
                min_val, min_ind = p.max(dim=1)
                fun_9 = (distance @ data['meta_test_ind'][index].t().to(device)).diag()
                loss_refine = (-1) * torch.log(fun_9).mean() 

                #Update parameters
                only_stov_optim.zero_grad()

                loss_refine.backward()
                only_stov_optim.step()

                #show the results
                train_correct = (min_ind == data['meta_test_label'][index].to(device)).sum()
                acc = train_correct * 1.0 / min_ind.size()[0]
                print('%s || epoch:%d || loss:%f || acc_tr:%f' % (caption, i_epoch, loss_refine, acc))
                if acc == 1:
                    stopflag = 1
                    stopnum += 1
                else:
                    stopflag = 0
                    stopnum = 0

                torch.cuda.empty_cache()


        #Test
        print("----Test Begin----")
        #update lr
        print("distance-LR:", StepLRDistance.get_lr(),"RefineLR",StepLRDRefine.get_lr())
        StepLRDistance.step()
        StepLRDRefine.step()

        #eval
        att_generator.eval()
        lstm.eval()
        generator_stov.eval()

        te_x = data['x_te'].to(device)
        te_len = torch.tensor(data['u_len'])

        if config['ctxEmb'] == 'bert':
            mask = torch.zeros(te_x.size())
            for i in range(te_x.size()[0]):
                mask[i][:te_len[i]] = 1

            batch = 50
            index = (te_x.size()[0] - 1) 
            pre_emb = []
            for i in range(index):
                ind = range(i * batch, min((i + 1) * batch, te_x.size()[0]))
                batch_te_x = te_x[ind]
                batch_mask = mask[ind]
                output = word_embedding(input_ids=batch_te_x, attention_mask=batch_mask.to(device))
                if i==0:
                    pre_emb = output.last_hidden_state
                else:
                    pre_emb = torch.cat((pre_emb,output.last_hidden_state),dim=0)
                # pre_emb.append(output.last_hidden_state)
                torch.cuda.empty_cache()

        else:
            pre_emb = myembedding[te_x]
        torch.cuda.empty_cache()

        if config['ifBN'] == True:
            pre_emb = pre_emb.transpose(1,2)
            pre_emb = BN(pre_emb).transpose(1,2)


        iwf = real_test_iwf
        un_stw = un_stw.to(device)
        w =  torch.softmax(pre_emb @ un_stw.to(device), dim=2)
        st = (-1)*torch.mul(w, torch.log(w+1e-6)).sum(dim=2)

        st = 1.0 / st
        st = st.unsqueeze(dim=2)

        x_emb = lstm(pre_emb, te_len)
        emb = att_generator(x_emb, te_len, data['embedding'], iwf, st, ifattention).squeeze(1)

        y_vec = torch.from_numpy(data['uc_vec']).to(device)

        if config['ifBN'] == True:
            y_vec = BN(y_vec)
        label_tov = generator_stov(y_vec) #generator_stov(torch.from_numpy(data['uc_vec']).to(device))
        distance = torch.zeros(emb.size()[0], label_tov.size()[0]).to(device)

        if config['BN2'] == True:
            emb_norm = BN2(emb)
            label_tov_norm =BN2(label_tov)
        if Norm == True:
            emb_norm = F.normalize(emb,p=2,dim=1)
            label_tov_norm = F.normalize(label_tov,p=2,dim=1)
        else:
            emb_norm = emb
            label_tov_norm = label_tov


        for i in range(distance.size()[0]):
            distance[i] = torch.norm((emb_norm[i] - label_tov_norm), p=2, dim=1)

        p1 = F.softmax(-1 * distance, dim=1)

        min_val1, min_ind1 = p1.max(dim=1)

        min_val, min_ind = p1.max(dim=1)
        test_mask = [i for i in range(min_val.size()[0]) if min_val[i] < config['test_flag']]
        class_id = torch.zeros(len(data['unseen_class']))
        for i in range(len(data['unseen_class'])):
            class_id[i] = data['uc_dict'][data['unseen_class'][i][0]]
        for i in range(len(test_mask)):
            new_distance = p1[test_mask[i]][class_id.long()]
            new_val, new_ind = new_distance.max(dim=0)
            min_ind[test_mask[i]] = class_id[new_ind]

        test_correct = (min_ind == data['y_te'].to(device)).sum()
        testacc = test_correct * 1.0 / min_ind.size()[0]

        class_id = torch.zeros(len(data['unseen_class']))
        for i in range(len(data['unseen_class'])):
            class_id[i] = data['uc_dict'][data['unseen_class'][i][0]]
        test_seen = torch.zeros(data['y_te'].size())
        seen_num = 0
        seen_num_cor = 0
        unseen_num = 0
        unseen_num_cor = 0
        total_unseen_test = []
        total_unseen_pred = []
        total_seen_test = []
        total_seen_pred = []
        for i in range(data['y_te'].size()[0]):
            if data['y_te'][i] in class_id:
                test_seen[i] = 1  # unseen
                total_unseen_test.append(data['y_te'][i])
                total_unseen_pred.append(min_ind[i])
                unseen_num += 1
                if data['y_te'][i].to(device) == min_ind[i]:
                    unseen_num_cor += 1
            else:
                test_seen[i] = -1  # seen
                seen_num += 1
                total_seen_test.append(data['y_te'][i])
                total_seen_pred.append(min_ind[i])
                if data['y_te'][i].to(device) == min_ind[i]:
                    seen_num_cor += 1

        total_unseen_pred = torch.tensor(total_unseen_pred)
        total_unseen_test = torch.tensor(total_unseen_test)
        total_seen_pred = torch.tensor(total_seen_pred)
        total_seen_test = torch.tensor(total_seen_test)
        pre = precision_recall_fscore_support(data['y_te'], min_ind.cpu() ,average='weighted')
        uspre = precision_recall_fscore_support(total_unseen_test, total_unseen_pred,average='weighted')
        spre = precision_recall_fscore_support(total_seen_test, total_seen_pred,average='weighted')

        log_hard = precision_recall_fscore_support(data['y_te'],  min_ind.cpu())
        pfm = parse_sklearn_log(log_hard, config['n_seen_class'])
        print("GZS:")
        print(pfm)


        p3 =  p1
        min_val, min_ind = p3.max(dim=1)
        test_mask = [i for i in range(min_val.size()[0]) if min_val[i] < 2]
        class_id = torch.zeros(len(data['unseen_class']))
        for i in range(len(data['unseen_class'])):
            class_id[i] = data['uc_dict'][data['unseen_class'][i][0]]
        for i in range(len(test_mask)):
            new_distance = p3[test_mask[i]][class_id.long()]
            new_val, new_ind = new_distance.max(dim=0)
            min_ind[test_mask[i]] = class_id[new_ind].to(device)

        test_correct = (min_ind == data['y_te'].to(device)).sum()
        testacc = test_correct * 1.0 / min_ind.size()[0]

        class_id = torch.zeros(len(data['unseen_class']))
        for i in range(len(data['unseen_class'])):
            class_id[i] = data['uc_dict'][data['unseen_class'][i][0]]
        test_seen = torch.zeros(data['y_te'].size())
        seen_num = 0
        seen_num_cor = 0
        unseen_num = 0
        unseen_num_cor = 0
        total_unseen_test = []
        total_unseen_pred = []
        total_seen_test = []
        total_seen_pred = []
        for i in range(data['y_te'].size()[0]):
            if data['y_te'][i] in class_id:
                test_seen[i] = 1  # unseen
                total_unseen_test.append(data['y_te'][i])
                total_unseen_pred.append(min_ind[i])
                unseen_num += 1
                if data['y_te'][i].to(device) == min_ind[i]:
                    unseen_num_cor += 1
            else:
                test_seen[i] = -1  # seen
                seen_num += 1
                total_seen_test.append(data['y_te'][i])
                total_seen_pred.append(min_ind[i])
                if data['y_te'][i].to(device) == min_ind[i]:
                    seen_num_cor += 1

        total_unseen_pred = torch.tensor(total_unseen_pred)
        total_unseen_test = torch.tensor(total_unseen_test)
        total_seen_pred = torch.tensor(total_seen_pred)
        total_seen_test = torch.tensor(total_seen_test)
        pre = precision_recall_fscore_support(data['y_te'], min_ind.cpu(), average='weighted')
        uspre = precision_recall_fscore_support(total_unseen_test, total_unseen_pred, average='weighted')
        spre = precision_recall_fscore_support(total_seen_test, total_seen_pred, average='weighted')
        s_testacc = seen_num_cor * 1.0 / seen_num
        u_testacc = unseen_num_cor * 1.0 / unseen_num
        log_hard2 = precision_recall_fscore_support(total_unseen_test, total_unseen_pred)

        pfm2 = parse_sklearn_log(log_hard2, 0)
        print("SZS:")
        print(pfm2)

     
    return pfm['micro_rec_all'],pfm['micro_f1_all'],pfm['micro_rec_seen'],pfm['micro_f1_seen'],pfm['micro_rec_unseen'],pfm['micro_f1_unseen'],pfm2['micro_rec_unseen'],pfm2['micro_f1_unseen']

     