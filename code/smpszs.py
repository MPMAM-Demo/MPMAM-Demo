# import os
import time
# import pickle
from random import *


import os
import json
from argparse import ArgumentParser
import pandas as pd
import torch
from numpy.random import seed

import input_data
from mymodel import *
a = Random()
a.seed(0)

# Setting here!
test_description = 'saved_models'
rep_num = 1
# SNIP, SMP18
choose_dataset = "SMP18"

# without seen: 0, with seen: 1, fixed with some classes: -1
dataSetting = {}
dataSetting['test_mode'] = 1
######
dataSetting['random_class'] = False
dataSetting['training_prob'] = 0.8
dataSetting['test_intrain_prob'] = 0.3

# =============================================================================

dataSetting['data_prefix'] = '../data/SNIP/'
dataSetting['dataset_name'] = 'dataSNIP.txt'
dataSetting['wordvec_name'] = 'wiki.en.vec'
dataSetting['sim_name_withOS'] = 'SNIP_split.mat'
dataSetting['ctxEmb'] = 'emb'

if choose_dataset == "SMP18":
    dataSetting['data_prefix'] = '../data/SMP18/'
    dataSetting['dataset_name'] = 'dataSMP18.txt'
    dataSetting['wordvec_name'] = 'sgns_merge_subsetSMP.txt'
    dataSetting['sim_name_withOS'] = 'SMP_split.mat'
    dataSetting['ctxEmb'] = 'emb'

if choose_dataset == 'SNIP':
    dataSetting['unseen_class'] = [['playlist'], ['book']]
elif choose_dataset == 'SMP18':
    dataSetting['unseen_class'] = [['天气'], ['公交'], ['app'], ['飞机'], ['电影'], ['音乐']]

id_split = range(0, 1) 
# ==============================================================================

def setting(data):

    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]
    config = {}
    if dataSetting['ctxEmb'] == 'bert':
        config['ctxEmb'] = 'bert'
        config['bertname'] = dataSetting['bertname']
    else:
        config['ctxEmb'] = 'emb'


    config['model_name'] = "SMP-SZS"
    config['dataset'] = choose_dataset
    config['test_mode'] = dataSetting['test_mode']
    config['training_prob'] = dataSetting['training_prob']
    config['test_intrain_prob'] = dataSetting['test_intrain_prob']
    config['wordvec'] = dataSetting['wordvec_name']
    config['sim_name_withOS'] = dataSetting['sim_name_withOS']
    config['vocab_size'] = vocab_size  # vocab size of word vectors (10,895)
    config['max_time'] = max_time
    config['sample_num'] = sample_num  # sample number of training data
    config['test_num'] = test_num  # number of test data
    config['s_cnum'] = s_cnum  # seen class num
    config['u_cnum'] = u_cnum  # unseen class num
    config['word_emb_size'] = word_emb_size  # embedding size of word vectors (300)
    config['nlayers'] = 2  # default for bilstm
    config['seen_class'] = data['seen_class']
    config['unseen_class'] = data['unseen_class']
    config['data_prefix'] = dataSetting['data_prefix']
    config['ckpt_dir'] = './' + test_description + '/'  # check point dir
    config['experiment_time'] = time.strftime('%y%m%d%I%M%S')
    config['report'] = True
    config['cuda_id'] = 0
    config['untrain_classlen'] = data['untrain_classlen']
    config['use_gpu'] = True
    config['device'] = torch.device("cuda" if torch.cuda.is_available() & config['use_gpu'] else "cpu")
    config['text_represent'] = 'w2v'



    config['batch_size'] = 100
    config['refine_batch_size'] = 100
    config['learning_rate'] = 0.008
    config['learning_rate2'] = 0.002
    config['n_tr_sample'] = data['x_tr'].shape[0]
    config['n_te_sample'] = data['x_te'].shape[0]
    config['n_seen_class'] = len(data['seen_class'])
    config['n_unseen_class'] = len(data['unseen_class'])
    config['n_vocab'] = len(data['embedding'])
    config['d_emb'] = len(data['embedding'][0])
    config['n_epoch'] = 100
    config['meta_test_num'] = 5
    config['atttolstm'] = False
    config['ifattention'] = True
    config['pre_p_opt_refine'] =False
    config['test_flag'] = 0.8
    config['lamdaGAN'] = 10
    config['norm_a'] = 0
    config['d_hp'] = 1
    config['stepsize'] = 20
    config['LRDGamma'] = 0.5
    config['ifsigmoid'] = False
    config['ifDS']=True
    config['stw_sample']=False

    config['sample_dim'] = 300
    config['attention_hidden_dim'] = 25
    config['lstm_output_dim'] = 64
    config['label_dim'] = 300
    config['hsize'] = 150


    return config


def seed_torch(seed=5): #3
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    seed_torch()
    if dataSetting['test_mode'] != 1:
        id_split = range(1)

    overall_recall_list=[]
    overall_f1_list=[]
    s_recall_list=[]
    s_f1_list=[]
    u_recall_list=[]
    u_f1_list=[]
    only_u_recall_list=[]
    only_u_f1_list=[]

    for idsplit in id_split:
        dataSetting['id_split'] = idsplit

        # load data
        data = input_data.read_datasets(dataSetting)

        # load settings
        config = setting(data)
        cuda_id = config['cuda_id']

        x_tr = torch.from_numpy(data['x_tr'])
        y_tr = torch.from_numpy(data['y_tr'])
        y_tr_id = torch.from_numpy(data['y_tr'])
        y_te_id = torch.from_numpy(data['y_te'])
        y_ind = torch.from_numpy(data['s_label'])
        s_len = torch.from_numpy(data['s_len'])
        embedding = torch.from_numpy(data['embedding'])
        x_te = torch.from_numpy(data['x_te'])
        u_len = torch.from_numpy(data['u_len'])

        if torch.cuda.is_available():
            x_tr = x_tr.cuda(cuda_id)
            y_tr = y_tr.cuda(cuda_id)
            y_tr_id = y_tr_id.cuda(cuda_id)
            y_te_id = y_te_id.cuda(cuda_id)
            y_ind = y_ind.cuda(cuda_id)
            s_len = s_len.cuda(cuda_id)
            embedding = embedding.cuda(cuda_id)
            x_te = x_te.cuda(cuda_id)
            u_len = u_len.cuda(cuda_id)
            print('------------------use gpu------------------')

            overall_recall, overall_f1, s_recall, s_f1, u_recall, u_f1, only_u_recall, only_u_f1 = train_outlier_detection(
                data, config)
            overall_recall_list.append(overall_recall)
            overall_f1_list.append(overall_f1)
            s_recall_list.append(s_recall)
            s_f1_list.append(s_f1)
            u_recall_list.append(u_recall)
            u_f1_list.append(u_f1)
            only_u_recall_list.append(only_u_recall)
            only_u_f1_list.append(only_u_f1)

    overall_recall = torch.tensor(overall_recall_list).mean()
    overall_f1 = torch.tensor(overall_f1_list).mean()
    s_recall = torch.tensor(s_recall_list).mean()
    s_f1 = torch.tensor(s_f1_list).mean()
    u_recall = torch.tensor(u_recall_list).mean()
    u_f1 = torch.tensor(u_f1_list).mean()
    only_u_recall = torch.tensor(only_u_recall_list).mean()
    only_u_f1 = torch.tensor(only_u_f1_list).mean()

    # print("Generalization zeroshot:overall:recall=%f,f1=%f---seen:recall=%f,f1=%f----unseen:recall=%f,f1=%f"
    #       % (overall_recall, overall_f1, s_recall, s_f1, u_recall, u_f1))

    print("Standard zeroshot:recall=%f,f1=%f"
          % (only_u_recall, only_u_f1))
