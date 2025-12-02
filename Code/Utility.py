import math
import heapq
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict
import os
import re
import tensorflow as tf
import random as rd
import scipy.sparse as sp
def test(model, data_generator, test_user_list, data_type, batch_size, Ks, layer_size):
    Ks_list = eval(Ks)

    users, items, user_gt_item = get_test_instance(data_generator, test_user_list)
    num_test_batches = len(users) // batch_size + 1
    test_preds = []

    if data_type == 'source':
        desc = 'test_source'
    else:
        desc = 'test_target'

    for current_batch in tqdm(range(num_test_batches), desc=desc, ascii=True):
        min_idx = current_batch * batch_size
        max_idx = min((current_batch + 1) * batch_size, len(users))
        batch_input_users = users[min_idx:max_idx]
        batch_input_items = items[min_idx:max_idx]

        if data_type == 'source':
            inputs = {'users_s': batch_input_users, 'items_s': batch_input_items, 'label_s': np.zeros(len(batch_input_users), dtype=np.float32),
                      'users_t': np.zeros_like(batch_input_users, dtype=np.int32), 'items_t': np.zeros_like(batch_input_items, dtype=np.int32), 'label_t': np.zeros_like(batch_input_users, dtype=np.float32)}
            scores, _, _, _ = model(inputs, training=False, node_dropout=0.0, mess_dropout=[0.0] * len(eval(layer_size)))
        else:
            inputs = {'users_s': np.zeros_like(batch_input_users, dtype=np.int32), 'items_s': np.zeros_like(batch_input_items, dtype=np.int32), 'label_s': np.zeros_like(batch_input_users, dtype=np.float32),
                      'users_t': batch_input_users, 'items_t': batch_input_items, 'label_t': np.zeros(len(batch_input_users), dtype=np.float32)}
            _, scores, _, _ = model(inputs, training=False, node_dropout=0.0, mess_dropout=[0.0] * len(eval(layer_size)))

        test_preds.extend(scores.numpy())
    assert len(test_preds) == len(users), 'Prediction count does not match user count'


    user_item_preds = defaultdict(lambda: defaultdict(float))
    for sample_id in range(len(users)):
        user = users[sample_id]
        item = items[sample_id]
        pred = test_preds[sample_id]
        user_item_preds[user][item] = pred

    all_hits, all_ndcgs, all_precisions, all_recalls, all_f1s, all_aps = [], [], [], [], [], []
    all_labels_flat, all_preds_flat = [], []

    for user in user_item_preds.keys():
        item_pred = user_item_preds[user]
        gtItem = user_gt_item[user]

        item_scores = sorted(item_pred.items(), key=lambda x: x[1], reverse=True)

        r = [1 if item == gtItem else 0 for item, score in item_scores]
        all_pos_num = 1

        for item, pred in item_scores:
            all_preds_flat.append(pred)
            all_labels_flat.append(1 if item == gtItem else 0)

        user_ap = average_precision(r, cut=len(r))
        all_aps.append(user_ap)

        user_metrics_k = defaultdict(list)
        for k in Ks_list:
            ranklist = [item for item, score in item_scores[:k]]
            r_k = r[:k]

            user_metrics_k['HR'].append(getHitRatio(ranklist, gtItem))
            user_metrics_k['NDCG'].append(getNDCG(ranklist, gtItem))

            prec = precision_at_k(r_k, k)
            rec = recall_at_k(r_k, k, all_pos_num)
            f1 = F1(prec, rec)

            user_metrics_k['Precision'].append(prec)
            user_metrics_k['Recall'].append(rec)
            user_metrics_k['F1'].append(f1)

        all_hits.append(user_metrics_k['HR'])
        all_ndcgs.append(user_metrics_k['NDCG'])
        all_precisions.append(user_metrics_k['Precision'])
        all_recalls.append(user_metrics_k['Recall'])
        all_f1s.append(user_metrics_k['F1'])


    results = {
        'Ks': Ks_list,
        'HR': np.array(all_hits).mean(axis=0).tolist(),
        'NDCG': np.array(all_ndcgs).mean(axis=0).tolist(),
        'Precision': np.array(all_precisions).mean(axis=0).tolist(),
        'Recall': np.array(all_recalls).mean(axis=0).tolist(),
        'F1': np.array(all_f1s).mean(axis=0).tolist(),
        'AUC': auc(np.array(all_labels_flat), np.array(all_preds_flat)),
        'MAP': np.mean(all_aps)
    }
    return results

def get_test_instance(data_generator, test_user_list):

    users, items = [], []
    user_gt_item = {}
    rating_list = np.array(data_generator.ratingList)
    negative_list = data_generator.negativeList

    for idx in test_user_list:
        rating = rating_list[idx]
        items_neg = negative_list[idx]
        u = rating[0]
        gtItem = rating[1]
        user_gt_item[u] = gtItem
        for item in items_neg:
            users.append(u)
            items.append(item)
        users.append(u)
        items.append(gtItem)

    return np.array(users), np.array(items), user_gt_item

def getHitRatio(ranklist, gtItem):

    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):

    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

__author__ = "xiangwang"
import os
import re

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def pprint(_str,file):
    print(_str)
    print(_str,file=file)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def early_stopping(log_hr, best_hr,stopping_step , flag_step=100):

    if log_hr >= best_hr :
        stopping_step = 0
        best_hr = log_hr
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_hr))
        should_stop = True
    else:
        should_stop = False
    return best_hr, stopping_step, should_stop

def search_index_from_file(string):
    p1 = re.compile(r'hit=[[](.*?)[]]', re.S)
    p2 = re.compile(r'ndcg=[[](.*?)[]]',re.S)
    p3 = re.compile(r'=(\d*\.\d*?) \+',re.S)
    p4 = re.compile(r'\+ (\d*\.\d*?)[],]',re.S)
    return re.findall(p1, string), re.findall(p2, string),re.findall(p3, string),re.findall(p4, string)

import tensorflow as tf
def optimizer(learner,loss,learning_rate,momentum=0.9):
    optimizer=None
    if learner.lower() == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                             initial_accumulator_value=1e-8).minimize(loss)
    elif learner.lower() == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "gd" :
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif learner.lower() == "momentum" :
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
    else :
        raise ValueError("please select a suitable optimizer")
    return optimizer

def pairwise_loss(loss_function,y,margin=1):
    loss=None
    if loss_function.lower() == "bpr":
        loss = -tf.reduce_sum(tf.log_sigmoid(y))
    elif loss_function.lower() == "hinge":
        loss = tf.reduce_sum(tf.maximum(y+margin, 0))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(1-y))
    else:
        raise Exception("please choose a suitable loss function")
    return loss

def pointwise_loss(loss_function,y_rea,y_pre):
    loss=None
    if loss_function.lower() == "cross_entropy":
        loss = tf.losses.sigmoid_cross_entropy(y_rea,y_pre)

    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(y_rea-y_pre))
    else:
        raise Exception("please choose a suitable loss function")
    return loss

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.helper import *

class Data(object):
    def __init__(self, path, batch_size,neg_num):
        self.neg_num = neg_num
        self.path = path
        self.batch_size = batch_size
        train_file = path +'/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file, "r") as f:
            line = f.readline().strip('\n')
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                self.n_users = max(self.n_users, u)
                self.n_items = max(self.n_items, i)
                self.n_train += 1
                line = f.readline().strip('\n')

        self.n_items += 1
        self.n_users += 1

        self.negativeList = self.read_neg_file(path)
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.ratingList = []
        self.train_items, self.test_set = {}, {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n').split('\t')
                    user, item, rating = int(l[0]), int(l[1]), float(l[2])
                    if user in self.train_items.keys():
                        self.train_items[user].append(item)
                    else:
                        self.train_items[user] = [item]
                    if (rating > 0):
                        self.R[user, item] = 1.0

                line = f_test.readline().strip('\n')
                while line != None and line != "":
                    arr = line.split("\t")
                    user, item = int(arr[0]), int(arr[1])
                    if user in self.test_set.keys():
                        self.test_set[user].append(item)
                    else:
                        self.test_set[user] = [item]
                    self.ratingList.append([user, item])
                    self.n_test += 1
                    line = f_test.readline().strip('\n')

    def get_R_mat(self):
        return self.R
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)


        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def read_neg_file(self,path):
        try:
            test_neg = path + '/test_neg.txt'
            test_neg_f = open(test_neg ,'r')
        except:
            negativeList = None
            return negativeList
        negativeList = []
        line = test_neg_f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                item = int(x)
                negatives.append(item)
            negativeList.append(negatives)
            line = test_neg_f.readline()
        return negativeList
    def get_test_neg_item(self,u,negativeList):
        neg_items = negativeList[u]
        return neg_items
    def get_train_instance(self):
        user_input, item_input, labels = [],[],[]
        for (u, i) in self.R.keys():
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            for _ in range(self.neg_num):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R.keys():
                    j = np.random.randint(self.n_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return np.array(user_input),np.array(item_input),np.array(labels)

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self,save_log):
        pprint('n_users=%d, n_items=%d' % (self.n_users, self.n_items),save_log)
        pprint('n_interactions=%d' % (self.n_train + self.n_test),save_log)
        pprint('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)),save_log)
    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split('\t')])
            print('get sparsity split.')
        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write('\t'.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state

import numpy as np
from sklearn.metrics import roc_auc_score

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):

    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r,cut):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num
def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.
def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.
def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run BiTGCF.")
    parser.add_argument('--weights_path', nargs='?', default='../',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='/media/mountHDD2/chuyenmt/ReSys/BiTGCF-MTI-2/Data-unlimited-user-iem/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='./',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='miRNA-disease_miRNA-target',
                        help='Choose a dataset')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=8192*4,
                        help='Batch size.')
    parser.add_argument('--lambda_s',default='0.8')
    parser.add_argument('--lambda_t',default='0.8')
    parser.add_argument('--isconcat',type = int ,default=1,
                        help='does transfer inter domain?')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001*2,
                        help='Learning rate.')
    parser.add_argument('--initial_type', default='x')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=1,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--weight_id', type=float, default=0.1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--weight_loss',nargs='?', default='[1.,1.]')
    parser.add_argument('--n_interaction',type=int, default=3)
    parser.add_argument('--neg_num',type=int, default=4)
    parser.add_argument('--connect_type',type=str, default='concat',
                        help='concat or mean')
    parser.add_argument('--layer_fun',type=str, default='gcf',
                        help='feature propagation way')
    parser.add_argument('--fuse_type_in',type=str, default='la2add',
                        help='inter-domain feature fuse type')
    parser.add_argument('--sparcy_flag',type=int, default=0)
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume training from')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='Ratio of data to keep (default: 1.0)')
    return parser.parse_args()