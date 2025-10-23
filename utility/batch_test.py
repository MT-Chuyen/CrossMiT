'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
Updated for TensorFlow 2.x compatibility by [Your Name], 2025
'''
import math
import heapq  # for retrieval topK
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict

def test(model, data_generator, test_user_list, data_type, batch_size, Ks, layer_size):
    """
    Evaluate the model on test data.
    
    Args:
        model: BiTGCF model instance (TensorFlow 2.x Keras model)
        data_generator: Data object providing test instances
        test_user_list: List of user IDs to evaluate
        data_type: 'source' or 'target' domain
        batch_size: Batch size for evaluation
        Ks: List of k values for HR@k and NDCG@k
        layer_size: Layer size configuration (used for dropout consistency)
    
    Returns:
        hr: Hit ratio for each k in Ks (numpy array)
        ndcg: NDCG for each k in Ks (numpy array)
    """
    # Get test instances
    users, items, user_gt_item = get_test_instance(data_generator, test_user_list)
    num_test_batches = len(users) // batch_size + 1

    test_preds = []
    if data_type == 'source':
        for current_batch in tqdm(range(num_test_batches), desc='test_source', ascii=True):
            min_idx = current_batch * batch_size
            max_idx = min((current_batch + 1) * batch_size, len(users))
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]

            # Inputs contains only tensor data
            inputs = {
                'users_s': batch_input_users,
                'items_s': batch_input_items,
                'label_s': np.zeros(len(batch_input_users), dtype=np.float32),
                'users_t': np.zeros_like(batch_input_users, dtype=np.int32),
                'items_t': np.zeros_like(batch_input_items, dtype=np.int32),
                'label_t': np.zeros_like(batch_input_users, dtype=np.float32)
            }

            # Pass node_dropout and mess_dropout as keyword arguments
            scores_s, scores_t, emb_s, emb_t = model(
                inputs, 
                training=False, 
                node_dropout=0.0, 
                mess_dropout=[0.0] * len(eval(layer_size))
            )
            test_preds.extend(scores_s.numpy())

        assert len(test_preds) == len(users), 'Source prediction count does not match user count'
    else:  # target
        for current_batch in tqdm(range(num_test_batches), desc='test_target', ascii=True):
            min_idx = current_batch * batch_size
            max_idx = min((current_batch + 1) * batch_size, len(users))
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]

            # Inputs contains only tensor data
            inputs = {
                'users_s': np.zeros_like(batch_input_users, dtype=np.int32),
                'items_s': np.zeros_like(batch_input_items, dtype=np.int32),
                'label_s': np.zeros_like(batch_input_users, dtype=np.float32),
                'users_t': batch_input_users,
                'items_t': batch_input_items,
                'label_t': np.zeros(len(batch_input_users), dtype=np.float32)
            }

            # Pass node_dropout and mess_dropout as keyword arguments
            scores_s, scores_t, emb_s, emb_t = model(
                inputs, 
                training=False, 
                node_dropout=0.0, 
                mess_dropout=[0.0] * len(eval(layer_size))
            )
            test_preds.extend(scores_t.numpy())

        assert len(test_preds) == len(users), 'Target prediction count does not match user count'

    # Aggregate predictions by user
    user_item_preds = defaultdict(lambda: defaultdict(float))
    for sample_id in range(len(users)):
        user = users[sample_id]
        item = items[sample_id]
        pred = test_preds[sample_id]
        user_item_preds[user][item] = pred

    # Compute HR and NDCG for each user
    hits, ndcgs = [], []
    for user in user_item_preds.keys():
        item_pred = user_item_preds[user]
        hrs, nds = [], []
        for k in Ks:
            ranklist = heapq.nlargest(k, item_pred, key=item_pred.get)
            hr = getHitRatio(ranklist, user_gt_item[user])
            ndcg = getNDCG(ranklist, user_gt_item[user])
            hrs.append(hr)
            nds.append(ndcg)
        hits.append(hrs)
        ndcgs.append(nds)

    # Average across users
    hr = np.array(hits).mean(axis=0)
    ndcg = np.array(ndcgs).mean(axis=0)
    return hr, ndcg

def get_test_instance(data_generator, test_user_list):
    """
    Generate test instances from rating and negative lists.
    
    Args:
        data_generator: Data object with ratingList and negativeList
        test_user_list: List of user indices to evaluate
    
    Returns:
        users: Array of user IDs (repeated for each item)
        items: Array of item IDs (ground truth + negatives)
        user_gt_item: Dict mapping users to their ground truth item
    """
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
    """
    Compute Hit Ratio for a ranked list.
    
    Args:
        ranklist: List of top-k predicted items
        gtItem: Ground truth item
    
    Returns:
        1 if gtItem is in ranklist, 0 otherwise
    """
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    """
    Compute NDCG for a ranked list.
    
    Args:
        ranklist: List of top-k predicted items
        gtItem: Ground truth item
    
    Returns:
        NDCG value (0 if gtItem not in ranklist)
    """
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0