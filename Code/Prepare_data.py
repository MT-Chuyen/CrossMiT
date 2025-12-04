import pandas as pd
import random
import math
import os

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
PATH_CSV_MTI = '../Data/miRNA-target.csv'
PATH_CSV_MDA = '../Data/miRNA-disease.csv'
OUTPUT_ROOT = '../Data/Data-kFold/'
# =======================================================

def ensureDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def list_str(list_data):
    return [str(i) for i in list_data]

def load_and_filter_csv(csv_path):
    ratings = pd.read_csv(csv_path, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating', 'timestamp']
    rate_size_dic = ratings.groupby('itemId').size()
    choosed_index_del = rate_size_dic.index[rate_size_dic < 0] 
    ratings = ratings[~ratings['itemId'].isin(list(choosed_index_del))]
    return ratings

def reindex_data_shared(ratings_mda, ratings_mti):
    user_unique_mda = ratings_mda['userId'].unique()
    user_unique_mti = ratings_mti['userId'].unique()
    
    all_users = list(set(user_unique_mda) & set(user_unique_mti))
    all_users = sorted(all_users)
    
    dic_u_shared = dict(zip(all_users, range(len(all_users))))
    
    print(f"Total Shared MiRNAs (n_users): {len(all_users)}")
    
    def map_domain_data(ratings, dic_u_shared):
        ratings = ratings[ratings['userId'].isin(dic_u_shared.keys())].copy()
        
        item_unique = list(ratings['itemId'].unique())
        item_index = list(range(len(item_unique)))
        dic_m = dict(zip(item_unique, item_index))
        
        data = []
        for element in ratings.values:
            data.append((dic_u_shared[element[0]], dic_m[element[1]], 1))
        data = sorted(data, key=lambda x: x[0])
        return data, dic_m

    data_mda, dic_m_mda = map_domain_data(ratings_mda, dic_u_shared)
    data_mti, dic_m_mti = map_domain_data(ratings_mti, dic_u_shared)
    
    return data_mda, data_mti, dic_u_shared

def build_data_dic_and_nitems(data):
    dic = {}
    n_items = 0
    for u, i, r in data:
        if u not in dic:
            dic[u] = []
        dic[u].append(i)
        if i > n_items:
            n_items = i
    return dic, n_items

def generate_negatives(user_id, train_items, test_item, n_items, neg_num=99):
    neg_items = []
    train_set = set(train_items)
    while len(neg_items) < neg_num:
        neg_id = random.randint(0, n_items)
        if neg_id not in train_set and neg_id != test_item and neg_id not in neg_items:
            neg_items.append(neg_id)
    return neg_items

def write_files(save_dir, train_data, test_data, neg_data):
    ensureDir(save_dir)
    
    with open(os.path.join(save_dir, 'train.txt'), 'w') as f:
        for u, i in train_data:
            f.write(f"{u}\t{i}\t1\n")
            
    with open(os.path.join(save_dir, 'test.txt'), 'w') as f:
        for u, i in test_data:
            f.write(f"{u}\t{i}\t1\n")
            
    with open(os.path.join(save_dir, 'test_neg.txt'), 'w') as f:
        for u, i, negs in neg_data:
            line = f"({u}, {i})" + '\t' + '\t'.join(list_str(negs))
            f.write(line + '\n')

def process_static_domain_for_fold(data_dic, n_items):
    train_out, test_out, neg_out = [], [], []
    random.seed(2025) 
    
    print(f"   MDA: Processing {len(data_dic)} users...")
    
    for u, items in data_dic.items():
        if len(items) > 0:
            items_copy = items[:]
            random.shuffle(items_copy)
            
            test_item = items_copy[-1]
            train_items = items_copy[:-1]
            
            test_out.append([u, test_item])
            for i in train_items:
                train_out.append([u, i])
            
            negs = generate_negatives(u, train_items, test_item, n_items)
            neg_out.append([u, test_item, negs])
            
    return train_out, test_out, neg_out

def prepare_kfold_data():
    print("=== STARTING 5-FOLD DATA PREPARATION ===\n")
    
    print("[1] Reading & Filtering CSVs...")
    ratings_mda = load_and_filter_csv(PATH_CSV_MDA)
    ratings_mti = load_and_filter_csv(PATH_CSV_MTI)
    
    print("[2] Reindexing data with shared MiRNA index...")
    data_mda_reindexed, data_mti_reindexed, dic_u_shared = reindex_data_shared(ratings_mda, ratings_mti)
    
    dic_mda, n_items_mda = build_data_dic_and_nitems(data_mda_reindexed)
    dic_mti, n_items_mti = build_data_dic_and_nitems(data_mti_reindexed)
    
    print(f"   Max MiRNA Index (n_users): {len(dic_u_shared)}")
    print(f"   MDA: {n_items_mda + 1} items (Diseases)")
    print(f"   MTI: {n_items_mti + 1} items (Targets)")

    print("\n[3] Processing Source Domain (Static LOO Split for Evaluation)...")
    train_mda, test_mda, neg_mda = process_static_domain_for_fold(dic_mda, n_items_mda)
    
    print("[4] Splitting Target Domain MiRNAs into 5-Folds...")
    users_mti = list(dic_mti.keys())
    random.seed(2025)
    random.shuffle(users_mti)
    
    k_fold = 5
    fold_size = math.ceil(len(users_mti) / k_fold)
    user_groups = [users_mti[i:i+fold_size] for i in range(0, len(users_mti), fold_size)]
    print(f"   Total {len(users_mti)} users split into {len(user_groups)} folds\n")
    
    print("[5] Generating data for 5 folds...")
    for i, test_group in enumerate(user_groups):
        fold_idx = i + 1
        fold_name = f"Fold_{fold_idx}"
        print(f"   {fold_name}...", end=" ")
        
        fold_root = os.path.join(OUTPUT_ROOT, fold_name)
        
        dir_target = os.path.join(fold_root, 'miRNA-target_miRNA-disease/')
        train_mti, test_mti, neg_mti = [], [], []
        test_group_set = set(test_group)
        
        for u, items in dic_mti.items():
            items_copy = items[:]
            random.shuffle(items_copy)
            
            if u in test_group_set:
                if len(items_copy) > 0:
                    test_item = items_copy[-1]
                    train_items = items_copy[:-1]
                else:
                    continue 

                test_mti.append([u, test_item])
                for item in train_items:
                    train_mti.append([u, item])
                
                negs = generate_negatives(u, train_items, test_item, n_items_mti)
                neg_mti.append([u, test_item, negs])
            else:
                for item in items_copy:
                    train_mti.append([u, item])
        
        write_files(dir_target, train_mti, test_mti, neg_mti)
        
        dir_source = os.path.join(fold_root, 'miRNA-disease_miRNA-target/')
        write_files(dir_source, train_mda, test_mda, neg_mda)
        print("✓")
        
    print(f"\n=== COMPLETE! ===")
    print(f"k-Fold data is located at: {OUTPUT_ROOT}")

if __name__ == '__main__':
    random.seed(42) 
    prepare_kfold_data()