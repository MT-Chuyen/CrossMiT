from utility.utility_3 import *
from BiTGCF_2 import BiTGCF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from random import randint, random
from tqdm import tqdm
from time import time
import wandb
import scipy.sparse as sp # Thêm import sp cho các ma trận thưa

# Các biến toàn cục được sử dụng trong print_test_result (được giữ lại để tương thích với cấu trúc cũ)
global epoch, losses, loss_source, loss_target, args, save_log_file, t0

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def get_adj_mat(config, data_generator, adj_type, domain_type):
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if adj_type == 'plain':
        config['norm_adj_%s' % domain_type] = plain_adj
        print('%s use the plain adjacency matrix' % domain_type)
    elif adj_type == 'norm':
        config['norm_adj_%s' % domain_type] = norm_adj
        print('%s use the normalized adjacency matrix' % domain_type)
    elif adj_type == 'gcmc':
        config['norm_adj_%s' % domain_type] = mean_adj
        print('%s use the gcmc adjacency matrix' % domain_type)
    else:
        # Giả sử đây là trường hợp 'mean' hoặc mặc định
        config['norm_adj_%s' % domain_type] = mean_adj + sp.eye(mean_adj.shape[0]) 
        print('%s use the mean adjacency matrix' % domain_type)


# HÀM IN KẾT QUẢ MỚI (Sử dụng output dictionary từ test())
def print_test_result_new(results, train_time, test_time, domain_type, data_status):
    global epoch, losses, loss_source, loss_target, args, save_log_file 
    
    if args.verbose > 0:
        hr, ndcg, prec, rec, f1 = results['HR'], results['NDCG'], results['Precision'], results['Recall'], results['F1']
        auc_score, map_score = results['AUC'], results['MAP']
        Ks = results['Ks']

        # In thông tin cơ bản
        perf_str_info = 'Epoch %d [%.1fs + %.1fs]: train==[%.4f=%.4f + %.4f] (%s)\n' % (
            epoch, train_time, test_time, losses, loss_source, loss_target, data_status)
        perf_str_info += f"--- {domain_type.upper()} GLOBAL METRICS ---\n"
        perf_str_info += f"AUC: {auc_score:.4f}, MAP (AUPR proxy): {map_score:.4f}\n"

        # In Ranking Metrics @K
        perf_str_k = f"--- {domain_type.upper()} RANKING METRICS @K (K={Ks[0]} to {Ks[-1]}) ---\n"
        for i, k in enumerate(Ks):
             perf_str_k += f"K={k:2d}: HR={hr[i]:.4f}, NDCG={ndcg[i]:.4f}, Prec={prec[i]:.4f}, Rec={rec[i]:.4f}, F1={f1[i]:.4f}\n"

        pprint(perf_str_info, save_log_file)
        pprint(perf_str_k, save_log_file)


# Giữ lại hàm cũ, nhưng nó sẽ chỉ sử dụng HR và NDCG (chỉ dùng cho mục đích tương thích nếu cần)
def print_test_result(hr, ndcg, train_time, test_time, domain_type, data_status):
    global epoch, losses, loss_source, loss_target, args, save_log_file 
    if args.verbose > 0:
        # Lưu ý: Hàm này không còn phù hợp với output mới của test(), nên sử dụng print_test_result_new
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.4f=%.4f + %.4f], hit=%s, ndcg=%s at %s' % (
            epoch, train_time, test_time, losses, loss_source, loss_target, str(['%.4f' % i for i in hr]),
            str(['%.4f' % i for i in ndcg]), data_status)
        pprint(perf_str, save_log_file)

def find_best_epoch(ndcg_loger, hit_loger, domain_type):
    global t0, save_log_file
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    hit_kmax = hit[:, -1]  # Lấy giá trị HR@K_max (mặc định K=10)
    best_rec_0 = max(hit_kmax)
    idx = list(hit_kmax).index(best_rec_0)
    pprint('{:*^40}'.format(domain_type + ' part'), save_log_file)
    final_perf = "Best Iter=[%d]@[%.1f]\t hit=%s, ndcg=%s" % (
        idx, time() - t0, str(['%.4f' % i for i in list(hit[idx])]), str(['%.4f' % i for i in list(ndcgs[idx])]))
    pprint(final_perf, save_log_file)
    return final_perf

if __name__ == '__main__':
    args = parse_args()

    # Thiết lập GPU dựa trên args.gpu_id
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
        if args.gpu_id >= 0 and args.gpu_id < len(gpus):
            # Đặt GPU được sử dụng
            try:
                tf.config.set_visible_devices(gpus[args.gpu_id], 'GPU')
                print(f"Using GPU: {gpus[args.gpu_id].name}")
                # Cấu hình tăng trưởng bộ nhớ (memory growth) để tránh chiếm toàn bộ VRAM
                tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
            except RuntimeError as e:
                print(e)
                print("Could not set up GPU configuration.")
        else:
            print(f"GPU ID {args.gpu_id} is out of range. Using default/CPU.")
    else:
        print("No GPU found. Using CPU.")

    # Khởi tạo W&B
    wandb.init(
        project="BiTGCF-miRNA", 
        name=f"run_lr_{args.lr}_b_{args.batch_size}_layers_{args.layer_size}", 
        config={ 
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "layer_size": args.layer_size,
            "epochs": args.epoch,
            "adj_type": args.adj_type,
            "connect_type": args.connect_type,
            "mess_dropout": args.mess_dropout,
            "dataset": args.dataset,
            "neg_num": args.neg_num,
            "Ks": args.Ks # Log Ks mới
        }
    )

    weight_id = random()
    save_log_dir = './logs/%s/%s/' % (args.dataset, str(args.layer_size))
    ensureDir(save_log_dir)
    save_log_file = open(save_log_dir + 'lr_%s_b%s_id_%.4f.txt' % (str(args.lr), args.batch_size, weight_id), 'w+')
    
    config = dict()
    Ks = args.Ks # Lấy string Ks mới
    layer_size = args.layer_size
    BATCH_SIZE = args.batch_size
    neg_num = args.neg_num
    source_name, target_name = args.dataset.split('_')
    
    # Khởi tạo Data Generator
    data_generator_s = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, neg_num=neg_num)
    data_generator_t = Data(path=args.data_path + target_name + '_' + source_name, batch_size=args.batch_size, neg_num=neg_num)
    
    pprint('{:*^40}'.format('source data info'), save_log_file)
    data_generator_s.print_statistics(save_log_file)
    pprint('{:*^40}'.format('target data info'), save_log_file)
    data_generator_t.print_statistics(save_log_file)
    assert data_generator_s.n_users == data_generator_t.n_users, 'data-erro,user should be shared'

    # Compute domain_adj
    domain_adj = sp.dok_matrix((data_generator_s.n_users, 2), dtype=np.float32)
    domain_adj = domain_adj.tolil()
    R_s = data_generator_s.get_R_mat()
    R_t = data_generator_t.get_R_mat()
    domain_adj[:, 0] = R_s.sum(1)
    domain_adj[:, 1] = R_t.sum(1)
    domain_adj = domain_adj.todok()
    degree_sum = np.array(domain_adj.sum(1))
    d_inv = np.power(degree_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv[:, 0])
    norm_domain_adj = d_mat_inv.dot(domain_adj)
    config['domain_adj'] = np.array(norm_domain_adj.todense())

    config['n_users'] = data_generator_s.n_users
    config['n_items_s'] = data_generator_s.n_items
    config['n_items_t'] = data_generator_t.n_items

    # Generate adjacency matrices
    get_adj_mat(config, data_generator_s, args.adj_type, 's')
    get_adj_mat(config, data_generator_t, args.adj_type, 't')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    if args.sparcy_flag:
        split_ids_s, split_status_s = data_generator_s.get_sparsity_split()
        split_ids_t, split_status_t = data_generator_t.get_sparsity_split()
    else:
        split_ids_s, split_ids_t, split_status_s, split_status_t = [], [], [], []
        split_ids_s.append(range(data_generator_s.n_users))
        split_ids_t.append(range(data_generator_t.n_users))
        split_status_s.append('full rating, #user=%d' % data_generator_s.n_users)
        split_status_t.append('full rating, #user=%d' % data_generator_t.n_users)

    # Instantiate the model
    model = BiTGCF(data_config=config, args=args, pretrain_data=pretrain_data)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Checkpoint for saving/loading weights
    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s_%s/%s/N_layer=%s/l%s_b%s_layer%s_adj%s_connect%s_drop%s' % (
            args.weights_path, args.dataset, args.keep_ratio, args.layer_fun, layer,
            str(args.lr), args.batch_size, layer, args.adj_type, args.connect_type, args.mess_dropout
        )
        ensureDir(weights_save_path)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, weights_save_path, max_to_keep=1)

        # Load checkpoint if resuming
        if args.resume_epoch > 0:
            latest_checkpoint = tf.train.latest_checkpoint(weights_save_path)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint).expect_partial()
                pprint(f'Resumed training from checkpoint: {latest_checkpoint} at epoch {args.resume_epoch}', save_log_file)
            else:
                pprint(f'No checkpoint found at {weights_save_path}, starting from scratch', save_log_file)

    # Load pretrained weights if applicable
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s_%s/%s/N_layer=%s/l%s_b%s_layer%s_adj%s_connect%s_drop%s' % (
            args.weights_path, args.dataset, args.keep_ratio, args.layer_fun, layer,
            str(args.lr), args.batch_size, layer, args.adj_type, args.connect_type, args.mess_dropout
        )
        pprint(pretrain_path, save_log_file)
        try:
            checkpoint.restore(pretrain_path).expect_partial()
            pprint(f'Loaded pretrained model parameters from: {pretrain_path}', save_log_file)
        except Exception as e:
            pprint(f'Failed to load pretrained weights: {e}. Starting without pretraining.', save_log_file)
    else:
        pprint('Training without pretraining.', save_log_file)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    pre_loger_t, rec_loger_t, ndcg_loger_t, hit_loger_t = [], [], [], []

    # Initial evaluation
    pprint('\n{:*^40}'.format('source initial result'), save_log_file)
    for test_user_list_s, data_status in zip(split_ids_s, split_status_s):
        # Sử dụng hàm test mới
        test_results_s = test(model, data_generator_s, test_user_list_s, 'source', 2048, Ks, layer_size)
        hr_s, ndcg_s = test_results_s['HR'], test_results_s['NDCG']
        
        best_hr_s, best_ndcg_s = hr_s[-1], ndcg_s[-1]
        
        # In kết quả initial
        pprint(f"--- SOURCE INITIAL RESULTS ({data_status}) ---", save_log_file)
        for i, k in enumerate(test_results_s['Ks']):
             pprint(f"K={k:2d}: HR={hr_s[i]:.4f}, NDCG={ndcg_s[i]:.4f}, Prec={test_results_s['Precision'][i]:.4f}, Rec={test_results_s['Recall'][i]:.4f}, F1={test_results_s['F1'][i]:.4f}", save_log_file)
        pprint(f"AUC: {test_results_s['AUC']:.4f}, MAP: {test_results_s['MAP']:.4f}", save_log_file)
        # Lưu HR và NDCG cho mục đích early stopping và find_best_epoch
        initial_hr_s = hr_s
        initial_ndcg_s = ndcg_s
        

    pprint('\n{:*^40}'.format('target initial result'), save_log_file)
    for test_user_list_t, data_status in zip(split_ids_t, split_status_t):
        # Sử dụng hàm test mới
        test_results_t = test(model, data_generator_t, test_user_list_t, 'target', 2048, Ks, layer_size)
        hr_t, ndcg_t = test_results_t['HR'], test_results_t['NDCG']
        
        best_hr_t, best_ndcg_t = hr_t[-1], ndcg_t[-1]

        # In kết quả initial
        pprint(f"--- TARGET INITIAL RESULTS ({data_status}) ---", save_log_file)
        for i, k in enumerate(test_results_t['Ks']):
             pprint(f"K={k:2d}: HR={hr_t[i]:.4f}, NDCG={ndcg_t[i]:.4f}, Prec={test_results_t['Precision'][i]:.4f}, Rec={test_results_t['Recall'][i]:.4f}, F1={test_results_t['F1'][i]:.4f}", save_log_file)
        pprint(f"AUC: {test_results_t['AUC']:.4f}, MAP: {test_results_t['MAP']:.4f}", save_log_file)
        # Lưu HR và NDCG cho mục đích early stopping và find_best_epoch
        initial_hr_t = hr_t
        initial_ndcg_t = ndcg_t

    # LƯU KẾT QUẢ INITIAL VÀO LOGER
    ndcg_loger.append(initial_ndcg_s)
    hit_loger.append(initial_hr_s)
    ndcg_loger_t.append(initial_ndcg_t)
    hit_loger_t.append(initial_hr_t)

    if args.save_flag == 1:
        checkpoint_manager.save()
        pprint(f'Saved weights to: {weights_save_path}', save_log_file)

    # Training loop
    stopping_step, stopping_step_s = 0, 0
    should_stop_s, should_stop_t = False, False

    # Điều chỉnh vòng lặp để bắt đầu từ args.resume_epoch
    for epoch in range(args.resume_epoch, args.epoch):
        t1 = time()
        loss, loss_source, loss_target = [], [], []

        user_input_s, item_input_s, label_s = data_generator_s.get_train_instance()
        user_input_t, item_input_t, label_t = data_generator_t.get_train_instance()
        train_len_s = len(user_input_s)
        train_len_t = len(user_input_t)
        
        shuffled_idx_s = np.random.permutation(np.arange(train_len_s))
        train_u_s, train_i_s, train_r_s = user_input_s[shuffled_idx_s], item_input_s[shuffled_idx_s], label_s[shuffled_idx_s]
        
        shuffled_idx_t = np.random.permutation(np.arange(train_len_t))
        train_u_t, train_i_t, train_r_t = user_input_t[shuffled_idx_t], item_input_t[shuffled_idx_t], label_t[shuffled_idx_t]
        
        n_batch_s = train_len_s // args.batch_size + (1 if train_len_s % args.batch_size != 0 else 0)
        n_batch_t = train_len_t // args.batch_size + (1 if train_len_t % args.batch_size != 0 else 0)
        n_batch_max = max(n_batch_s, n_batch_t)
        n_batch_min = min(n_batch_s, n_batch_t)
        
        # --- Single training loop for the larger domain ---
        if n_batch_s != n_batch_t:
            if n_batch_s > n_batch_t:
                pprint('source domain single train', save_log_file)
                # Training domain Source từ n_batch_min đến n_batch_max
                for i in tqdm(range(n_batch_min, n_batch_max), desc='train_source (single)', ascii=True):
                    min_idx = i * BATCH_SIZE
                    max_idx = min([(i + 1) * BATCH_SIZE, train_len_s])
                    
                    # Lấy batch và xử lý padding nếu cần
                    if max_idx < (i + 1) * BATCH_SIZE:
                        pad_len = (i + 1) * BATCH_SIZE - max_idx
                        idex = list(range(min_idx, max_idx)) + list(np.random.randint(0, train_len_s, pad_len))
                    else:
                        idex = list(range(min_idx, max_idx))
                    
                    train_u_batch, train_i_batch, train_r_batch = train_u_s[idex], train_i_s[idex], train_r_s[idex]
                    
                    inputs = {
                        'users_s': train_u_batch, 'items_s': train_i_batch, 'label_s': train_r_batch,
                        'users_t': np.zeros_like(train_u_batch, dtype=np.int32), 
                        'items_t': np.zeros_like(train_i_batch, dtype=np.int32),
                        'label_t': np.zeros_like(train_r_batch, dtype=np.float32)
                    }

                    with tf.GradientTape() as tape:
                        scores_s, scores_t, emb_s, emb_t = model(
                            inputs, training=True,
                            node_dropout=eval(args.node_dropout)[0] if args.node_dropout_flag else 0.0,
                            mess_dropout=eval(args.mess_dropout)
                        )
                        _, batch_loss_source, _ = model.compute_loss((scores_s, scores_t), (emb_s, emb_t))
                    gradients = tape.gradient(batch_loss_source, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    loss_source.append(batch_loss_source.numpy())
            else:
                pprint('target domain single train', save_log_file)
                # Training domain Target từ n_batch_min đến n_batch_max
                for i in tqdm(range(n_batch_min, n_batch_max), desc='train_target (single)', ascii=True):
                    min_idx = i * BATCH_SIZE
                    max_idx = min([(i + 1) * BATCH_SIZE, train_len_t])
                    
                    # Lấy batch và xử lý padding nếu cần
                    if max_idx < (i + 1) * BATCH_SIZE:
                        pad_len = (i + 1) * BATCH_SIZE - max_idx
                        idex = list(range(min_idx, max_idx)) + list(np.random.randint(0, train_len_t, pad_len))
                    else:
                        idex = list(range(min_idx, max_idx))
                    
                    train_u_batch, train_i_batch, train_r_batch = train_u_t[idex], train_i_t[idex], train_r_t[idex]
                    
                    inputs = {
                        'users_s': np.zeros_like(train_u_batch, dtype=np.int32),
                        'items_s': np.zeros_like(train_i_batch, dtype=np.int32), 
                        'label_s': np.zeros_like(train_r_batch, dtype=np.float32),
                        'users_t': train_u_batch, 'items_t': train_i_batch, 'label_t': train_r_batch
                    }

                    with tf.GradientTape() as tape:
                        scores_s, scores_t, emb_s, emb_t = model(
                            inputs, training=True,
                            node_dropout=eval(args.node_dropout)[0] if args.node_dropout_flag else 0.0,
                            mess_dropout=eval(args.mess_dropout)
                        )
                        _, _, batch_loss_target = model.compute_loss((scores_s, scores_t), (emb_s, emb_t))
                    gradients = tape.gradient(batch_loss_target, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    loss_target.append(batch_loss_target.numpy())

        # --- Joint training loop ---
        for i in tqdm(range(n_batch_min), desc='train_join', ascii=True):
            min_idx = i * BATCH_SIZE
            max_idx = min([(i + 1) * BATCH_SIZE, min(train_len_s, train_len_t)])
            
            # Lấy batch và xử lý padding nếu cần (Giữ nguyên logic padding cũ, có thể không tối ưu)
            if max_idx < (i + 1) * BATCH_SIZE:
                pad_len = (i + 1) * BATCH_SIZE - max_idx
                rand_idx = np.random.randint(0, min(train_len_s, train_len_t), pad_len)
                idex = list(range(min_idx, max_idx)) + list(rand_idx)
            else:
                idex = list(range(min_idx, max_idx))
            
            train_u_batch_s, train_i_batch_s, train_r_batch_s = train_u_s[idex], train_i_s[idex], train_r_s[idex]
            train_u_batch_t, train_i_batch_t, train_r_batch_t = train_u_t[idex], train_i_t[idex], train_r_t[idex]

            inputs = {
                'users_s': train_u_batch_s, 'items_s': train_i_batch_s, 'label_s': train_r_batch_s,
                'users_t': train_u_batch_t, 'items_t': train_i_batch_t, 'label_t': train_r_batch_t
            }

            with tf.GradientTape() as tape:
                scores_s, scores_t, emb_s, emb_t = model(
                    inputs, training=True,
                    node_dropout=eval(args.node_dropout)[0] if args.node_dropout_flag else 0.0,
                    mess_dropout=eval(args.mess_dropout)
                )
                batch_loss, batch_loss_source, batch_loss_target = model.compute_loss((scores_s, scores_t), (emb_s, emb_t))
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            loss.append(batch_loss.numpy())
            loss_source.append(batch_loss_source.numpy())
            loss_target.append(batch_loss_target.numpy())

        losses = np.mean(loss) if loss else 0.0
        loss_source = np.mean(loss_source) if loss_source else 0.0
        loss_target = np.mean(loss_target) if loss_target else 0.0

        t2 = time()

        # Evaluation Source
        pprint('\n{:*^40}'.format('source result'), save_log_file)
        for test_user_list_s, data_status in zip(split_ids_s, split_status_s):
            # Sử dụng hàm test mới
            test_results_s = test(model, data_generator_s, test_user_list_s, 'source', 2048, Ks, layer_size)
            hr_s, ndcg_s = test_results_s['HR'], test_results_s['NDCG']
            print_test_result_new(test_results_s, t2 - t1, time() - t2, 'source', data_status)

        t3 = time()
        
        # Evaluation Target
        pprint('\n{:*^40}'.format('target result'), save_log_file)
        for test_user_list_t, data_status in zip(split_ids_t, split_status_t):
            # Sử dụng hàm test mới
            test_results_t = test(model, data_generator_t, test_user_list_t, 'target', 2048, Ks, layer_size)
            hr_t, ndcg_t = test_results_t['HR'], test_results_t['NDCG']
            print_test_result_new(test_results_t, t2 - t1, time() - t3, 'target', data_status)

        t4 = time()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "loss/total": losses,
            "loss/source": loss_source,
            "loss/target": loss_target,
            
            # Global Metrics
            "source/auc": test_results_s['AUC'],
            "source/map": test_results_s['MAP'],
            "target/auc": test_results_t['AUC'],
            "target/map": test_results_t['MAP'],
            
            # Metrics @K_max (K=10)
            "source/hr_kmax": hr_s[-1], 
            "source/ndcg_kmax": ndcg_s[-1], 
            "source/prec_kmax": test_results_s['Precision'][-1],
            "source/rec_kmax": test_results_s['Recall'][-1],
            "source/f1_kmax": test_results_s['F1'][-1],

            "target/hr_kmax": hr_t[-1],
            "target/ndcg_kmax": ndcg_t[-1],
            "target/prec_kmax": test_results_t['Precision'][-1],
            "target/rec_kmax": test_results_t['Recall'][-1],
            "target/f1_kmax": test_results_t['F1'][-1],
            
            # Thời gian
            "time/train": t2 - t1,
            "time/eval_source": t3 - t2,
            "time/eval_target": t4 - t3
        })
        
        # Log metrics cho TỪNG K (1 đến 10)
        for i, k in enumerate(test_results_s['Ks']):
            wandb.log({
                f"source/hr@{k}": hr_s[i],
                f"source/ndcg@{k}": ndcg_s[i],
                f"source/prec@{k}": test_results_s['Precision'][i],
                f"source/rec@{k}": test_results_s['Recall'][i],
                f"source/f1@{k}": test_results_s['F1'][i],
                f"target/hr@{k}": hr_t[i],
                f"target/ndcg@{k}": ndcg_t[i],
                f"target/prec@{k}": test_results_t['Precision'][i],
                f"target/rec@{k}": test_results_t['Recall'][i],
                f"target/f1@{k}": test_results_t['F1'][i],
            }, step=epoch)

        loss_loger.append(losses)
        ndcg_loger.append(ndcg_s)
        hit_loger.append(hr_s)
        ndcg_loger_t.append(ndcg_t)
        hit_loger_t.append(hr_t)

        best_hr_s, stopping_step_s, should_stop_s = early_stopping(hr_s[-1], best_hr_s, stopping_step_s, flag_step=5)
        best_hr_t, stopping_step, should_stop_t = early_stopping(hr_t[-1], best_hr_t, stopping_step, flag_step=5)

        # if all([should_stop_s, should_stop_t]):
        #     break

        # Lưu checkpoint sau mỗi epoch nếu save_flag=1
        if args.save_flag == 1:
            checkpoint_manager.save()
            pprint(f'Saved weights for epoch {epoch} to: {weights_save_path}', save_log_file)

        if hr_t[-1] == best_hr_t and args.save_flag == 1:
            checkpoint_manager.save()
            pprint(f'Saved weights to: {weights_save_path} (best HR on target)', save_log_file)

        save_log_file.flush()

    # Kết thúc W&B run
    wandb.finish()

    final_perf_s = find_best_epoch(ndcg_loger, hit_loger, 'source')
    final_perf_t = find_best_epoch(ndcg_loger_t, hit_loger_t, 'target')
    save_path = '%soutput/%s.result' % (args.proj_path, args.dataset)
    ensureDir(save_path)

    with open(save_path, 'a') as f:
        f.write(
            '\n lr=%.4f, layer_fun=%s, fuse_type_in=%s, neg_num=%s, n_interaction=%s, connect_type=%s\n\t%s\n%s\n%s\n%s' % (
                args.lr, args.layer_fun, args.fuse_type_in, args.neg_num, args.n_interaction, args.connect_type,
                'source result', final_perf_s, 'target result', final_perf_t
            )
        )

    save_log_file.close()