import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Utility import *

class CrossMiT(tf.keras.Model):
    def __init__(self, data_config, args, pretrain_data):
        super(CrossMiT, self).__init__()

        # Argument settings
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.initial_type = args.initial_type
        self.pretrain_data = pretrain_data
        self.fuse_type_in = args.fuse_type_in

        self.n_mirnas = data_config['n_mirnas']
        self.n_disease = data_config['n_disease']
        self.n_items_t = data_config['n_items_t']

        self.n_fold = 100

        self.norm_adj_s = data_config['norm_adj_s']
        self.norm_adj_t = data_config['norm_adj_t']
        self.n_nonzero_elems_s = self.norm_adj_s.count_nonzero()
        self.n_nonzero_elems_t = self.norm_adj_t.count_nonzero()

        self.domain_laplace = data_config['domain_adj']
        self.connect_way = args.connect_type
        self.layer_fun = args.layer_fun

        self.lr = args.lr
        self.n_interaction = args.n_interaction
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.lambda_s = eval(args.lambda_s)
        self.lambda_t = eval(args.lambda_t)

        if self.initial_type == 'x':
            self.initializer = tf.keras.initializers.GlorotUniform()
        elif self.initial_type == 'u':
            self.initializer = tf.random.normal_initializer()

        self.weight_source, self.weight_target = eval(args.weight_loss)[:]
        self.node_dropout_flag = args.node_dropout_flag

        # Initialize weights
        self.weights_source = self._init_weights('source', self.n_disease, None)
        self.weights_target = self._init_weights('target', self.n_items_t, None)

    def _init_weights(self, name_scope, n_items, mirna_embedding):
        all_weights = dict()

        if self.pretrain_data is None:
            if mirna_embedding is None:
                all_weights['mirna_embedding'] = tf.keras.layers.Embedding(
                    self.n_mirnas, self.emb_dim, embeddings_initializer=self.initializer, name=f'mirna_embedding{name_scope}'
                )
                all_weights['item_embedding'] = tf.keras.layers.Embedding(
                    n_items, self.emb_dim, embeddings_initializer=self.initializer, name=f'item_embedding_{name_scope}'
                )
                print('using xavier initialization')
            else:
                all_weights['mirna_embedding'] = tf.keras.layers.Embedding(
                    self.n_mirnas, self.emb_dim, embeddings_initializer=tf.constant_initializer(mirna_embedding), 
                    trainable=True, name=f'mirna_embedding{name_scope}'
                )
                all_weights['item_embedding'] = tf.keras.layers.Embedding(
                    n_items, self.emb_dim, embeddings_initializer=self.initializer, name=f'item_embedding_{name_scope}'
                )
        else:
            all_weights['mirna_embedding'] = tf.keras.layers.Embedding(
                self.n_mirnas, self.emb_dim, embeddings_initializer=tf.constant_initializer(self.pretrain_data['mirna_embed']),
                trainable=True, name=f'mirna_embedding{name_scope}'
            )
            all_weights['item_embedding'] = tf.keras.layers.Embedding(
                n_items, self.emb_dim, embeddings_initializer=tf.constant_initializer(self.pretrain_data['item_embed']),
                trainable=True, name=f'item_embedding_{name_scope}'
            )
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights[f'W_gc_{k}'] = tf.keras.layers.Dense(
                self.weight_size_list[k+1], kernel_initializer=self.initializer, name=f'W_gc_{k}_{name_scope}'
            )
            all_weights[f'b_gc_{k}'] = tf.keras.layers.Dense(
                self.weight_size_list[k+1], use_bias=True, bias_initializer=self.initializer, name=f'b_gc_{k}_{name_scope}'
            )
            all_weights[f'W_bi_{k}'] = tf.keras.layers.Dense(
                self.weight_size_list[k+1], kernel_initializer=self.initializer, name=f'W_bi_{k}_{name_scope}'
            )
            all_weights[f'b_bi_{k}'] = tf.keras.layers.Dense(
                self.weight_size_list[k+1], use_bias=True, bias_initializer=self.initializer, name=f'b_bi_{k}_{name_scope}'
            )
            all_weights[f'W_trans_{k}'] = tf.keras.layers.Dense(
                self.weight_size_list[k+1], kernel_initializer=self.initializer, input_dim=2*self.weight_size_list[k+1],
                name=f'W_trans_{k}_{name_scope}'
            )

        return all_weights

    def call(self, inputs, training=False, node_dropout=0.0, mess_dropout=None):
        mirnas_s = inputs['mirnas_s']
        disease = inputs['disease']
        label_s = inputs['label_s']
        mirnas_t = inputs['mirnas_t']
        target = inputs['target']
        label_t = inputs['label_t']

        if mess_dropout is None:
            mess_dropout = [0.0] * self.n_layers

        # Get initial embeddings directly from Embedding layers
        u_g_embeddings_s = self.weights_source['mirna_embedding'](mirnas_s)
        i_g_embeddings_s = self.weights_source['item_embedding'](disease)
        u_g_embeddings_t = self.weights_target['mirna_embedding'](mirnas_t)
        i_g_embeddings_t = self.weights_target['item_embedding'](target)

        # Apply graph convolution if needed (simplified for now, can reintroduce _create_embed later)
        if self.n_layers > 0:
            ua_embeddings_s, ia_embeddings_s, ua_embeddings_t, ia_embeddings_t = self._create_embed(
                self.weights_source, self.weights_target, self.norm_adj_s, self.norm_adj_t, 
                node_dropout, mess_dropout, training
            )
            u_g_embeddings_s = tf.gather(ua_embeddings_s, mirnas_s)
            i_g_embeddings_s = tf.gather(ia_embeddings_s, disease)
            u_g_embeddings_t = tf.gather(ua_embeddings_t, mirnas_t)
            i_g_embeddings_t = tf.gather(ia_embeddings_t, target)

        scores_s = self.get_scores(u_g_embeddings_s, i_g_embeddings_s)
        scores_t = self.get_scores(u_g_embeddings_t, i_g_embeddings_t)

        return scores_s, scores_t, (u_g_embeddings_s, i_g_embeddings_s, label_s), (u_g_embeddings_t, i_g_embeddings_t, label_t)

    def _split_A_hat(self, X, n_items):
        A_fold_hat = []
        fold_len = (self.n_mirnas + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = self.n_mirnas + n_items if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, n_items, keep_prob):
        A_fold_hat = []
        fold_len = (self.n_mirnas + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = self.n_mirnas + n_items if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, float(keep_prob), n_nonzero_temp))
        return A_fold_hat

    def s_t_la2add_layer(self, input_s, input_t, lambda_s, lambda_t, domain_laplace):
        u_g_embeddings_s, i_g_embeddings_s = tf.split(input_s, [self.n_mirnas, self.n_disease], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(input_t, [self.n_mirnas, self.n_target], 0)
        laplace_s = tf.constant(self.domain_laplace[:, 0], name='laplace_s')
        laplace_t = tf.constant(self.domain_laplace[:, 1], name='laplace_t')
        u_g_embeddings_s_lap = tf.transpose(tf.add(laplace_s * tf.transpose(u_g_embeddings_s), laplace_t * tf.transpose(u_g_embeddings_t)))
        u_g_embeddings_t_lap = u_g_embeddings_s_lap
        u_g_embeddings_s_lam = tf.add(lambda_s * u_g_embeddings_s, (1 - lambda_s) * u_g_embeddings_t)
        u_g_embeddings_t_lam = tf.add((1 - lambda_t) * u_g_embeddings_s, lambda_t * u_g_embeddings_t)
        u_g_embeddings_s = (u_g_embeddings_s_lap + u_g_embeddings_s_lam) / 2
        u_g_embeddings_t = (u_g_embeddings_t_lap + u_g_embeddings_t_lam) / 2
        ego_embeddings_s = tf.concat([u_g_embeddings_s, i_g_embeddings_s], axis=0)
        ego_embeddings_t = tf.concat([u_g_embeddings_t, i_g_embeddings_t], axis=0)
        return ego_embeddings_s, ego_embeddings_t

    def _create_embed(self, weights_s, weights_t, norm_adj_s, norm_adj_t, node_dropout, mess_dropout, training):
        def one_graph_layer_gcf(A_fold_hat, ego_embeddings, weights, k):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings + tf.multiply(ego_embeddings, side_embeddings)
            if training:
                dropout_rate = float(mess_dropout[k])
                ego_embeddings = tf.nn.dropout(ego_embeddings, rate=dropout_rate)
            return ego_embeddings

        keep_prob = 1.0 - float(node_dropout) if self.node_dropout_flag and training else 1.0

        if self.node_dropout_flag and training:
            A_fold_hat_s = self._split_A_hat_node_dropout(norm_adj_s, self.n_disease, keep_prob)
            A_fold_hat_t = self._split_A_hat_node_dropout(norm_adj_t, self.n_target, keep_prob)
        else:
            A_fold_hat_s = self._split_A_hat(norm_adj_s, self.n_disease)
            A_fold_hat_t = self._split_A_hat(norm_adj_t, self.n_target)
        # Use full index tensors for graph convolution
        all_mirnas = tf.range(self.n_mirnas, dtype=tf.int32)
        all_disease = tf.range(self.n_disease, dtype=tf.int32)
        all_target = tf.range(self.n_target, dtype=tf.int32)
        ego_embeddings_s = tf.concat([weights_s['mirna_embedding'](all_mirnas), weights_s['item_embedding'](all_disease)], axis=0)
        ego_embeddings_t = tf.concat([weights_t['mirna_embedding'](all_mirnas), weights_t['item_embedding'](all_target)], axis=0)

        if self.connect_way == 'concat':
            all_embeddings_s = [ego_embeddings_s]
            all_embeddings_t = [ego_embeddings_t]
        elif self.connect_way == 'mean':
            all_embeddings_s = ego_embeddings_s
            all_embeddings_t = ego_embeddings_t

        for k in range(self.n_layers):
            if self.layer_fun == 'gcf':
                ego_embeddings_s = one_graph_layer_gcf(A_fold_hat_s, ego_embeddings_s, weights_s, k)
                ego_embeddings_t = one_graph_layer_gcf(A_fold_hat_t, ego_embeddings_t, weights_t, k)
            if k >= self.n_layers - self.n_interaction and self.n_interaction > 0:
                if self.fuse_type_in == 'la2add':
                    ego_embeddings_s, ego_embeddings_t = self.s_t_la2add_layer(
                        ego_embeddings_s, ego_embeddings_t, self.lambda_s, self.lambda_t, self.domain_laplace
                    )

            norm_embeddings_s = tf.math.l2_normalize(ego_embeddings_s, axis=1)
            norm_embeddings_t = tf.math.l2_normalize(ego_embeddings_t, axis=1)

            if self.connect_way == 'concat':
                all_embeddings_s.append(norm_embeddings_s)
                all_embeddings_t.append(norm_embeddings_t)
            elif self.connect_way == 'mean':
                all_embeddings_s += norm_embeddings_s
                all_embeddings_t += norm_embeddings_t

        if self.connect_way == 'concat':
            all_embeddings_s = tf.concat(all_embeddings_s, 1)
            all_embeddings_t = tf.concat(all_embeddings_t, 1)
        elif self.connect_way == 'mean':
            all_embeddings_s = all_embeddings_s / (self.n_layers + 1)
            all_embeddings_t = all_embeddings_t / (self.n_layers + 1)

        u_g_embeddings_s, i_g_embeddings_s = tf.split(all_embeddings_s, [self.n_mirnas, self.n_disease], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(all_embeddings_t, [self.n_mirnas, self.n_target], 0)
        return u_g_embeddings_s, i_g_embeddings_s, u_g_embeddings_t, i_g_embeddings_t

    def get_scores(self, mirnas, pos_items):
        return tf.reduce_sum(tf.multiply(mirnas, pos_items), axis=1)

    def create_cross_loss(self, mirnas, pos_items, label, scores):
        regularizer = tf.nn.l2_loss(mirnas) + tf.nn.l2_loss(pos_items)
        regularizer = regularizer / self.batch_size
        # Chuyển label sang float32 để khớp với logits
        label = tf.cast(label, tf.float32)
        mf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=scores))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32)
        return mf_loss, emb_loss, reg_loss

    def compute_loss(self, predictions, additional_data):
        scores_s, scores_t = predictions
        u_g_embeddings_s, i_g_embeddings_s, label_s = additional_data[0]
        u_g_embeddings_t, i_g_embeddings_t, label_t = additional_data[1]

        mf_loss_s, emb_loss_s, reg_loss_s = self.create_cross_loss(u_g_embeddings_s, i_g_embeddings_s, label_s, scores_s)
        mf_loss_t, emb_loss_t, reg_loss_t = self.create_cross_loss(u_g_embeddings_t, i_g_embeddings_t, label_t, scores_t)

        loss_source = mf_loss_s + emb_loss_s + reg_loss_s
        loss_target = mf_loss_t + emb_loss_t + reg_loss_t
        total_loss = self.weight_source * loss_source + self.weight_target * loss_target
        return total_loss, loss_source, loss_target

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        noise_shape = [n_nonzero_elems]
        random_tensor = float(keep_prob) + tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(X, dropout_mask)
        return pre_out * (1. / float(keep_prob))