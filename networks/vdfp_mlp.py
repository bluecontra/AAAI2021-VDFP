"""
    Value Decomposed DDPG with Future Prediction Algorithm
    with a MLP for the preditive dynamics model
    (VDFP_MLP)

"""

import tensorflow as tf
import numpy as np


class VDFP(object):
    def __init__(self, s_dim, a_dim, sess,
                 lr_a=0.00025, lr_p=0.001, lr_r=0.0005,
                 gamma=0.99,
                 a_batch_size=64, m_batch_size=64,
                 memory_size=100000,
                 m_dim=100, z_dim=20,
                 sequence_length=64, min_sequence_length=16,
                 aggregate_sequence_length=64,
                 filter_sizes=None, filters_nums=None):

        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.aggregate_sequence_length = aggregate_sequence_length

        self.filter_sizes = [1, 2, 4, 8, 16, 32, 64] if filter_sizes is None else filter_sizes
        self.filters_nums = [20, 20, 10, 10, 5, 5, 5] if filters_nums is None else filters_nums

        self.s_dim, self.a_dim = s_dim, a_dim
        self.m_dim, self.z_dim = m_dim, z_dim

        self.lr_a, self.lr_r, self.lr_p = lr_a, lr_r, lr_p
        self.gamma = gamma
        self.a_batch_size, self.m_batch_size = a_batch_size, m_batch_size

        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, s_dim + a_dim + (s_dim + a_dim) * sequence_length + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = sess

        self.train_cnt = 0

        # -------------------------- placeholders ------------------------------
        self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.ss = tf.placeholder(tf.float32, [None, sequence_length, s_dim + a_dim], 's_sequence')

        self.u = tf.placeholder(tf.float32, [None, 1], 'utility')
        self.drop_rate = tf.placeholder(tf.float32, None, name='drop_rate')  # input Action

        self.a = self._build_a(self.s, trainable=True)
        self.m = self._build_m(self.ss)

        self.d = self._build_predictor(self.s, self.a)

        self.r = self._build_r(self.m)

        self.mm_gen = self._build_predictor(self.s, self.a, reuse=True)
        self.objective = self._build_r(self.mm_gen, reuse=True)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        m_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Measurement')
        p_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        # e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Return')
        # d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        self.a_loss = - tf.reduce_mean(self.objective)
        self.a_train = tf.train.AdamOptimizer(learning_rate=self.lr_a).minimize(self.a_loss, var_list=a_params)

        self.predict_loss = tf.reduce_mean(tf.square(self.m - self.d))
        self.predict_train = tf.train.AdamOptimizer(learning_rate=self.lr_p).minimize(self.predict_loss, var_list=p_params)

        self.r_loss = tf.reduce_mean(tf.square(self.u - self.r))
        self.r_train = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_loss, var_list=m_params + r_params)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: VDFP_MLP initialized.')
        print('==========================')

    def _print_hyperparams(self):
        print('------------------- Hyperparameters ----------------------')
        print('-- Sequence_Length:', self.sequence_length, self.min_sequence_length)
        print('-- z_dim:', self.z_dim)
        print('-- S_Dim:', self.s_dim)
        print('-- A_Dim:', self.a_dim)
        print('-- LR_Predictor:', self.lr_p)
        print('-- LR_Return:', self.lr_r)
        print('-- LR_Actor:', self.lr_a)
        print('-- Gamma:', self.gamma)
        print('-- Batch_Size:', self.a_batch_size, self.m_batch_size)
        print('-- Memory_Size:', self.memory_size)
        print('--')

    def _build_a(self, s, trainable):
        with tf.variable_scope('Actor'):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='l2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            a = tf.layers.dense(fc2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            return a

    def _build_predictor(self, s, a, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Predictor', reuse=reuse):
            concat = tf.concat([s, a], axis=1, name='s_a_concat')
            df1 = tf.layers.dense(concat, 400, activation=tf.nn.relu, name='df1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / (self.s_dim + self.a_dim))))
            df2 = tf.layers.dense(df1, 400, activation=tf.nn.relu, name='df2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            df3 = tf.layers.dense(df2, 200, activation=tf.nn.relu, name='df3', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            # FIXME 1226 1-dimension or more (?)
            d = tf.layers.dense(df3, self.m_dim, activation=None, name='Predictor', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            return d

    def _build_m(self, ss, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Measurement', reuse=reuse):
            if self.sequence_length > 64:
                # FIXME -- aggregate long-trajectory
                ss_resh = tf.reshape(ss, [-1, self.sequence_length * (self.s_dim + self.a_dim)
                                          // self.aggregate_sequence_length])

                ss_agg = tf.layers.dense(ss_resh, self.s_dim + self.a_dim, activation=tf.nn.relu, name='ss_aggre',
                                         trainable=trainable,
                                         kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / (
                                             self.sequence_length * (self.s_dim + self.a_dim)
                                             // self.aggregate_sequence_length
                                         ))))

                ss = tf.reshape(ss_agg, [-1, self.aggregate_sequence_length, self.s_dim + self.a_dim, 1])
            else:
                ss = tf.reshape(ss, [-1, self.sequence_length, self.s_dim + self.a_dim, 1])
            convs = [tf.squeeze(tf.layers.conv2d(inputs=ss,
                                                 filters=n,
                                                 kernel_size=[h, self.s_dim + self.a_dim],
                                                 activation=tf.nn.relu),
                                axis=2)
                     for h, n in zip(self.filter_sizes, self.filters_nums)]

            pools = [tf.squeeze(tf.layers.max_pooling1d(inputs=conv, pool_size=[conv.shape[1]], strides=1),
                                axis=1)
                     for conv in convs]

            concat = tf.concat(pools, axis=1, name='conv_concat')

            highway = tf.layers.dense(concat, sum(self.filters_nums), activation=None, name='highway', trainable=trainable,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / sum(self.filters_nums))))
            joint = tf.nn.sigmoid(highway) * tf.nn.relu(highway) + (1. - tf.nn.sigmoid(highway)) * concat
            dropout = tf.layers.dropout(inputs=joint, rate=self.drop_rate, name='dropout')

            m = tf.layers.dense(dropout, self.m_dim, activation=None, name='m', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / sum(self.filters_nums))))

            return m

    def _build_r(self, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Return', reuse=reuse):
            r = tf.layers.dense(m, 1, activation=None, name='return', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))

            return r

    def choose_action(self, s):
        return self.sess.run(self.a, {self.s: s[np.newaxis, :]})[0]

    def train_reward_model(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        b_u = batch_exps[:, self.s_dim + self.a_dim]
        b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length:]
        b_ss = np.reshape(b_saoff, [-1, self.sequence_length, self.s_dim + self.a_dim])

        r_loss, _ = self.sess.run([self.r_loss, self.r_train],
                                  feed_dict={self.s: b_s, self.ss: b_ss,
                                             self.u: b_u.reshape((-1, 1)),
                                             self.drop_rate: 0.2})

        return r_loss

    def train_predictor(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        b_a = batch_exps[:, self.s_dim: self.s_dim + self.a_dim]
        b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length:]

        b_ss = np.reshape(b_saoff, [-1, self.sequence_length, self.s_dim + self.a_dim])

        self.sess.run(self.predict_train, feed_dict={self.s: b_s, self.a: b_a,
                                                 self.ss: b_ss,
                                                 self.drop_rate: 0.0})

        # FIXME - losses are not calculated to reduce the training time (modify this if necessary)
        return 0

    def train_actor(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        self.sess.run(self.a_train, feed_dict={self.s: b_s, self.drop_rate: 0.0})

        self.train_cnt += 1

        # FIXME - losses are not calculated to reduce the training time (modify this if necessary)
        # return actor_loss
        return 0

    def store_experience(self, trajectory, is_padding=False):
        s_traj, a_traj, r_traj = trajectory

        # for the convenience of manipulation
        arr_s_traj = np.array(s_traj)
        arr_a_traj = np.array(a_traj)
        arr_r_traj = np.array(r_traj)

        zero_pads = np.zeros(shape=[self.sequence_length, self.s_dim + self.a_dim])

        # for i in range(len(s_traj) - self.sequence_length):
        for i in range(len(s_traj) - self.min_sequence_length):
            tmp_s = arr_s_traj[i]
            tmp_a = arr_a_traj[i]
            tmp_soff = arr_s_traj[i:i + self.sequence_length]
            tmp_aoff = arr_a_traj[i:i + self.sequence_length]
            tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

            tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
            tmp_saoff_padded_clip = tmp_saoff_padded[:self.sequence_length, :]
            tmp_soff_resh = tmp_saoff_padded_clip.reshape((self.s_dim + self.a_dim) * self.sequence_length)

            tmp_roff = arr_r_traj[i:i + self.sequence_length]
            tmp_u = np.matmul(tmp_roff, np.power(self.gamma, [j for j in range(len(tmp_roff))]))

            tmp_exp = tmp_s.tolist() + tmp_a.tolist() + [tmp_u] + tmp_soff_resh.tolist()

            index = self.pointer % self.memory_size
            self.memory[index, :] = tmp_exp
            self.pointer += 1

