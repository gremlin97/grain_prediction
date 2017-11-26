import random

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp


class BatchManager(object):
    def __init__(self, train_barns, test_barn, grain_type, dt_from, dt_to, pre_days):
        train_barns = train_barns[:]
        train_barns.remove(test_barn)
        self.train_barns = train_barns
        self.test_barn = test_barn
        # 天气和层温数据
        self.air_temp = pd.read_csv('../DL_data/temperature/JiangSu.csv')
        self.air_temp = self.air_temp[self.air_temp['city'] == '淮安']
        if grain_type == 'rice':
            self.layer_temp = pd.read_csv('../DL_data/layers_temperature/rice_layer_4.csv')
        elif grain_type == 'wheat':
            self.layer_temp = pd.read_csv('../DL_data/layers_temperature/wheat_layer_4.csv')
        self.layer_temp = self.layer_temp[self.layer_temp['平均温度'] < 45]
        # 粮仓温度点时间数据
        self.train_barns_dt = {}
        self.test_barn_dt = []
        for barn in barns:
            barn_layer_time = set(self.layer_temp[self.layer_temp['仓库信息'] == str(barn) + '仓']['粮情时间'])
            self.train_barns_dt[barn] = []
            barn_path = '../DL_data/points_temperature/' + str(barn) + '仓/'
            barn_dir = os.listdir(barn_path)
            for file_name in barn_dir:
                dt = file_name.split('.')[0]
                if dt in barn_layer_time:
                    self.train_barns_dt[barn].append(dt)
            self.train_barns_dt[barn] = pd.Series(self.train_barns_dt[barn])
            self.train_barns_dt[barn] = self.train_barns_dt[barn][self.train_barns_dt[barn] >= dt_from]
            self.train_barns_dt[barn] = self.train_barns_dt[barn][self.train_barns_dt[barn] <= dt_to]
            self.train_barns_dt[barn].sort_values(inplace=True, ascending=True)

        barn_path = '../DL_data/points_temperature/' + str(test_barn) + '仓/'
        barn_dir = os.listdir(barn_path)
        for file_name in barn_dir:
            dt = file_name.split('.')[0]
            self.test_barn_dt.append(dt)
        self.test_barn_dt = pd.Series(self.test_barn_dt)
        self.test_barn_dt = self.test_barn_dt[self.test_barn_dt >= dt_from]
        self.test_barn_dt = self.test_barn_dt[self.test_barn_dt <= dt_to]
        self.test_barn_dt.sort_values(inplace=True, ascending=True)

        self.train_barn_index = 0
        self.train_dt_index = 0
        self.test_dt_index = 0
        self.STEP_SIZE = 5
        # cur_barn, images, features, lables = self.whole_train()
        # self.X_scaler = pp.RobustScaler().fit(np.array(images).reshape((-1, 6 * 6 * 4)))
        # self.F_scaler = pp.RobustScaler().fit(features)
        self.data_cache = {}
        self.pre_days = pre_days

    def truncation(self, *batches):
        result = []
        for batch in batches:
            batch = np.array(batch)
            size = batch.shape[0]
            # re_size = int(size / self.STEP_SIZE) * self.STEP_SIZE
            # result.append(batch[0:re_size])
            if size < 20:
                result.append([])
            else:
                result.append(batch)
        return result

    def whole_train(self):
        points_batch = []  # 粮食温度点
        other_features_batch = []  # 其他特征
        output_batch = []  # 输出
        barns = self.train_barns
        for _ in barns:
            _, pb, fb, ob, days = self.next_train_batch(1000)
            points_batch = points_batch + pb
            other_features_batch = other_features_batch + fb
            output_batch = output_batch + ob
        return 'whole', points_batch, other_features_batch, output_batch

    def next_train_batch(self, batch_size):
        barn_now = self.train_barns[self.train_barn_index]
        if barn_now in self.data_cache:
            self.train_barn_index = (self.train_barn_index + 1) % len(self.train_barns)
            return self.data_cache[barn_now]
        points_batch = []  # 粮食温度点
        other_features_batch = []  # 其他特征
        output_batch = []  # 输出
        size = 0
        days = []
        months = []
        step_count = 1
        while size < batch_size:
            flag = False
            pt, of, pre, pre_day, month = None, None, None, None, None
            while not flag:
                flag, pt, of, pre, pre_day, month = self.next(barn_now, self.train_dt_index)
                self.train_dt_index += 1
                if self.train_dt_index == len(self.train_barns_dt[barn_now]):
                    self.train_barn_index = (self.train_barn_index + 1) % len(self.train_barns)
                    # self.train_dt_index = random.randint(0, 4)
                    self.train_dt_index = 0

                    # points_batch, other_features_batch = self.truncation(points_batch,
                    #                                                      other_features_batch
                    #                                                      )
                    if len(points_batch) == 0:
                        return self.next_train_batch(batch_size)
                    self.data_cache[barn_now] = (
                        barn_now, points_batch, other_features_batch, output_batch, days, months)
                    # return barn_now, points_batch, other_features_batch, output_batch, days
                    return self.data_cache[barn_now]
            points_batch.append(pt)
            other_features_batch.append(of)
            # if step_count % self.STEP_SIZE == 0:
            output_batch.append(pre)
            days.append(pre_day)
            months.append(month)
            step_count += 1
            size += 1
        # points_batch, other_features_batch = self.truncation(points_batch,
        #                                                      other_features_batch)
        if len(points_batch) == 0:
            return self.next_train_batch(batch_size)
        self.data_cache[barn_now] = (barn_now, points_batch, other_features_batch, output_batch, days, months)
        # return barn_now, points_batch, other_features_batch, output_batch, days
        return self.data_cache[barn_now]

    def next_test_batch(self, batch_size):
        if self.test_barn in self.data_cache:
            return self.data_cache[self.test_barn]
        points_batch = []  # 粮食温度点
        other_features_batch = []  # 其他特征
        output_batch = []  # 输出
        size = 0
        days = []
        months = []
        step_count = 1
        while size < batch_size:
            flag = False
            pt, of, pre, pre_day, month = None, None, None, None, None
            while not flag:
                flag, pt, of, pre, pre_day, month = self.next(self.test_barn, self.test_dt_index)
                self.test_dt_index += 1
                if self.test_dt_index == len(self.test_barn_dt):
                    self.test_dt_index = 0
                    # points_batch, other_features_batch = self.truncation(points_batch,
                    #                                                            other_features_batch)
                    if len(points_batch) == 0:
                        return self.next_test_batch(batch_size)
                    self.data_cache[self.test_barn] = points_batch, other_features_batch, output_batch, days, months
                    # return points_batch, other_features_batch, output_batch, days
                    return self.data_cache[self.test_barn]
            points_batch.append(pt)
            other_features_batch.append(of)
            # if step_count % self.STEP_SIZE == 0:
            output_batch.append(pre)
            days.append(pre_day)
            months.append(month)
            step_count += 1
            size += 1
        # print(output_batch)
        # points_batch, other_features_batch = self.truncation(points_batch,
        #                                                            other_features_batch)
        if len(points_batch) == 0:
            return self.next_test_batch(batch_size)
        self.data_cache[self.test_barn] = points_batch, other_features_batch, output_batch, days, months
        # return points_batch, other_features_batch, output_batch, days
        return self.data_cache[self.test_barn]

    def next(self, barn, dt_index):
        if barn in self.train_barns:
            dts = self.train_barns_dt[barn]  # 该仓的时间
        else:
            dts = self.test_barn_dt
        layer_temp = self.layer_temp[self.layer_temp['仓库信息'] == str(barn) + '仓']
        dt = dts.iloc[dt_index]  # 当前时间
        delta = pd.to_timedelta(self.pre_days)
        pre_dt = (pd.to_datetime(dt) + delta).strftime('%Y-%m-%d')  # 十天后
        air_df = self.air_temp[[dt < date <= pre_dt for date in self.air_temp['date']]]
        path = '../DL_data/points_temperature/' + str(barn) + '仓/'
        dt_path = path + dt + '.npy'
        # 如果当前温度和预测温度存在，且之间的天气温度存在足够
        other_features = []
        if os.path.exists(dt_path) \
                and pre_dt in set(layer_temp['粮情时间']) \
                and air_df.shape[0] >= 8 \
                and dt in set(self.air_temp['date']):

            air_now = self.air_temp[self.air_temp['date'] == dt]['average'].iloc[0]
            if air_df.shape[0] != 10:
                air_df = self.fill_data(air_df, pre_dt)
            points_temp = np.load(dt_path)
            # pre_temp = layer_temp[layer_temp['粮情时间'] == pre_dt]['平均温度'].iloc[0]
            # print('points shape:', points_temp.shape)
            pre_path = path + pre_dt + '.npy'
            pre_temp = np.load(pre_path)[0, 1, 1]
            # 十一天气温加时间（1-365）
            other_features.append(air_now)

            other_features.extend(air_df['average'])
            day_in_year = int(pd.to_datetime(pre_dt).strftime('%j'))
            # month 0-11
            month = int(pd.to_datetime(pre_dt).strftime('%m')) - 1
            # day_in_year
            other_features.append(day_in_year / 365 - 0.5)
            return True, np.array(points_temp).transpose([2, 1, 0]), np.array(other_features), pre_temp, pre_dt, month
        else:
            return False, None, None, None, None, None

    def fill_data(self, df, pre_dt):
        mean = df['average'].mean()
        dts = pd.date_range(end=pre_dt, periods=10)
        for dt in dts:
            if dt.strftime('%Y-%m-%d') not in set(df['date']):
                df_temp = pd.DataFrame({'date': [dt.strftime('%Y-%m-%d')], 'average': [mean]})
                # df = df.append({'date': dt.strftime('%Y-%m-%d'), 'average': mean}, ignore_index=True)
                df = pd.concat([df, df_temp])
                df = df.sort_values(by=['date'])
        return df


class GrainNetwork(object):
    def __init__(self, output):
        # 初始化权值和偏置
        with tf.variable_scope("weights"):
            # 权重
            self.weights = {
                # 6*6*4->6*6*20
                'conv1': tf.get_variable('W_conv1', [2, 2, 4, 20],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 6*6*20->6*6*40->3*3*40
                'conv2': tf.get_variable('W_conv2', [2, 2, 20, 40],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 3*3*40+11+2->1024
                'fc1': tf.get_variable('W_fc1', [3 * 3 * 40 + 11 + 1, 800],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                # 1024->512
                # 512->64
                'fc2': tf.get_variable('W_fc2', [800, 500],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                # 256->64
                'fc3': tf.get_variable('W_fc3', [500, 300],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                # 64->1
                'output': tf.get_variable('W_output', [300, 1],
                                          initializer=tf.contrib.layers.xavier_initializer()),
            }

            # 偏置
            self.biases = {
                'conv1': tf.get_variable('b_conv1', [20, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('b_conv2', [40, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('b_fc1', [800, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1_t': tf.get_variable('bt_fc1', [12, 800],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('b_fc2', [500, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2_t': tf.get_variable('bt_fc2', [12, 500],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc3': tf.get_variable('b_fc3', [300, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc3_t': tf.get_variable('bt_fc3', [12, 300],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'output': tf.get_variable('b_output', [1, ],
                                          initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'output_t': tf.get_variable('bt_output', [12, 1],
                                            initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
            }
            self.keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
            self.keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')
            # 输入类图片和温度
        with tf.name_scope('inputs'):
            self.images = tf.placeholder(tf.float32, [None, 6, 6, 4], name='images')
            self.temperatures = tf.placeholder(tf.float32, [None, 11 + 1], name='temperatures')
        with tf.name_scope('targets'):
            self.targets = tf.placeholder(tf.float32, [None, 1], name='targets')
            self.months = tf.placeholder(tf.int32, [None, 1], name='months')
        self.init_s = None
        self.final_s = None
        self.batch_tensor = tf.placeholder(tf.int32, [])  # 用于保留batch_size
        self.STEP_SIZE = 5
        self.month_map = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        self.wheat_weights = [0.0402227666955,
                              0.031469607781,
                              0.0441908958449,
                              0.0619128091163,
                              0.0779277661702,
                              0.0949637405299,
                              0.117478400516,
                              0.133751192474,
                              0.130655772891,
                              0.111494953891,
                              0.0928757160844,
                              0.0630563780055, ]
        self.rice_weights = [0.032534470896,
                             0.0245423446013,
                             0.035091664762,
                             0.0551522064438,
                             0.0759471321252,
                             0.0966625524964,
                             0.1271142898,
                             0.145481189775,
                             0.144061524447,
                             0.120033676453,
                             0.0919763712315,
                             0.0514025769694,
                             ]
        for m in range(12):
            if self.month_map[m] == 1:
                self.wheat_weights[m] *= 2
                self.rice_weights[m] *= 2
        self.output_file = open(output, 'w+')

    def add_layers(self):
        # 第一卷积层
        with tf.name_scope('conv1'):
            with tf.name_scope('conv'):
                conv1 = tf.nn.bias_add(
                    tf.nn.conv2d(self.images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'),
                    self.biases['conv1']
                )
            with tf.name_scope('relu'):
                conv1 = tf.nn.relu(conv1)

        # # 第一平均池化层
        # with tf.name_scope('pool1'):
        #     pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二卷积层
        with tf.name_scope('conv2'):
            with tf.name_scope('conv'):
                conv2 = tf.nn.bias_add(
                    tf.nn.conv2d(conv1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),
                    self.biases['conv2']
                )
            with tf.name_scope('relu'):
                conv2 = tf.nn.relu(conv2)

        # 第二平均池化层
        with tf.name_scope('pool2'):
            pool2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            # pool2 = tf.nn.dropout(pool2, self.keep_prob1)
        # 第一全连接层
        with tf.name_scope('fc1'):
            with tf.name_scope('fc1'):
                flatten = tf.reshape(pool2, [-1, 3 * 3 * 40], name='flatten')
                connectted = tf.concat([self.temperatures, flatten], axis=1, name='concat')
                fc1 = tf.matmul(connectted, self.weights['fc1']) + self.biases['fc1']
                month_to_bias = tf.gather(self.month_map, tf.reshape(self.months, [-1, ]))
                fc1 = fc1 + tf.gather(self.biases['fc1_t'], month_to_bias)
            with tf.name_scope('drop1'):
                # 第一dropout层
                drop1 = tf.nn.dropout(fc1, self.keep_prob1)
                drop1 = tf.nn.relu(drop1)

        # RNN层
        # with tf.name_scope('rnn'):
        #     rnn_inputs = tf.reshape(drop1, shape=[-1, self.STEP_SIZE, 1024])
        #     rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
        #     # rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob2)
        #     # batch_size = tf.shape(self.batch_tensor)[0] / self.STEP_SIZE
        #     self.init_s = rnn_cell.zero_state(batch_size=self.batch_tensor, dtype=tf.float32)
        #     rnn_outputs, self.final_s = tf.nn.dynamic_rnn(
        #         rnn_cell,  # cell you have chosen
        #         rnn_inputs,  # input
        #         initial_state=self.init_s,  # the initial hidden state
        #         time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
        #     )
        #     rnn_outputs = rnn_outputs[:, -1, :]
        #     rnn_outputs = tf.reshape(rnn_outputs, shape=[-1, 512])
        # 第二全连接层
        with tf.name_scope('fc2'):
            with tf.name_scope('fc2'):
                fc2 = tf.matmul(drop1, self.weights['fc2']) + self.biases['fc2']
                month_to_bias = tf.gather(self.month_map, tf.reshape(self.months, [-1, ]))
                fc2 = fc2 + tf.gather(self.biases['fc2_t'], month_to_bias)

            with tf.name_scope('drop2'):
                # 第一dropout层
                drop2 = tf.nn.dropout(fc2, self.keep_prob2)
                drop2 = tf.nn.relu(drop2)

        # 第三全连接层
        with tf.name_scope('fc3'):
            with tf.name_scope('fc3'):
                fc3 = tf.matmul(drop2, self.weights['fc3']) + self.biases['fc3']
                month_to_bias = tf.gather(self.month_map, tf.reshape(self.months, [-1, ]))
                fc3 = fc3 + tf.gather(self.biases['fc3_t'], month_to_bias)
                # print(tf.gather(self.biases['fc3_t'], tf.reshape(self.months, [-1, ])).shape)
            with tf.name_scope('drop2'):
                #     # 第一dropout层
                drop3 = tf.nn.dropout(fc3, self.keep_prob2)
                drop3 = tf.nn.relu(drop3)

        # 输出层
        with tf.name_scope('output'):
            output = tf.matmul(drop3, self.weights['output']) + self.biases['output']
            month_to_bias = tf.gather(self.month_map, tf.reshape(self.months, [-1, ]))
            output = output + tf.gather(self.biases['output_t'], month_to_bias)
        return output

    def loss(self, pred, grain_type='rice'):
        # alpha = 0.8
        # beta = 1.8
        # alpha = 0.82
        # beta = 1.66
        alpha = 1
        beta = 1
        with tf.name_scope('loss'):
            if grain_type == 'rice':
                month_weights = self.rice_weights
            else:
                month_weights = self.wheat_weights
            weights = tf.gather(month_weights, tf.reshape(self.months, [-1, ]))
            weights = tf.reshape(weights, [-1, 1])
            mse = tf.losses.mean_squared_error(labels=self.targets, predictions=pred, weights=weights)
            # mse = tf.reduce_mean(tf.where(tf.less(self.targets, pred), alpha * tf.square(pred - self.targets),
            #                              beta * tf.square(pred - self.targets)))
            tf.summary.scalar('loss', mse)
        return mse

    def accuracy(self, pred, grain_type='rice'):
        u, v = tf.constant(0, tf.float32), tf.constant(0, tf.float32)
        y_mean = tf.reduce_mean(pred)
        for m in range(0, 12):
            month_mask = tf.equal(self.months, m)
            num = tf.cast(tf.count_nonzero(tf.cast(month_mask, tf.float32)), tf.float32)
            y_true = tf.boolean_mask(self.targets, month_mask)
            y_pre = tf.boolean_mask(pred, month_mask)
            # y_true_mean = tf.reduce_mean(y_true)
            if grain_type == 'rice':
                w = self.rice_weights[m]
            else:
                w = self.wheat_weights[m]
            u = tf.add(tf.div(tf.multiply(w, tf.reduce_sum(tf.square(y_true - y_pre))), num),
                       u)
            v = tf.add(tf.div(tf.multiply(w, tf.reduce_sum(tf.square(y_true - y_mean))), num),
                       v)
        accuracy = tf.subtract(1., tf.div(u, v))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def optimizer(self, loss, lr=0.01):
        train_optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss)
        return train_optimizer

    def paint(self):
        pred = self.add_layers()
        loss = self.loss(pred)
        optimizer = self.optimizer(loss, lr=0.01)
        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

    def train(self, max_iter, train_barns, test_barn, grain_type, pre_days):
        pred = self.add_layers()
        loss = self.loss(pred)
        optimizer = self.optimizer(loss, lr=0.01)
        accuracy = self.accuracy(pred, grain_type)

        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        plt.figure(1, figsize=(15, 5))
        plt.ion()
        batch_manager = BatchManager(train_barns, test_barn, grain_type, '2015-09-01', '2017-9-20', pre_days)
        pre_barn = None
        final_s_ = None
        for iter in range(max_iter):
            cur_barn, images, features, lables, days, months = batch_manager.next_train_batch(500)
            # images = batch_manager.X_scaler.transform(np.array(images).reshape((-1, 6 * 6 * 4)))
            # images = images.reshape((-1, 6, 6, 4))
            # features = batch_manager.F_scaler.transform(features)
            # cur_barn, images, features, lables = batch_manager.whole_train()
            lables = np.array(lables)[:, np.newaxis]
            images = np.array(images)
            features = np.array(features)
            # months = list(map(lambda m: self.month_map[m], months))
            months = np.array(months)[:, np.newaxis]
            # print(images.shape)
            if cur_barn is pre_barn:
                feed_dict = {self.images: images, self.targets: lables,
                             self.months: months,
                             self.temperatures: features,
                             self.keep_prob1: 0.5, self.keep_prob2: 0.5,
                             self.batch_tensor: images.shape[0] / self.STEP_SIZE}
            else:
                feed_dict = {self.images: images, self.targets: lables,
                             self.months: months,
                             self.temperatures: features,
                             self.keep_prob1: 0.5, self.keep_prob2: 0.5,
                             self.batch_tensor: images.shape[0] / self.STEP_SIZE}
            pre_barn = cur_barn
            # print(lables, features)
            _, pred_, loss_ = sess.run([optimizer, pred, loss], feed_dict)
            # plt.scatter(days, lables.reshape(-1, ), s=20, edgecolor="black",
            #             c="darkorange", label="data")
            # # plt.plot(days, test_y.reshape(-1, ), 'r-')
            # plt.plot(days, pred_.reshape(-1, ), 'b-', label='prediction')
            # plt.legend()
            # # plt.ylim((-1.2, 1.2))
            # plt.draw()
            # plt.pause(0.05)
            # plt.clf()
            print(cur_barn, 'iter', iter, 'loss', loss_)
            if iter % 5 == 0:
                batch_manager.test_dt_index = 0
                test_imgs, test_f, test_y, days, months = batch_manager.next_test_batch(500)
                # test_imgs = batch_manager.X_scaler.transform(np.array(test_imgs).reshape((-1, 6 * 6 * 4)))
                # test_imgs = test_imgs.reshape((-1, 6, 6, 4))
                # test_f = batch_manager.F_scaler.transform(test_f)
                test_y = np.array(test_y)[:, np.newaxis]
                test_imgs = np.array(test_imgs)
                test_f = np.array(test_f)
                months = np.array(months)[:, np.newaxis]
                feed_dict = {self.images: test_imgs, self.targets: test_y,
                             self.months: months,
                             self.temperatures: test_f,
                             self.keep_prob1: 1, self.keep_prob2: 1,
                             self.batch_tensor: test_imgs.shape[0] / self.STEP_SIZE}
                rs, pred_, loss_, accuracy_ = sess.run([merged, pred, loss, accuracy], feed_dict)
                writer.add_summary(rs, iter)

                print('loss', loss_, 'accuracy', accuracy_)
                plt.scatter(pd.to_datetime(days), test_y.reshape(-1, ), s=20, edgecolor="black",
                            c="darkorange", label="data")
                # plt.plot(days, test_y.reshape(-1, ), 'r-')
                plt.plot(pd.to_datetime(days), pred_.reshape(-1, ), 'b-', label='prediction')
                plt.legend()
                # plt.ylim((-1.2, 1.2))
                plt.draw()
                plt.pause(0.01)
                # plt.savefig('../figures/rice_barn9_point.png')
                plt.clf()

                # plt.ioff()
                # plt.show()


if __name__ == '__main__':
    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]

    barns = [2, 3, 5, 6, 7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 29, 30, 32, 34, 35]
    barn = 9
    net = GrainNetwork('../DL_data/day10_barn9.ac')
    net.train(1200, barns, barn, 'rice', '10 days')
    # net.paint()
