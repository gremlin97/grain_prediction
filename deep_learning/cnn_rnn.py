import random

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class BatchManager(object):
    def __init__(self, train_barns, test_barn, grain_type, dt_from, dt_to):
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

        # 粮仓温度点时间数据
        self.train_barns_dt = {}
        self.test_barn_dt = []
        for barn in barns:
            self.train_barns_dt[barn] = []
            barn_path = '../DL_data/points_temperature/' + str(barn) + '仓/'
            barn_dir = os.listdir(barn_path)
            for file_name in barn_dir:
                dt = file_name.split('.')[0]
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

    def next_train_batch(self, batch_size):
        barn_now = self.train_barns[self.train_barn_index]
        points_batch = []  # 粮食温度点
        other_features_batch = []  # 其他特征
        output_batch = []  # 输出
        size = 0
        step_count = 1
        while size < batch_size:
            flag = False
            pt, of, pre = None, None, None
            while not flag:
                flag, pt, of, pre, _ = self.next(barn_now, self.train_dt_index)
                self.train_dt_index += 1
                if self.train_dt_index == len(self.train_barns_dt[barn_now]):
                    self.train_barn_index = (self.train_barn_index + 1) % len(self.train_barns)
                    self.train_dt_index = random.randint(0, 4)
                    points_batch, other_features_batch = self.truncation(points_batch,
                                                                         other_features_batch
                                                                         )
                    if len(points_batch) == 0:
                        return self.next_train_batch(batch_size)
                    return barn_now, points_batch, other_features_batch, output_batch
            points_batch.append(pt)
            other_features_batch.append(of)
            if step_count % self.STEP_SIZE == 0:
                output_batch.append(pre)
            step_count += 1
            size += 1
        points_batch, other_features_batch = self.truncation(points_batch,
                                                             other_features_batch)
        if len(points_batch) == 0:
            return self.next_train_batch(batch_size)
        return barn_now, points_batch, other_features_batch, output_batch

    def next_test_batch(self, batch_size):
        points_batch = []  # 粮食温度点
        other_features_batch = []  # 其他特征
        output_batch = []  # 输出
        size = 0
        days = []
        step_count = 1
        while size < batch_size:
            flag = False
            pt, of, pre, day_in_year = None, None, None, None
            while not flag:
                flag, pt, of, pre, day_in_year = self.next(self.test_barn, self.test_dt_index)
                self.test_dt_index += 1
                if self.test_dt_index == len(self.test_barn_dt):
                    self.test_dt_index = random.randint(0, 4)
                    points_batch, other_features_batch = self.truncation(points_batch,
                                                                               other_features_batch)
                    if len(points_batch) == 0:
                        return self.next_test_batch(batch_size)
                    return points_batch, other_features_batch, output_batch, days
            points_batch.append(pt)
            other_features_batch.append(of)
            if step_count % self.STEP_SIZE == 0:
                output_batch.append(pre)
                days.append(day_in_year)
            step_count += 1
            size += 1
        print(output_batch)
        points_batch, other_features_batch = self.truncation(points_batch,
                                                                   other_features_batch)
        if len(points_batch) == 0:
            return self.next_test_batch(batch_size)
        return points_batch, other_features_batch, output_batch, days

    def next(self, barn, dt_index):
        if barn in self.train_barns:
            dts = self.train_barns_dt[barn]  # 该仓的时间
        else:
            dts = self.test_barn_dt
        layer_temp = self.layer_temp[self.layer_temp['仓库信息'] == str(barn) + '仓']
        dt = dts.iloc[dt_index]  # 当前时间
        delta = pd.to_timedelta('10 days')
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
            pre_temp = layer_temp[layer_temp['粮情时间'] == pre_dt]['平均温度'].iloc[0]
            # 十一天气温加时间（1-365）
            other_features.append(air_now)

            other_features.extend(air_df['average'])
            day_in_year = int(pd.to_datetime(dt).strftime('%j'))
            other_features.append(day_in_year)

            return True, np.array(points_temp).transpose([2, 1, 0]), np.array(other_features), pre_temp, day_in_year
        else:
            return False, None, None, None, None

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
    def __init__(self):
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
                'fc1': tf.get_variable('W_fc1', [3 * 3 * 40 + 11 + 1, 1024]),
                # 1024->512
                # 512->64
                'fc2': tf.get_variable('W_fc2', [512, 64]),
                # 64->1
                'output': tf.get_variable('W_output', [64, 1]),
            }

            # 偏置
            self.biases = {
                'conv1': tf.get_variable('b_conv1', [20, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('b_conv2', [40, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('b_fc1', [1024, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('b_fc2', [64, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'output': tf.get_variable('b_output', [1, ],
                                          initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
            }
            self.keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
            self.keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')
            # 输入类图片和温度
        with tf.name_scope('inputs'):
            self.images = tf.placeholder(tf.float32, [None, 6, 6, 4], name='images')
            self.temperatures = tf.placeholder(tf.float32, [None, 11 + 1], name='temperatures')
        with tf.name_scope('targets'):
            self.targets = tf.placeholder(tf.float32, [None, 1], name='targets')
        self.init_s = None
        self.final_s = None
        self.batch_tensor = tf.placeholder(tf.int32, [])  # 用于保留batch_size
        self.STEP_SIZE = 5

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

        # 第一全连接层
        with tf.name_scope('fc1'):
            with tf.name_scope('fc1'):
                flatten = tf.reshape(pool2, [-1, 3 * 3 * 40], name='flatten')
                connectted = tf.concat([self.temperatures, flatten], axis=1, name='concat')
                fc1 = tf.matmul(connectted, self.weights['fc1']) + self.biases['fc1']
            with tf.name_scope('drop1'):
                # 第一dropout层
                drop1 = tf.nn.dropout(fc1, self.keep_prob1)

        # RNN层
        with tf.name_scope('rnn'):
            rnn_inputs = tf.reshape(drop1, shape=[-1, self.STEP_SIZE, 1024])
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
            # rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob2)
            # batch_size = tf.shape(self.batch_tensor)[0] / self.STEP_SIZE
            self.init_s = rnn_cell.zero_state(batch_size=self.batch_tensor, dtype=tf.float32)
            rnn_outputs, self.final_s = tf.nn.dynamic_rnn(
                rnn_cell,  # cell you have chosen
                rnn_inputs,  # input
                initial_state=self.init_s,  # the initial hidden state
                time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
            )
            rnn_outputs = rnn_outputs[:, -1, :]
            rnn_outputs = tf.reshape(rnn_outputs, shape=[-1, 512])
        # 第二全连接层
        with tf.name_scope('fc2'):
            with tf.name_scope('fc2'):
                fc2 = tf.matmul(rnn_outputs, self.weights['fc2']) + self.biases['fc2']
            with tf.name_scope('drop2'):
                # 第一dropout层
                drop2 = tf.nn.dropout(fc2, self.keep_prob2)

        # 输出层
        with tf.name_scope('output'):
            output = tf.matmul(drop2, self.weights['output']) + self.biases['output']

        return output

    def loss(self, pred):
        with tf.name_scope('loss'):
            mse = tf.losses.mean_squared_error(labels=self.targets, predictions=pred)
        return mse

    def optimizer(self, loss, lr=0.01):
        train_optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        return train_optimizer

    def paint(self):
        pred = self.add_layers()
        loss = self.loss(pred)
        optimizer = self.optimizer(loss, lr=0.01)
        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

    def train(self, max_iter, train_barns, test_barn, grain_type):
        pred = self.add_layers()
        loss = self.loss(pred)
        optimizer = self.optimizer(loss, lr=0.01)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        plt.figure(1, figsize=(12, 5))
        plt.ion()
        batch_manager = BatchManager(train_barns, test_barn, grain_type, '2016-01-01', '2016-12-30')
        pre_barn = None
        final_s_ = None
        for iter in range(max_iter):
            cur_barn, images, features, lables = batch_manager.next_train_batch(20)
            lables = np.array(lables)[:, np.newaxis]
            images = np.array(images)
            features = np.array(features)
            print(images.shape)
            if cur_barn is pre_barn:
                feed_dict = {self.images: images, self.targets: lables,
                             self.temperatures: features,
                             self.init_s: final_s_, self.keep_prob1: 0.5, self.keep_prob2: 0.5,
                             self.batch_tensor: images.shape[0] / self.STEP_SIZE}
            else:
                feed_dict = {self.images: images, self.targets: lables,
                             self.temperatures: features,
                             self.keep_prob1: 0.8, self.keep_prob2: 0.8,
                             self.batch_tensor: images.shape[0] / self.STEP_SIZE}
            pre_barn = cur_barn
            # print(lables, features)
            _, final_s_, loss_ = sess.run([optimizer, self.final_s, loss], feed_dict)
            print('iter', iter, 'loss', loss_)
            if iter % 20 == 0:
                # batch_manager.test_dt_index = random.randint(0, 4)

                test_imgs, test_f, test_y, days = batch_manager.next_test_batch(190)
                test_y = np.array(test_y)[:, np.newaxis]
                test_imgs = np.array(test_imgs)
                test_f = np.array(test_f)
                print(test_y)

                feed_dict = {self.images: test_imgs, self.targets: test_y,
                             self.temperatures: test_f,
                             self.keep_prob1: 1, self.keep_prob2: 1,
                             self.batch_tensor: test_imgs.shape[0] / self.STEP_SIZE}
                pred_, loss_ = sess.run([pred, loss], feed_dict)
                print('loss', loss_)
                plt.plot(days, test_y.reshape(-1, ), 'r-')
                plt.plot(days, pred_.reshape(-1, ), 'b-')
                # plt.ylim((-1.2, 1.2))
                plt.draw()
                plt.pause(0.05)
                plt.clf()
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]

    barns = [1, 8, 10, 12, 15, 16, 20, 26, 27, 28, 31, 33]
    barn = 1
    net = GrainNetwork()
    net.train(100000, barns, barn, 'wheat')
