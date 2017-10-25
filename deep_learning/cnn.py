import tensorflow as tf


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
        # 第一平均池化层
        with tf.name_scope('pool1'):
            pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二卷积层
        with tf.name_scope('conv2'):
            with tf.name_scope('conv'):
                conv2 = tf.nn.bias_add(
                    tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),
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
            rnn_inputs = tf.reshape(drop1, shape=[1, -1, 1024])
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
            # rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.keep_prob2)
            init_s = rnn_cell.zero_state(1, dtype=tf.float32)
            rnn_outputs, final_s = tf.nn.dynamic_rnn(
                rnn_cell,  # cell you have chosen
                rnn_inputs,  # input
                initial_state=init_s,  # the initial hidden state
                time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)

            )
            rnn_outputs = tf.reshape(rnn_outputs, shape=[-1, 512])

        # 第一全连接层
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

if __name__ == '__main__':
    net = GrainNetwork()
    net.paint()
