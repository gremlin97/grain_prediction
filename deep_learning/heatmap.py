import seaborn as sns

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


class BatchManager(object):
    def __init__(self, train_barns, test_barn, grain_type, dt_from, dt_to, pre_days, pre_point):
        train_barns = train_barns[:]
        if test_barn in train_barns:
            train_barns.remove(test_barn)
        self.train_barns = train_barns
        self.test_barn = test_barn
        # 天气和层温数据
        self.air_temp = pd.read_csv('../DL_data/temperature/JiangSu.csv')
        self.air_temp = self.air_temp[self.air_temp['city'] == '淮安']
        if grain_type == 'rice':
            self.layer_temp = pd.read_csv('../DL_data/layers_temperature/rice_layer_1.csv')
        elif grain_type == 'wheat':
            self.layer_temp = pd.read_csv('../DL_data/layers_temperature/wheat_layer_1.csv')
        self.layer_temp = self.layer_temp[self.layer_temp['平均温度'] < 45]
        # 粮仓温度点时间数据
        self.train_barns_dt = {}
        self.test_barn_dt = []
        for barn in self.train_barns:
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

        if test_barn:
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
        self.pre_point = pre_point

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
        delta = pd.to_timedelta(str(self.pre_days) + ' days')
        pre_dt = (pd.to_datetime(dt) + delta).strftime('%Y-%m-%d')  # 十天后
        air_df = self.air_temp[[dt < date <= pre_dt for date in self.air_temp['date']]]
        air_df.sort_values('date', inplace=True)

        path = '../DL_data/points_temperature/' + str(barn) + '仓/'
        dt_path = path + dt + '.npy'
        pre_path = path + pre_dt + '.npy'
        # 如果当前温度和预测温度存在，且之间的天气温度存在足够
        other_features = []
        if os.path.exists(dt_path) \
                and os.path.exists(pre_path) \
                and pre_dt in set(layer_temp['粮情时间']) \
                and air_df.shape[0] >= self.pre_days * .6 \
                and dt in set(self.air_temp['date']):

            air_now = self.air_temp[self.air_temp['date'] == dt]['average'].iloc[0]
            if air_df.shape[0] != self.pre_days:
                air_df = self.fill_data(air_df, pre_dt)

            points_temp = np.load(dt_path)
            # print(points_temp.shape, dt, barn)
            pre_points = np.load(pre_path)
            # pre_temp = pre_points[self.pre_point[0], self.pre_point[1], self.pre_point[2]]
            pre_temp = pre_points
            # pre_temp = layer_temp[layer_temp['粮情时间'] == pre_dt]['平均温度'].iloc[0]

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
        dts = pd.date_range(end=pre_dt, periods=self.pre_days)
        for dt in dts:
            if dt.strftime('%Y-%m-%d') not in set(df['date']):
                df_temp = pd.DataFrame({'date': [dt.strftime('%Y-%m-%d')], 'average': [mean]})
                # df = df.append({'date': dt.strftime('%Y-%m-%d'), 'average': mean}, ignore_index=True)
                df = pd.concat([df, df_temp])
                df = df.sort_values(by=['date'])
        return df


if __name__ == '__main__':
    # barns = [2, 3, 5, 6, 7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 29, 30, 32, 34, 35]
    barns = [1, 8, 10, 12, 15, 16, 20, 26, 27, 28, 31, 33]
    # barns = [1]
    for day in range(10, 11):

        IMGs = []
        LABLEs = []
        batch_manager = BatchManager(barns, None, 'wheat', '2015-09-01', '2017-9-20', day,
                                     None)
        for barn in barns:
            cur_barn, images, features, lables, days, months = batch_manager.next_train_batch(1000)
            IMGs.extend(images)
            LABLEs.extend(lables)
        z, y, x = 1, 5, 5
        Y = np.array(LABLEs)[:, z, y, x]
        print(np.array(IMGs)[:, :, :, 0].shape)
        dir_path = 'z{}y{}x{}'.format(z, y, x)
        for layer in range(4):
            LAYER = np.array(IMGs)[:, :, :, layer].reshape((-1, 36)).transpose([1, 0])
            print(LAYER.shape)
            print(list(map(lambda x: pearsonr(x, Y), LAYER)))
            corr = np.array(list(map(lambda x: pearsonr(x, Y)[0], LAYER)))
            corr = corr.reshape((6, 6))
            print(corr)
            plt.figure(figsize=(14, 12))
            plt.title('Pearson Correlation of Features', y=1.05, size=15)
            colormap = plt.cm.RdBu
            sns.heatmap(corr, linewidths=0.1, vmax=1.0,
                        square=True, cmap=colormap, linecolor='white', annot=True)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            plt.savefig('layer{}.png'.format(layer))
            plt.close()
