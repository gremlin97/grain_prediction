import pandas as pd
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class LearnPredic:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.data = pd.DataFrame()
        self.month_map = [3, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
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
            if self.month_map[m] == 1 or self.month_map[m] == 2:
                self.wheat_weights[m] *= 2
                self.rice_weights[m] *= 2

    def load_data(self, dir_name, barns, test_barn):
        for barn_index in barns:
            tmp_df = pd.read_csv(dir_name + str(barn_index) + '.csv',
                                 index_col='date')
            self.data = self.data.append(tmp_df)
        self.data.dropna(inplace=True)
        self.data.index = self.data.index.astype('datetime64[ns]')
        # print(self.data)
        self.train_x = self.data.drop(labels=['grain_average', 'air_average'], axis=1)
        self.train_y = self.data['grain_average']
        # 加载测试集
        test_data = pd.read_csv(dir_name + str(test_barn) + '.csv',
                                index_col='date')
        test_data = test_data.dropna()
        test_data.index = test_data.index.astype('datetime64[ns]')
        self.test_y = test_data['grain_average']
        self.test_months = list(map(lambda date: int(date.strftime('%m'))-1, test_data.index))
        self.test_months = np.array(self.test_months)
        self.test_x = test_data.drop(labels=['grain_average', 'air_average'], axis=1)
        self.test_barn = test_barn


    def split_data(self, test_size):
        # Y = self.data['grain_average']
        # X = self.data.drop(labels=['grain_average', 'air_average'], axis=1)
        # self.train_x, self.test_x, self.train_y, self.test_y \
        #     = train_test_split(X, Y, test_size=test_size)
        pass

    def get_score(self, pred, grain_type='rice'):
        u, v = .0, .0
        pred = np.array(pred, dtype=np.float32)
        y = np.array(self.test_y, dtype=np.float32)
        y_mean = pred.mean()
        for m in range(0, 12):
            month_mask = self.test_months == m
            y_true = y[month_mask]
            if y_true.shape[0] == 0:
                continue
            y_pre = pred[month_mask]
            if grain_type == 'rice':
                w = float(self.rice_weights[m])
            else:
                w = float(self.wheat_weights[m])
            u += ((y_true - y_pre) ** 2).sum() * w / y_true.shape[0]
            v += ((y_true - y_mean) ** 2).sum() * w / y_true.shape[0]
        ac = 1 - u/v
        return ac

    def learn(self):
        scaler = pp.StandardScaler().fit(self.train_x)
        test_x = scaler.transform(self.test_x)
        train_x = scaler.transform(self.train_x)
        clf_1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                          max_depth=4, random_state=0, loss='lad')
        clf_1.fit(self.train_x, self.train_y)
        clf_2 = svm.SVR(kernel='linear')
        clf_2.fit(train_x, self.train_y)

        pre1 = clf_1.predict(self.test_x)
        pre2 = clf_2.predict(test_x)
        # print('GBR Score:', clf_1.score(self.test_x, self.test_y))
        # print('DTR Score:', clf_2.score(test_x, self.test_y))



        plt.figure(figsize=(10, 5))
        plt.scatter(self.test_x.index, self.test_y, s=20, edgecolor="black",
                    c="darkorange", label="Data")
        plt.plot(self.test_x.index, pre1, color="cornflowerblue",
                 label="GradientBoosting", linewidth=1)
        plt.plot(self.test_x.index, pre2, color="yellowgreen", label="SVR", linewidth=1)
        print(self.test_barn, self.get_score(pre1, 'rice'), self.get_score(pre2, 'rice'), sep=',')
        plt.xlabel("date")
        plt.ylabel("target")
        plt.title("Regressors")
        plt.legend()
        # plt.show()


if __name__ == '__main__':
    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]

    grain_barns = [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]
    for barn in grain_barns:
        # barn = 9
        bs = grain_barns.copy()
        bs.remove(barn)
        lp = LearnPredic()
        lp.load_data('data/features/feature_rice_layer1_7days/', bs, barn)
        lp.learn()
        # plt.savefig('figures/barn_'+str(barn)+'_layer_1.png')
        plt.close('all')

