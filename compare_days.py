import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


class CompareDays:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.data = pd.DataFrame()

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
        self.test_x = test_data.drop(labels=['grain_average', 'air_average'], axis=1)

    def split_data(self, test_size):
        # Y = self.data['grain_average']
        # X = self.data.drop(labels=['grain_average', 'air_average'], axis=1)
        # self.train_x, self.test_x, self.train_y, self.test_y \
        #     = train_test_split(X, Y, test_size=test_size)
        pass

    def learn(self):
        clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=4, random_state=0, loss='lad', max_features=5)
        # clf = svm.SVR(C=1.0, epsilon=0.2, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        clf.fit(self.train_x, self.train_y)
        # for feature, important in zip(self.test_x.columns, clf.feature_importances_):
        #     print(feature, important)
        return clf.score(self.test_x, self.test_y)

    def trans_to_diff(self, days_back):
        """转换为温差"""
        for day in range(days_back, days_back + 5):
            self.train_x[str(day) + 'days_diff'] = self.train_x[str(day) + '_days_air'] \
                                                   - self.train_x[str(day) + '_days_grain']
            self.test_x[str(day) + 'days_diff'] = self.test_x[str(day) + '_days_air'] \
                                                  - self.test_x[str(day) + '_days_grain']

    def drop_temp(self, days_back):
        for day in range(days_back, days_back + 5):
            del self.train_x[str(day) + '_days_air'], self.train_x[str(day) + '_days_grain']
            del self.test_x[str(day) + '_days_air'], self.test_x[str(day) + '_days_grain']


if __name__ == '__main__':

    grain_barns = [1, 8, 10, 12, 15, 16, 20, 26, 27, 28, 31, 33]
    barns = [1, 10, 16, 26, 31]
    for barn in barns:
        bs = grain_barns.copy()
        bs.remove(barn)
        layers_score = list()
        mix_score = list()
        temp_diff_score = list()
        for back_to in range(5, 11):
            cd = CompareDays()
            # cd.load_data('data/feature_wheat_layer3_' + str(back_to) + 'days/',
            #              [8, 10, 12, 15, 16, 20, 26, 27, 28, 31, 33], 1)
            cd.load_data('data/feature_wheat_layer4_' + str(back_to) + 'days/'
                         , bs
                         , barn)
            print(back_to)
            print('temp only')
            layers_score.append(cd.learn())
            print('mix')
            cd.trans_to_diff(back_to)
            mix_score.append(cd.learn())
            print('diff only')
            cd.drop_temp(back_to)
            temp_diff_score.append(cd.learn())
        plt.figure(figsize=(10, 5))
        plt.title('Wheat Barn ' + str(barn) + ' Layer 4')
        plt.xlabel("Future days")
        plt.ylabel("Score")
        plt.plot(range(5, 11), layers_score, label="Temperature only")
        plt.plot(range(5, 11), mix_score, label="Temperature and diff")
        plt.plot(range(5, 11), temp_diff_score, label="Diff only")
        plt.legend()
        plt.savefig('figures/wheat_barn_' + str(barn) + '_layer_4.png')
        # plt.show()
        plt.close('all')
