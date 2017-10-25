import pandas as pd
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor


class LearnPredic:
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
        print('GBR Score:', clf_1.score(self.test_x, self.test_y))
        print('DTR Score:', clf_2.score(test_x, self.test_y))
        # for feature, important in zip(self.test_x.columns, clf_1.feature_importances_):
        #     print(feature, important)
        # for feature, important in zip(self.test_x.columns, clf_2.feature_importances_):
        #     print(feature, important)

        plt.figure(figsize=(10, 5))
        plt.scatter(self.test_x.index, self.test_y, s=20, edgecolor="black",
                    c="darkorange", label="Data")
        plt.plot(self.test_x.index, pre1, color="cornflowerblue",
                 label="GradientBoosting", linewidth=1)
        plt.plot(self.test_x.index, pre2, color="yellowgreen", label="DecisionTree", linewidth=1)
        plt.xlabel("date")
        plt.ylabel("target")
        plt.title("Gradient Boosting Regressor")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]

    grain_barns = [1,8,10,12,15,16,20,26,27,28,31,33]
    barns = [10]
    for barn in grain_barns:
        bs = grain_barns.copy()
        bs.remove(barn)
        lp = LearnPredic()
        lp.load_data('data/feature_wheat_layer2_6days/', bs, barn)
        # lp.load_data('data/feature_rice_layer4_10days/'
        #              , [2, 5, 6, 7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 29, 30, 32, 34, 35]
        #              , 3)

        # lp.split_data(0.3)
        lp.learn()
        # plt.savefig('figures/barn_'+str(barn)+'_layer_1.png')
        plt.close('all')

