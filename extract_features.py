import pandas as pd
import numpy as np


def fill_data(dic):
    l = []
    for k, v in dic.items():
        if v is not np.nan:
            l.append(v)
    print(len(l), ' ', len(dic))
    if len(l) >= (len(dic) / 2.):
        mean = np.array(l).mean()
        for k, v in dic.items():
            if v is np.nan:
                # print(l, mean)
                dic[k] = round(mean, 2)
        return True, dic
    else:
        return False, None


class ExtractFeature:
    def __init__(self):
        self.data = None
        self.temp_data = None
        self.merged_data = pd.DataFrame()
        self.features = pd.DataFrame()

    def to_csv(self, filename):
        self.features.to_csv(filename)

    def extract(self, duration, days_back_to, date_from, date_to):
        features = list()  # 存放特征
        today = dict()
        before_air = dict()
        before_data = dict()
        # 筛选时间
        self.merged_data.index = self.merged_data.index.astype('datetime64[ns]')
        merged_data = self.merged_data[self.merged_data.index >= date_from]
        merged_data = merged_data[merged_data.index <= date_to]
        # 抽取特征
        for index, row in merged_data.iterrows():
            if pd.isnull(row['层平均温度']):
                # 如果当前行没有层平均温度信息则跳过
                # print(row['层平均温度'])
                continue
            today['date'] = index
            today['grain_average'] = row['层平均温度']
            today['air_average'] = row['average']
            # print(today)
            td = pd.to_timedelta(str(days_back_to) + ' days')
            grain_dates = pd.date_range(end=index - td, periods=duration - days_back_to)
            td = pd.to_timedelta('1 days')
            temp_dates = pd.date_range(end=index - td, periods=duration)
            day_back = duration - 1
            for d in grain_dates:
                print(d)
                # 加入粮食K天前温度
                if d in self.data.index:
                    # print(self.data.loc[d]['层平均温度'])
                    before_data[str(day_back) + '_days_grain'] = round(float(self.data.loc[d]['层平均温度']), 2)
                else:
                    before_data[str(day_back) + '_days_grain'] = np.nan
                    # print(d)
                day_back -= 1
            day_back = duration - 1
            for d in temp_dates:
                # 加入空气K天前温度
                if d in self.temp_data.index:
                    before_air[str(day_back) + '_days_air'] = float(self.temp_data.loc[d]['average'])
                else:
                    before_air[str(day_back) + '_days_air'] = np.nan
                # 天数+1
                day_back -= 1
            # 拟合空缺数据
            b1, d1 = fill_data(before_data)
            b2, d2 = fill_data(before_air)
            if b1 and b2:
                before_data = d1
                before_air = d2
                features.append(dict(today, **before_data, **before_air))
                # print(dict(today, **before_data, **before_air))
        self.features = pd.DataFrame(features).set_index('date')
        print(self.features.dropna())

    def merge_data_temp(self, data_file, temp_file, barn):
        # 读取数据并预处理
        self.data = self.handle_csv(data_file, '粮情时间')
        self.data = self.data[self.data['仓库信息'] == barn]
        # 同一天的求平均
        dates = pd.date_range(start='2014-05-05', end='2017-09-09')
        templist = list()
        for date in dates:
            if date in self.data.index:
                df = self.data.loc[date]
                if type(df) is pd.DataFrame:
                    tempdict = df.iloc[0].to_dict()
                    tempdict['粮情时间'] = date
                    tempdict['层平均温度'] = df['层平均温度'].mean()
                    # print(df_copy)
                    self.data.drop(axis=0, labels=[date], inplace=True)
                    templist.append(tempdict)
        df = pd.DataFrame(templist)
        df.set_index('粮情时间', inplace=True)
        self.data = self.data.append(df)
        self.data.index = self.data.index.astype('datetime64[ns]')
        # 按索引排序
        self.data.sort_index(inplace=True)
        self.temp_data = self.handle_csv(temp_file, 'date')
        self.temp_data = self.temp_data[self.temp_data['city'] == '淮安']
        self.temp_data.sort_index(inplace=True)
        # 处理数据
        self.merged_data = pd.merge(left=self.temp_data, right=self.data
                                    , how='outer', left_index=True, right_index=True)
        self.merged_data.drop_duplicates(inplace=True)
        # print(self.merged_data.loc['2016-01-02'])
        # print(self.merged_data[[not x and not y for x, y in zip(self.merged_data['average'].isnull(), self.merged_data['层平均温度'].isnull())]])
        # print(self.merged_data)

    def handle_csv(self, file_name, time_index):
        raw_data = pd.read_csv(file_name, index_col=time_index)
        raw_data.index = raw_data.index.astype('datetime64[ns]')
        return raw_data


if __name__ == '__main__':
    import os

    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]
    # barns = [1, 8, 10, 12, 15, 16, 20, 26, 27, 28, 31, 33]  # 所有白麦仓
    barns = [2, 3, 5, 6, 7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 24, 29, 30, 32, 34, 35]  # 所有粳稻仓
    for layer in range(1, 5):
        for back_to in range(5, 11):
            for barn in barns:
                ef = ExtractFeature()
                ef.merge_data_temp('data/rice_by_layer/rice_layer' + str(layer) + '.csv', 'data/JiangSu.csv',
                                   barn=str(barn) + '仓')
                ef.extract(back_to + 5, back_to, pd.datetime(2016, 1, 1), pd.datetime(2016, 12, 30))
                output_dir = 'data/feature_rice_layer' + str(layer) + '_' + str(back_to) + 'days/'
                output_file = output_dir + str(barn) + '.csv'
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                else:
                    print('dir is already exists:', output_dir)
                ef.to_csv(output_file)
