import pandas as pd
import numpy as np


class LoadAndFilter:
    def __init__(self):
        self.condition = None

    def load(self, filePath):
        """加载数据"""
        if self.condition is None:
            self.condition = pd.read_csv(filePath)
        else:
            self.condition = self.condition.append(pd.read_csv(filePath))

    def filter(self, grain_type, start_time, end_time, layer, max_temp, min_temp):
        result = self.condition[self.condition['品种'] == grain_type]
        # result = result[start_time <= result['粮情时间']]
        # result = result[result['粮情时间'] <= end_time]
        result = result[result['粮食层数'] == layer]
        # 去除非层平均温度为0的项
        result = result[result['层平均温度'] != .0]
        # 去除温度不符合区间的项
        result = result[result['层平均温度'] < max_temp]
        result = result[result['层平均温度'] > min_temp]
        time_series = result['粮情时间']
        time_arr = np.zeros(time_series.shape, dtype=np.dtype((str, 35)))
        for i in range(time_series.shape[0]):
            time_arr[i] = str(time_series.iloc[i])[0: 8]
            # time_series.iloc[i] = str(time_series.iloc[i])[0: 8]
        result['粮情时间'] = pd.to_datetime(pd.Series(time_arr, time_series.index), format='%Y%m%d')
        result.set_index('粮情时间', inplace=True)
        print(result)
        return result


if __name__ == '__main__':
    filter = LoadAndFilter()
    filter.load('data/condition_1406_1604.csv')
    filter.load('data/condition_1604_1703.csv')
    filter.load('data/condition_1703_1709.csv')

    df = filter.filter('粳稻', 20140600000000, 20170900000000, 4, 50, -15)
    print(df['仓库信息'].value_counts())
    df.to_csv('data/rice_by_layer/rice_layer4.csv')
