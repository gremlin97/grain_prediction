import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读粮情数据
    data_df = pd.read_csv('data/wheat_layer1.csv', index_col='粮情时间')
    data_df = data_df[data_df['仓库信息'] == '10仓']
    data_df = data_df[data_df['层平均温度'] < 50]
    data_df = data_df[['层平均温度']]
    print(data_df)
    # 限制时间
    data_df.index = data_df.index.astype('datetime64[ns]')
    data_df = data_df[data_df.index >= pd.datetime(2015, 9, 16)]
    data_df = data_df[data_df.index < pd.datetime(2017, 8, 31)]

    # # 时间作为索引
    # data_df['粮情时间'] = pd.to_datetime(data_df['粮情时间'], format='%Y%m%d')
    # data_df.set_index('粮情时间', inplace=True)
    # 读温度数据
    temp_df = pd.read_csv('data/JiangSu.csv', index_col='date')
    # 只要淮安
    temp_df = temp_df[temp_df.city == '淮安']
    temp_df.index = temp_df.index.astype('datetime64[ns]')
    temp_df = temp_df[temp_df.index >= pd.datetime(2015, 9, 16)]
    temp_df = temp_df[temp_df.index <= pd.datetime(2017, 8, 31)]
    # 画图
    plt.figure()
    ax = data_df.层平均温度.plot(label='grain')
    ax.set_ylabel('temperature')
    ax.set_xlabel('date')
    ax.legend(loc=2)
    ax2 = temp_df.average.plot(style='g', label='air')
    ax2.legend(loc=2)
    ax2.set_xlabel('date')
    plt.show()
