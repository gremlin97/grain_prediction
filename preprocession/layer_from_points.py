import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    """温度点求平均得到层温数据"""


    # wheat [1,8,10,12,15,16,20,26,27,28,31,33]
    # rice [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]
    barns = [2,3,5,6,7,9,11,13,14,17,18,19,21,22,23,24,29,30,32,34,35]
    barn_list = []  # 仓库信息
    aver_list = []  # 平均温度
    dt_list = []  # 粮情时间
    for barn_name in barns:
        barn_path = '../DL_data/points_temperature/' + str(barn_name) + '仓/'
        barn_dir = os.listdir(barn_path)
        for file_name in barn_dir:
            file_path = barn_path + file_name
            arr = np.load(file_path)
            # 求得层平均
            # layers = arr.mean(axis=2).mean(axis=1)
            layers = arr[:, 1, 1] # 四个层的某个点
            # 粮情时间
            dt = file_name.split('.')[0]
            # 平均温度
            aver_temp = layers[0]  # 第4层
            barn_list.append(str(barn_name)+'仓')
            aver_list.append(aver_temp)
            dt_list.append(dt)
            print(dt)
    df = pd.DataFrame({
        '仓库信息': barn_list,
        '平均温度': aver_list,
        '粮情时间': dt_list
    }).sort_values(by=['仓库信息', '粮情时间'])
    df.to_csv('../data/rice_by_point/rice_point_4.csv', index=None)
