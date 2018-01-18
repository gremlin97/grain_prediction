import numpy as np
import os
import math
if __name__ == '__main__':
    dir_path = '../DL_data/points_temperature'
    out_path = '../DL_data/points_temperature'
    # 遍历每个仓
    for barn in range(40):
        barn_path = str(barn) + '仓'
        if not os.path.exists(os.path.join(dir_path, barn_path)):
            print(barn_path, 'is not exists')
            continue
        # 对于每个仓，读取仓下所有.npy文件名
        for file in os.listdir(os.path.join(dir_path, barn_path)):
            # 加载某仓某天温度点，格式为[4,6,6]，分别对应z，y，x轴
            arr = np.load(os.path.join(dir_path, barn_path, file))
            valid = []
            invalid = []
            # 遍历每个点，分出不合理的点和合理的点
            for z in range(4):
                for y in range(6):
                    for x in range(6):
                        temp = arr[z,y,x]
                        if temp > 45:
                            invalid.append(temp)
                            print('.....')
                        else:
                            valid.append(temp)
            if len(valid) < 72:
                print(barn)
                continue
            # 将不合理的点用合理点的均值补充
            mean_temp = np.array(valid).mean()
            for z in range(4):
                for y in range(6):
                    for x in range(6):
                        if arr[z,y,x] > 45:
                            arr[z,y,x] = mean_temp
            path = os.path.join(out_path, barn_path)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, file)
            np.save(path, arr)
        print(barn)
