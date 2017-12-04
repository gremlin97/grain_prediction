import numpy as np
import os
import math
if __name__ == '__main__':
    dir_path = '../DL_data/points_temperature'
    out_path = '../DL_data/points_temperature'
    for barn in range(40):
        barn_path = str(barn) + '仓'
        if not os.path.exists(os.path.join(dir_path, barn_path)):
            print(barn_path, 'is not exists')
            continue
        for file in os.listdir(os.path.join(dir_path, barn_path)):
            arr = np.load(os.path.join(dir_path, barn_path, file))
            valid = []
            invalid = []
            for z in range(4):
                for y in range(6):
                    for x in range(6):
                        temp = arr[z,y,x]
                        if temp > 45:
                            invalid.append(temp)
                            print('.....')
                        else:
                            valid.append(temp)
        #     if len(valid) < 72:
        #         print(barn)
        #         continue
        #     mean_temp = np.array(valid).mean()
        #     for z in range(4):
        #         for y in range(6):
        #             for x in range(6):
        #                 if arr[z,y,x] > 45:
        #                     arr[z,y,x] = mean_temp
        #     path = os.path.join(out_path, barn_path)
        #     if not os.path.exists(path):
        #         os.mkdir(path)
        #     path = os.path.join(path, file)
        #     np.save(path, arr)
        # print(barn)