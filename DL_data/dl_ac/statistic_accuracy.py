import os
import numpy as np

for dir_m in os.listdir('./'):
    if os.path.isfile(dir_m):
        continue
    writer = open(dir_m + '.csv', 'w')
    for dir_b in os.listdir(dir_m):
        ac_list = []
        for file in os.listdir(os.path.join(dir_m, dir_b)):
            f_path = os.path.join(dir_m, dir_b, file)
            with open(f_path) as f:
                mac = 0
                for line in f:
                    ac = float(line.strip().split(',')[1])
                    mac = ac if ac > mac else mac
                ac_list.append(mac)
        mean_ac = np.mean(ac_list)
        writer.write(dir_b + ',' + str(mean_ac) + '\n')
    writer.close()

