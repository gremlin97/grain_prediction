import os
import numpy as np

for dir_m in os.listdir('./'):
    if os.path.isfile(dir_m):
        continue
    writer = open(dir_m + '.csv', 'w')
    whole = []

    for dir_b in os.listdir(dir_m):
        ac_list = []
        # point_files = [open(os.path.join(dir_m, dir_b, file)) for file in os.listdir(os.path.join(dir_m, dir_b))]
        # iter = 0
        # max_ac = 0
        # max_iter = 0
        # while iter < 1195:
        #     temp = 0.
        #     for file in point_files:
        #         fields = file.readline().strip().split(',')
        #         print(dir_b, file.name, fields)
        #         iter = int(fields[0])
        #         temp += float(fields[1])
        #     temp /= len(point_files)
        #     max_ac, max_iter = (temp, iter) if temp > max_ac else (max_ac, max_iter)
        # writer.write(dir_b + ',' + str(max_ac) + '\n')

        for file in os.listdir(os.path.join(dir_m, dir_b)):
            f_path = os.path.join(dir_m, dir_b, file)

            with open(f_path) as f:
                mac = 0
                for line in f:
                    ac = float(line.strip().split(',')[1])
                    mac = ac if ac > mac else mac
                ac_list.append(mac)
                whole.append(mac)
        mean_ac = np.mean(ac_list)
        writer.write(dir_b + ',' + str(mean_ac) + '\n')
    writer.write('var,'+str(np.nanvar(whole)))
    writer.close()

