import os
import numpy as np

for dir_m in os.listdir('./'):
    if os.path.isfile(dir_m):
        continue
    writer = open(dir_m + '.csv', 'w')
    whole = []
    for file_b in os.listdir(dir_m):
        ac_list = []
        f_path = os.path.join(dir_m, file_b)
        with open(f_path) as f:
            mac = 0
            b_list = []
            for line in f:
                ac = float(line.strip().split(',')[1])
                ac_list.append(ac)
                whole.append(ac)

        mean_ac = np.mean(ac_list)
        writer.write(file_b[:-3] + ',' + str(mean_ac) + '\n')
    writer.write('var,'+str(np.nanvar(whole)))
    writer.close()

