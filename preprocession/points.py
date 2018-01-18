import threading

import pandas as pd
import numpy as np
import os


class ExtractPoints:
    def __init__(self):
        self.points_data = None

    def load_data(self):
        # 读取粮层数据，按编号去重
        print('读取粮层数据...')
        info_df = pd.read_csv('../DL_data/layer.csv')
        info_df.drop_duplicates(subset=['关联粮情ID'], inplace=True)
        info_df = info_df[['关联粮情ID', '时间', '仓库名称']]
        # 读取测温点数据
        print('读取测温点数据上半部分...')
        xls1 = pd.ExcelFile('../data/HongZe_points1.xlsx')
        self.load_points(xls1, info_df)
        xls1.close()
        print('读取测温点数据下半部分...')
        xls2 = pd.ExcelFile('../data/HongZe_points2.xlsx')
        self.load_points(xls2, info_df)
        xls2.close()

    def load_points(self, xls, info_df):
        for sheet in xls.sheet_names:
            print('sheet', sheet, 'processing...')
            df_sheet = pd.read_excel(xls, sheetname=sheet)
            # 与仓库数据做连接，顺便删除无效数据
            df_sheet = pd.merge(left=df_sheet, right=info_df, how='inner', on='关联粮情ID')
            if self.points_data is None:
                self.points_data = df_sheet
            else:
                self.points_data = pd.concat([self.points_data, df_sheet])
        self.points_data.sort_values(by=['仓库名称'], inplace=True)

    def extract(self):
        # 所有粮情ID
        sids = self.points_data['关联粮情ID'].drop_duplicates()
        n_sid = sids.shape[0]
        n = 1
        threads = []
        batch_size = 20
        for sid in sids:
            print(n)
            n += 1
            if n % batch_size == 0:
                for i in threads:
                    i.join()
                    print(n_sid, 'left')
                    n_sid -= 1
                threads = []
            t = threading.Thread(target=self.func, args=(sid,))
            t.setDaemon(True)
            t.start()
            threads.append(t)
        print(str(len(threads)) + 'leaves')
        for i in threads:
            i.join()
            print(n_sid)
            n_sid -= 1

    def func(self, sid):
        # 获取粮情ID对应点位
        points_df = self.points_data[self.points_data['关联粮情ID'] == sid]
        # 获取粮仓和时间信息
        barn = points_df['仓库名称'].iloc[0]
        date = points_df['时间'].iloc[0]
        # 若文件存在则跳过
        path = '../DL_data/points_temperature/' + str(barn) + '/'
        filename = path + str(date) + '.npy'
        if os.path.exists(filename):
            print(filename, 'already exists')
            return

        # 存储点位
        points_df = points_df.sort_values(by=['Z点', 'Y点', 'X点'], ascending=True)
        xlist, ylist, zlist = list(points_df['X点'].drop_duplicates()), \
                              list(points_df['Y点'].drop_duplicates()), \
                              list(points_df['Z点'].drop_duplicates())
        nx, ny, nz = len(xlist), len(ylist), len(zlist)
        xdic, ydic, zdic = {}, {}, {}
        for i in range(nx):
            xdic[xlist[i]] = i
        for i in range(ny):
            ydic[ylist[i]] = i
        for i in range(nz):
            zdic[zlist[i]] = i
        # print(points_df[['X点','Y点','Z点','点位温度','关联粮情ID']])
        arr = np.ones((nz, ny, nx), dtype=np.float32) * 66  # 66表示无温度点
        for _, row in points_df.iterrows():
            arr[zdic[row['Z点']], ydic[row['Y点']], xdic[row['X点']]] = np.round(row['数值'], 2)
        # 补足缺失测温点
        arr_temp = arr[:, :, -2:]
        for i in arr_temp:
            for j in i:
                if j[1] == 66:
                    j[1] = j[0]
        # 补足测温为0数据
        arr[np.where(arr == 0)] = arr[np.where(arr != 0)].mean()

        # print(points_df['仓库名称'].iloc[0], points_df['关联粮情ID'].iloc[0])
        # 调整测温点数量
        if nx == 8:
            arr[:, :, 2] = (arr[:, :, 1] + arr[:, :, 2]) / 2
            arr[:, :, 1] = arr[:, :, 0]
            arr[:, :, -3] = (arr[:, :, -2] + arr[:, :, -3]) / 2
            arr[:, :, -2] = arr[:, :, -1]
            arr = arr[:, :, 1:-1]

        if nx == 7:
            arr[:, :, 2] = (arr[:, :, 1] + arr[:, :, 2]) / 2
            arr[:, :, 1] = arr[:, :, 0]
            arr = arr[:, :, 1:]

        # 保存数据
        if not os.path.exists(path):
            print(path, 'did not exist')
            os.mkdir(path)
        # np.save(filename, arr)
        print(filename, 'saved')
        # del self.points_data[self.points_data['关联粮情ID'] == sid]


if __name__ == '__main__':
    ep = ExtractPoints()
    ep.load_data()
    ep.extract()
    # print(ep.points_data)
