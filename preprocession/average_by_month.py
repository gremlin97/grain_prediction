import pandas as pd

if __name__ == '__main__':
    df1 = pd.read_csv('../data/rice_by_layer/rice_layer_1.csv')
    df2 = pd.read_csv('../data/rice_by_layer/rice_layer_2.csv')
    df3 = pd.read_csv('../data/rice_by_layer/rice_layer_3.csv')
    df4 = pd.read_csv('../data/rice_by_layer/rice_layer_4.csv')
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df.loc[:, '粮情时间'] = pd.to_datetime(df['粮情时间'])
    aves = []
    for month in range(1, 13):
        temp = df[[int(d.strftime('%m')) == month for d in df['粮情时间']]]
        ave = temp['平均温度'].mean()
        aves.append(ave)
    sum_ave = sum(aves)
    for ave in aves:
        print(str(ave / sum_ave) + ',')
