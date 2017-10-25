import pandas as pd

if __name__ == '__main__':
    # 用粮情表和关联仓库
    situation = pd.read_excel('../data/HongZe_situation.xls')[['ID编号', '仓库ID', '时间']]
    time_list = []
    for t in situation['时间']:
        t = str(t)[0:8]
        time_list.append(t)
    situation['时间'] = pd.to_datetime(time_list, format='%Y%m%d')
    print(situation)
    barn_info = pd.read_excel('../data/HongZe_barn_info.xlsx')[['关联粮情仓库ID', '仓库名称']]
    situation = pd.merge(left=situation, right=barn_info, how='left',
                         left_on=('仓库ID',), right_on=('关联粮情仓库ID',))[['ID编号', '时间', '仓库名称']]
    situation.rename(columns={'ID编号': '关联粮情ID'}, inplace=True)

    # 读取温度层数据
    xls = pd.ExcelFile('../data/HongZe_layer.xls')
    layer_df = None
    for sheet in range(11):
        df_sheet = pd.read_excel(xls, sheetname=sheet)
        # 去除重复数据
        df_sheet.drop_duplicates(
            subset=('层面', '平均值', '关联粮情ID'),
            inplace=True)
        # 去除明显错误数据
        df_sheet = df_sheet[df_sheet['平均值'] != .0]
        df_sheet = df_sheet[df_sheet['平均值'] < 60]
        df_sheet = df_sheet[df_sheet['平均值'] > -20]
        print(df_sheet)

        if layer_df is None:
            layer_df = df_sheet
        else:
            layer_df = pd.concat([layer_df, df_sheet])
        print('sheet', sheet, 'done...')
    layer_df.drop_duplicates(
            subset=('层面', '平均值', '关联粮情ID'),
            inplace=True)
    layer_df = pd.merge(left=layer_df, right=situation, how='left',
                        on=['关联粮情ID'])
    layer_df.sort_values(by=['仓库名称', '时间', '层面'], inplace=True)
    layer_df.to_csv('../DL_data/layer.csv', index=False)
    print(layer_df)
