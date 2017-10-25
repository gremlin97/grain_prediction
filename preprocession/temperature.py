import pandas as pd
import numpy as np

if __name__ == '__main__':
    temp_df = pd.read_csv('data/江苏.csv')
    y_s = temp_df['year']
    m_s = temp_df['month']
    d_s = temp_df['day']
    for i in range(y_s.size):
        y_s.iloc[i] = str(y_s.iloc[i]) \
                      + (str(m_s.iloc[i]) if m_s.iloc[i] >= 10 else '0' + str(m_s.iloc[i])) \
                      + (str(d_s.iloc[i]) if d_s.iloc[i] >= 10 else '0' + str(d_s.iloc[i]))
        print(y_s.iloc[i])
    temp_df['year'] = pd.to_datetime(temp_df['year'], format='%Y%m%d')
    temp_df.rename(columns={'year': 'date'}, inplace=True)
    temp_df.set_index('date', inplace=True)
    temp_df = temp_df[temp_df['average'] < 50]
    temp_df = temp_df[['city', 'average', 'max', 'min', 'rain']]
    temp_df.to_csv('data/JiangSu.csv')
