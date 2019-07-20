import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('GBM_11.csv')
df2 = pd.read_csv('cat_1.csv')

df_all = df1.copy()

df_all['amount_spent_per_room_night_scaled'] = 0.5*df1['amount_spent_per_room_night_scaled'] + \
                                               0.5*df2['amount_spent_per_room_night_scaled']

df_all.to_csv('ensemble_1.csv', index = False)