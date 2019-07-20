import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from statsmodels.tsa.ar_model import AR


demographics = pd.read_csv('demographics.csv')
event_calendar = pd.read_csv('event_calendar.csv')
historical_volume = pd.read_csv('historical_volume.csv')
industry_volume = pd.read_csv('industry_volume.csv')
price_sales_promotion = pd.read_csv('price_sales_promotion.csv')
weather = pd.read_csv('weather.csv')
df_test = pd.read_csv('volume_forecast_test.csv')

#1. For weather dataset
weather['YearMonth'] = weather.YearMonth.astype(str)
weather['Year'] = weather.YearMonth.str[0:4]
weather['Month'] = weather.YearMonth.str[4:6]
weather = weather.drop(['YearMonth'], axis = 1)
weather['Month'] = weather.Month.astype(str)
weather_jan = weather[weather.Month == '01']
Avg_Temps = weather_jan.groupby(['Agency'], as_index=False)['Avg_Max_Temp'].mean()

df_test = pd.merge(df_test, Avg_Temps,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#2. For Average population and House income
df_test = pd.merge(df_test, demographics,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#3. For Industry Volume
industry_volume['YearMonth'] = industry_volume.YearMonth.astype(str)
industry_volume['Year'] = industry_volume.YearMonth.str[0:4]
industry_volume['Month'] = industry_volume.YearMonth.str[4:6]
industry_volume = industry_volume.drop(['YearMonth'], axis = 1)
industry_volume['Month'] = industry_volume.Month.astype(str)
industry_volume = industry_volume[industry_volume.Month == '01']
industry_volume['Year'] = industry_volume.Year.astype(int)

df_test['Industry_Volume'] = 600000000

#4. For Price, Sales and Promotions
price_sales_promotion['YearMonth'] = price_sales_promotion.YearMonth.astype(str)
price_sales_promotion['Year'] = price_sales_promotion.YearMonth.str[0:4]
price_sales_promotion['Month'] = price_sales_promotion.YearMonth.str[4:6]
price_sales_promotion = price_sales_promotion.drop(['YearMonth'], axis = 1)
price_sales_promotion['Month'] = price_sales_promotion.Month.astype(str)
price_sales_promotion_Jan = price_sales_promotion[price_sales_promotion.Month == '01']
Max_Price = price_sales_promotion_Jan.groupby(['Agency', 'SKU'], as_index=False)['Price'].max()
Avg_Sales = price_sales_promotion_Jan.groupby(['Agency', 'SKU'], as_index=False)['Sales'].mean()
Avg_Promotions = price_sales_promotion_Jan.groupby(['Agency', 'SKU'], as_index=False)['Promotions'].mean()

df_test = pd.merge(df_test, Max_Price,  how='left',
                  left_on=['Agency', 'SKU'], right_on = ['Agency', 'SKU'])

df_test = pd.merge(df_test, Avg_Sales,  how='left',
                  left_on=['Agency', 'SKU'], right_on = ['Agency', 'SKU'])

df_test = pd.merge(df_test, Avg_Promotions,  how='left',
                  left_on=['Agency', 'SKU'], right_on = ['Agency', 'SKU'])

#5. For Events
event_calendar['YearMonth'] = event_calendar.YearMonth.astype(str)
event_calendar['Year'] = event_calendar.YearMonth.str[0:4]
event_calendar['Month'] = event_calendar.YearMonth.str[4:6]
event_calendar = event_calendar.drop(['YearMonth'], axis = 1)
event_calendar['Month'] = event_calendar.Month.astype(str)
event_calendar = event_calendar[event_calendar.Month == '01']

df_test['Easter Day'] = np.zeros(df_test.shape[0])
df_test['Good Friday'] = np.zeros(df_test.shape[0])
df_test['New Year'] = np.ones(df_test.shape[0])
df_test['Christmas'] = np.zeros(df_test.shape[0])
df_test['Labor Day'] = np.zeros(df_test.shape[0])
df_test['Independence Day'] = np.zeros(df_test.shape[0])
df_test['Revolution Day Memorial'] = np.zeros(df_test.shape[0])
df_test['Regional Games'] = np.zeros(df_test.shape[0])
df_test['FIFA U-17 World Cup'] = np.zeros(df_test.shape[0])
df_test['Football Gold Cup'] = np.zeros(df_test.shape[0])
df_test['Beer Capital'] = np.zeros(df_test.shape[0])
df_test['Music Fest'] = np.zeros(df_test.shape[0])

#Add the Month and Year Variables
df_test['Month'] = '01'
df_test['Month'] = df_test.Month.astype('category')
df_test['Year'] = int(2018)

#Set correct feature types
df_test['Agency'] = df_test.Agency.astype('category')
df_test['SKU'] = df_test.SKU.astype('category')

#Drop target variable (Volume)
df_test = df_test.drop(['Volume'], axis = 1)




