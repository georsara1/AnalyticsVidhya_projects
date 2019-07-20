import pandas as pd

demographics = pd.read_csv('demographics.csv')
event_calendar = pd.read_csv('event_calendar.csv')
historical_volume = pd.read_csv('historical_volume.csv')
industry_volume = pd.read_csv('industry_volume.csv')
price_sales_promotion = pd.read_csv('price_sales_promotion.csv')
weather = pd.read_csv('weather.csv')


#Join historical volume with respective weather temperatures
all_data = pd.merge(historical_volume, weather,  how='left',
                  left_on=['Agency','YearMonth'], right_on = ['Agency','YearMonth'])

#Join all data with demographics
all_data = pd.merge(all_data, demographics,  how='left',
                  left_on=['Agency'], right_on = ['Agency'])

#Join all data with industry volume
all_data = pd.merge(all_data, industry_volume,  how='left',
                  left_on=['YearMonth'], right_on = ['YearMonth'])

#Join all data volume with price sales promotion data
all_data = pd.merge(all_data, price_sales_promotion,  how='left',
                  left_on=['Agency','SKU','YearMonth'], right_on = ['Agency','SKU','YearMonth'])

#Join all data with event calendar
all_data = pd.merge(all_data, event_calendar,  how='left',
                  left_on=['YearMonth'], right_on = ['YearMonth'])

all_data.to_csv('all_data.csv', sep = ',',index = False)




