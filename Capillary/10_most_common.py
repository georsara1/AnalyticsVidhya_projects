import pandas as pd
import numpy as np

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('sample_submission.csv')


#Find 10 most common products for each customer
df_train60 = df_train.iloc[40000:,:]
most_common_10 = df_train60.groupby('UserId')['productid'].apply(lambda x: x.value_counts().head(10))
most_common_10 = most_common_10.reset_index()
del most_common_10['productid']
most_common_10.columns = ['UserId', 'product_list']

products = []
for idx, user in enumerate(df_test['UserId'].values):
    user_list = np.array(most_common_10['product_list'][most_common_10['UserId']==user])
    products.append(user_list)

products = [p.tolist() for p in products]

single_most_common = df_train['productid'][117000:].value_counts().head(4).reset_index()


for i in range(len(products)):
    if (len(products[i])<10) and (single_most_common['index'][0] not in products[i]):
        products[i].append(single_most_common['index'][0])

for i in range(len(products)):
    if (len(products[i])<10) and (single_most_common['index'][1] not in products[i]):
        products[i].append(single_most_common['index'][1])



df = pd.DataFrame({'UserId': df_test['UserId'].values,
                   'product_list': products})

df.to_csv('submission_common_10_plus2_117k_train40k.csv', index = False)