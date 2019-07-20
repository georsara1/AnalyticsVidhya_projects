import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import regularizers
from sklearn import preprocessing

#Import dataset
train_df = pd.read_csv('Train.csv', sep = ',')
test_df = pd.read_csv('Test.csv', sep = ',')
submission_df = pd.read_csv('SampleSubmission.csv', sep = ',')

train_y = train_df['Item_Outlet_Sales'].values
train_df = train_df.drop(['Item_Outlet_Sales'], axis = 1)

#---------------------------Exploratory Data Analysis--------------------------
print('Doing some pre-processing to the data...')
train_test_set = pd.concat([train_df, test_df], axis = 0, ignore_index= True)

#Create new feature from Item Identifier
train_test_set['Item_Category'] = train_test_set.Item_Identifier.str[:2]

#Make corrections to Item Fat Content
train_test_set['Item_Fat_Content'] = train_test_set['Item_Fat_Content'].astype('category')
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='LF'] = 'Low Fat'
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='low fat'] = 'Low Fat'
train_test_set['Item_Fat_Content'][train_test_set['Item_Fat_Content']=='reg'] = 'Regular'

#Set as categorical
train_test_set['Item_Category'] = train_test_set['Item_Category'].astype('category')

train_test_set['Item_Type'] = train_test_set['Item_Type'].astype('category')

train_test_set['Outlet_Establishment_Year'] = train_test_set['Outlet_Establishment_Year'].astype('category')

train_test_set['Outlet_Identifier'] = train_test_set['Outlet_Identifier'].astype('category')

train_test_set['Outlet_Location_Type'] = train_test_set['Outlet_Location_Type'].astype('category')

train_test_set['Outlet_Type'] = train_test_set['Outlet_Type'].astype('category')

train_test_set['Outlet_Size'] = train_test_set['Outlet_Size'].astype('category')

train_test_set['Item_Identifier'] = train_test_set['Item_Identifier'].astype('category')

#Check for missing values and replace with median
train_test_set.isnull().sum() #Find NaNs and try to replace them manually

for i in range(train_test_set.shape[0]):
    if pd.isnull(train_test_set.Outlet_Size[i]) and train_test_set.Outlet_Type[i] == 'Grocery Store':
        train_test_set.Outlet_Size[i] = 'Small'

s = pd.Series(train_test_set.Outlet_Size[train_test_set.Outlet_Type == 'Supermarket Type1'])
s.value_counts() #Most Supermarket type1 are 'Small'

for i in range(train_test_set.shape[0]):
    if pd.isnull(train_test_set.Outlet_Size[i]) and train_test_set.Outlet_Type[i] == 'Supermarket Type1':
        train_test_set.Outlet_Size[i] = 'Small'


visibility_pivot = pd.pivot_table(train_test_set,index=["Item_Type"], values = ["Item_Visibility"])

for i in range(train_test_set.shape[0]):
    if pd.isnull(train_test_set.Item_Visibility[i]):
        train_test_set.Item_Visibility[i] = visibility_pivot.Item_Visibility[visibility_pivot.index==train_test_set.Item_Type[i]]


weight_pivot = pd.pivot_table(train_test_set,index=["Item_Type"], values = ["Item_Weight"])

for i in range(train_test_set.shape[0]):
    if pd.isnull(train_test_set.Item_Weight[i]):
        train_test_set.Item_Weight[i] = weight_pivot.Item_Weight[weight_pivot.index==train_test_set.Item_Type[i]]

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)

#Split again in train and test set
train_set = train_test_set[:8523].values
test_set = train_test_set.iloc[8523:].values

train_set = preprocessing.scale(train_set)
test_set = preprocessing.scale(test_set)

print('Building Neural Network model...')
model = Sequential()
model.add(Dense(128, input_dim=train_set.shape[1], kernel_initializer='normal',
                kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
#model.add(BatchNormalization())
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.02), activation="relu"))
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02), activation="relu"))
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.02), activation="relu"))
model.add(Dense(1))
model.add(Activation("linear"))

model.compile(loss="mean_squared_error", optimizer='adam')

history = model.fit(train_set, train_y, validation_split=0.2, epochs=16, batch_size=12)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predict and save to submit
print('Writing to csv file...')
predictions = model.predict(test_set)

submission_df['Item_Outlet_Sales'] = predictions

#submission_df.to_csv('myNNsubmission.csv', index = False)

print('Complete! Enjoy your submission!')