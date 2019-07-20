#Import modules
print('Importing needed modules...')
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras import regularizers
from keras import optimizers
from keras import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics

#Import train and test datasets
print('Importing data...')
df=pd.read_csv('train.csv')

df_train, df_test = train_test_split(df, test_size= 0.25, random_state= 1984)

#Check if classes are evenly distributed
df_train.Approved.sum()/len(df_train.Approved) #0.01461
df_test.Approved.sum()/len(df_test.Approved) #0.01468

#Merge datasets for pre-processing
train_test_set = pd.concat([df_train, df_test], axis = 0, ignore_index= True)

#-------------------------Feature Engineering----------------------------------
#Calculate Age at the time of lead creation
print('Making new features and deleting others...')
train_test_set.DOB = pd.to_datetime(train_test_set.DOB)
train_test_set.Lead_Creation_Date = pd.to_datetime(train_test_set.Lead_Creation_Date)

#Dates before 1970 are not correctly regonized. The lines below fix this issue
ind=[]
dates=[]
for i,y in enumerate(train_test_set.DOB):
    if y.year > 2000:
        ind.append(i)
        dates.append(y)

dates= pd.to_datetime(dates)
dates = pd.to_datetime([y.replace(year = y.year-100) for y in dates])
train_test_set.DOB.iloc[ind]= dates

train_test_set['Age'] = (train_test_set.Lead_Creation_Date - train_test_set.DOB).astype('timedelta64[Y]')
train_test_set.Age = train_test_set['Age'].dropna().apply(np.int64)

#Calculate how much a customer will pay as fraction of monthly income
train_test_set['EMI2Income'] = (train_test_set.EMI + train_test_set.Existing_EMI) / train_test_set.Monthly_Income

#Delete unneeded variables
train_test_set = train_test_set.drop(['ID','Employer_Code', 'DOB','Lead_Creation_Date', 'Contacted'], axis = 1)

#Set correct feature types
print('Assigning correct data types...')
train_test_set.Gender = train_test_set.Gender.astype('category')
train_test_set.City_Code = train_test_set.City_Code.astype('category')
train_test_set.City_Category = train_test_set.City_Category.astype('category')
train_test_set.Employer_Category1 = train_test_set.Employer_Category1.astype('category')
train_test_set.Employer_Category2 = train_test_set.Employer_Category2.astype('category')
train_test_set.Customer_Existing_Primary_Bank_Code = train_test_set.Customer_Existing_Primary_Bank_Code.astype('category')
train_test_set.Primary_Bank_Type = train_test_set.Primary_Bank_Type.astype('category')
train_test_set.Source = train_test_set.Source.astype('category')
train_test_set.Source_Category = train_test_set.Source_Category.astype('category')
train_test_set.Var1 = train_test_set.Var1.astype('category')
train_test_set.Approved = train_test_set.Approved.astype('category')

#Encode categorical variables to numeric values
print('Converting categorical variables to numeric...')
var_numeric = train_test_set.select_dtypes(include=['number']).copy()
var_non_numeric = train_test_set.select_dtypes(exclude=['number']).copy()

col_names = list(var_non_numeric)

for col in col_names:
    var_non_numeric[col] = var_non_numeric[col].cat.codes

train_test_set= pd.concat([var_numeric,var_non_numeric], axis = 1)#Import train and test datasets

#Check for number of nulls in each variable and lets see if we can impute them
null_values = train_test_set.isnull().sum()

#If a customer does not have bank account we get NaNs. Change them to 'None' category (N) in Bank Type and also change
#Nans to 'B000' category in Customer_Existing_Primary_Bank_Code
#train_test_set.Primary_Bank_Type[train_test_set.Primary_Bank_Type.isnull()]='N'
#train_test_set.Customer_Existing_Primary_Bank_Code[train_test_set.Customer_Existing_Primary_Bank_Code.isnull()]='B000'

#The variable 'Existing_EMI' is null in the case the customer has no previous bank account. Set as 0.0
train_test_set.Existing_EMI[train_test_set.Existing_EMI.isnull()]=0

#Check rows where variable 'Age' is null. All these rows contain zero information. Delete them
#train_test_set.loc[train_test_set.Age.isnull()].head()
#train_test_set = train_test_set[train_test_set.Age.notnull()]
#train_test_set = train_test_set.reset_index(drop=True)

#Lets let python fill in the rest of the missing NAs in order to proceed
train_test_set = train_test_set.fillna(method = 'pad')
train_test_set = train_test_set.dropna(axis = 0)

#Split again in train and test sets
train = train_test_set.iloc[:df_train.shape[0],:]
test = train_test_set.iloc[df_train.shape[0]:,:]

#Make a new train set that is more balanced
train_all_ones = train[train.Approved == 1]
train_mostly_zeros = train.iloc[:train_all_ones.shape[0],:]

train_final = pd.concat([train_all_ones,train_mostly_zeros], axis= 0)
train_final = train_final.reset_index(drop=True)
train_final = train_final.sample(frac=1)

train_y = train_final.Approved
test_y = test.Approved

train_x = train_final.drop(['Approved'], axis = 1)
test_x = test.drop(['Approved'], axis = 1)

train_x = train_x.as_matrix()
test_x = test_x.as_matrix()

train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)

print('Building Neural Network model...')
#class_weight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(Dense(8, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(12,
                kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(4,
                kernel_regularizer=regularizers.l2(0.01),
                activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, validation_split=0.2, epochs=6, batch_size=32)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

predictions = model.predict(test_x)

predictions[predictions>0.5] = 1
predictions[predictions<=0.5] = 0

confusion_matrix(test_y, predictions)
roc_auc_score(test_y, predictions)



