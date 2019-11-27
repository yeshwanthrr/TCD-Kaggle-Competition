# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:50:35 2019

@author: 
"""

import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

#Drop features
def dropFeatures(file):
    na_val= ['#N/A','nA','#NA','#NUM!','unknown']
    df = pd.read_csv(file,na_values=na_val)
    instance = df['Instance']
    df = df.drop(['Instance','Wears Glasses','Hair Color'],axis=1)
    return df,instance


#Fill missing values
def fillMissingValueCon(df,cons):
    simpleimputer=SimpleImputer(strategy='mean')
    for col in cons:
        df[col]=simpleimputer.fit_transform(df[col].values.reshape(-1,1))
    return df

#Fill categorical columns
def fillMissingValueCat(df,cats):
    for col in cats:
        df[col] = df[col].fillna('missing')
    return df

def renameColumns(df):
    df = df.rename(columns={'Total Yearly Income [EUR]':'Income'})
    df = df.rename(columns={'Work Experience in Current Job [years]':'Work Exp'})
    df = df.rename(columns={'Body Height [cm]':'Height'})
    df = df.rename(columns={'Yearly Income in addition to Salary (e.g. Rental Income)':'Additional Sal'})
    return df

    
    
#Read training and test dataset
df_tr,instance_tr = dropFeatures('Train.csv')

#Drop Duplicates
df_tr = df_tr.drop_duplicates() 

#Read test data and drop columns
df_ts,instance_ts = dropFeatures('Test.csv')


#Call rename function for training and test data
df_tr = renameColumns(df_tr)
df_ts = renameColumns(df_ts)

#Drop Income variable in test data as all the values are NA
df_ts = df_ts.iloc[:,:-1]


#Prune the data
'''
df_tr['Profession'] = df_tr['Profession'].str.replace(" ","").str[:5].str.strip()
df_ts['Profession'] = df_ts['Profession'].str.replace(" ","").str[:5].str.strip()
'''

df_tr['Additional Sal'] = df_tr['Additional Sal'].str.split(" ", n = 1, expand = True)[0]
df_tr['Additional Sal'] = df_tr['Additional Sal'].astype(float)
df_ts['Additional Sal'] = df_ts['Additional Sal'].str.split(" ", n = 1, expand = True)[0]
df_ts['Additional Sal'] = df_ts['Additional Sal'].astype(float)

#Call the functions to fill the missing values
cats_tr = [col for col in df_tr.columns if df_tr[col].dtypes == 'object']
cons_tr = list(df_tr.columns[~df_tr.columns.isin(cats_tr)])

cats_na_tr = [x for x in cats_tr if df_tr[x].isna().any()]
cons_na_tr = [x for x in cons_tr if df_tr[x].isna().any()]

df_tr = fillMissingValueCat(df_tr,cons_na_tr)
df_tr = fillMissingValueCat(df_tr,cats_na_tr)

#Do the same for test data
cats_ts = [col for col in df_ts.columns if df_ts[col].dtypes == 'object']
cons_ts = list(df_ts.columns[~df_ts.columns.isin(cats_ts)])

cats_na_ts = [x for x in cats_ts if df_ts[x].isna().any()]
cons_na_ts = [x for x in cons_ts if df_ts[x].isna().any()]

df_ts = fillMissingValueCat(df_ts,cons_na_tr)
df_ts = fillMissingValueCat(df_ts,cats_na_tr)


#Split the training data to x and y
trainX = df_tr.iloc[:,:-1]
trainy = df_tr.iloc[:,-1]

#Perform targetencoding
tgtE = TargetEncoder()
tgtE.fit(trainX,trainy)
trainX = tgtE.transform(trainX)

test_data = tgtE.transform(df_ts)


#Build model

Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainX, trainy, test_size=0.3, random_state=0)


import lightgbm as lgb
data_train = lgb.Dataset(Xtrain, label=Ytrain)
data_val = lgb.Dataset(Xtest, label=Ytest)
params = {}
params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    #'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    'verbose': 0
}


clf = lgb.train(params, data_train, 20000, valid_sets = [data_train, data_val], verbose_eval=1000, early_stopping_rounds=500)

#Predict the outcome on test data
data_pred = clf.predict(test_data)


#Expor the result to csv file
df_exp = pd.DataFrame()
#test = pd.read_csv('Test.csv')
#df_exp['Instance'] = test['Instance']
df_exp['Instance'] = instance_ts
df_exp['Total Yearly Income [EUR]'] = data_pred
df_exp.to_csv(r'output_.0519.csv',index=False) 


'''
### Flags 
visualize=False
unixOptions = "ho:v"
gnuOptions = ["help", "visualize"]
# read commandline arguments, first
fullCmdArguments = sys.argv

# - further arguments
argumentList = fullCmdArguments[1:]
try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

for currentArgument, currentValue in arguments:
    if currentArgument in ("-vis", "--visualize"):
        print ("enabling Data Visualization")
        visualize=True
'''
'''
if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(10, 20))

        lgb.plot_importance(clf, ax=ax, height=0.2, xlim=None, ylim=None, 
                        title='Feature importance', xlabel='Feature importance', ylabel='Features', )
        plt.savefig('Importance.png')
'''










