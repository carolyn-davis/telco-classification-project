#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:58:14 2021

@author: carolyndavis
"""

# =============================================================================
#                 PREPARE TO PROJECT 
# =============================================================================

import env

from env import host, user, password
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =============================================================================
# CONFIGS 
# =============================================================================
SEED = 6969     #common nomeclatures... global variables are all caps...

# =============================================================================
#                             ACQUIRE PHASE
# =============================================================================


#Taking a look at the data/features 


def get_db_url(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM contract_types, customers, internet_service_types, payment_types'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df


def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('telco_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_df.csv')
        
    return df

telco_df = get_telco_data()




# =============================================================================
#                             PREPARE/TIDY DATA PHASE
# =============================================================================

telco_df = telco_df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
#drop dupes bc of faulty SQL query

telco_df = telco_df.set_index('customer_id')   #sets index to unique categ/index 'customer id

telco_df.to_excel(r'/Users/carolyndavis/Codeup/classification-exercises\File telco_df.xlsx', index = False)
#^^^ Read it into a an excel file for prepare 

telco_df.head(10)


telco_df.columns

telco_df.info()


churn_col = telco_df['churn'].copy()   #separate churn from the rest

#separate df in categorical vars and quant vars:
    
cat_df = telco_df[['contract_type', 'gender',
        'partner', 'dependents', 'phone_service',
       'multiple_lines', 'online_security',
       'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
       'streaming_movies', 'paperless_billing', 'payment_type', 'internet_service_type_id.1', 'payment_type_id.1',
       'contract_type_id.1', 'contract_type_id', 'senior_citizen', 'internet_service_type_id', 'payment_type_id']].copy()


quant_df = telco_df[['tenure', 'monthly_charges', 'total_charges']].copy()

quant_df


for col in telco_df.columns:
    print(col)
    print('-' * 20)
    print(telco_df[col].unique())
    print('-' * 20)


viz_df = cat_df.copy()
viz_df = viz_df.reset_index()
viz_index = viz_df["customer_id"]
change_rubric = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, 'No phone service': 2, 
                 'No internet service': 2}

row_catcher = {}

for col in viz_df.columns:
    row_catcher[col] = {}
    rows = viz_df[col]
    print(f"rows {rows}")
    for index, row in enumerate(rows):
        print(index, row)
        if row in list(change_rubric.keys()):
            print(row)
            row = change_rubric[row]
            print(row)
            row_catcher[col][index] = row
            # value = pd.Series(row)
            # value.set_index(index)
            # row_catcher = row_catcher.append(value, )

row_catcher2 = {key: pd.Series(value) for key, value in row_catcher.items() if len(value) > 0}
row_catcher2 = pd.concat(row_catcher2, axis=1)

# manually add in those few that have weird extra values as seen in the unique explorer

bar_df = {}
for col in row_catcher2.columns:
    og_length = len(row_catcher2[col])
    rows = row_catcher2[col].dropna()
    nans = og_length - len(rows)
    
    # trues = sum(rows)
    # falses = len(rows) - trues
    
    # i guess you can do like:
    temp0 = len(rows[rows == 0])
    temp1 = len(rows[rows == 1])
    temp2 = len(rows[rows == 2])
    
    
    bar_df[col] = {"0": temp0, "1": temp1, "2": temp2, "other": nans}
    
    
bar_df = pd.DataFrame.from_dict(bar_df, orient="index")


for col in bar_df.columns:
    print(col)
    bar_df[col].plot.bar(label=col)
    plt.legend()
    plt.show()
    plt.close()


cat_dummies = pd.get_dummies(row_catcher2, dummy_na=False)
cat_dummies["reindex"] = viz_index
cat_dummies = cat_dummies.set_index("reindex")

# scaling 
quant_df = quant_df[quant_df["tenure"] != 0]
quant_df["total_charges"] = quant_df["total_charges"].astype(float)

quant_df = quant_df.reset_index()
re_index = quant_df["customer_id"]
quant_df = quant_df.drop(["customer_id"], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

catcher = {}
for col in quant_df.columns:
    # value = pd.DataFrame(quant_df[col]).copy().astype(float)
    value = pd.DataFrame(quant_df[col]).copy()
    scaled = scaler.fit_transform(value)
    catcher[col] = pd.DataFrame(scaled)
    
catcher = pd.concat(catcher, axis=1).droplevel(1, axis=1)
catcher["reindex"] = re_index
catcher = catcher.set_index("reindex")

quants = catcher.copy()
cats = cat_dummies.copy()
# can put all 3 together now 

recombined = pd.concat([cats, quants, churn_col], axis=1).dropna()














# cols_to_drop = ['internet_service_type_id', 'payment_type_id']    #dropped all cols with service type id due to same value providing no indication of churn
# # telco_df = telco_df.drop(columns='contract_type_id')
# telco_df = telco_df.drop(cols_to_drop,axis=1)   #Now the og cols: internet_service_type_id, payment_type_id, and contract_type_id have been dropped


# #Looking at the groupings for values within the cols// what's looking interesting?
# for col in telco_df:
#     print(col)
#     print(telco_df[col].value_counts())
    
# #contracttype looks normal, semi equal gender distribution
# #partner: signif more without partner
# #senior citizen- =
# #dependents= more without
# #tenure ...
# #phone_service= way more with
# #multilines = more without
# #online_security= more without/
# #online_backup= a lot of people with and without online backup
# #device_protection= signif more without
# #tech_support= pretty even with and without
# #steaming tv= pretty even with and without
# #streaming_movies= signific more cust are streaming than not
# #paperless_billing= a lot of ppl are oaying sim bills/ with a few outliers
# #monthly_charges=...
# #total_charges = ...
# #churn= 5174 not churn/1869 are churn
# #internet_service_type= DSL 7043
# #payment_type= credit car all 7043

# #From this decided to drop the cols: 'internet_service_type', and 'payment_type'
# #Same value for all records in telco_df

# cols_to_drop = ['internet_service_type', 'payment_type']


# col_to_drop = ['contract_type']
# telco_df = telco_df.drop(col_to_drop,axis=1) 


# y = telco_df['churn']
# telco_df = telco_df.drop(["churn"], axis=1)

# #Created dummy_df to format str vals in cols to numeric to perform stat testing
# dummy_df = pd.get_dummies(telco_df[['gender', 'partner', 'dependents', 'phone_service',
#                                     'multiple_lines', 'online_security', 'online_backup',
#                                     'device_protection', 'tech_support', 'streaming_tv',
#                                     'streaming_movies', 'paperless_billing']], dummy_na=False)
# dummy_df.head()

# #concatting the dummy_df with the telco_df 

# # telco_df = pd.concat([telco_df, dummy_df], axis=1)
# # telco_df.head(1)



# # def clean_data(df):
# #     '''
# #     This function will drop any duplicate observations, 
# #     drop ['deck', 'embarked', 'class', 'age'], fill missing embark_town with 'Southampton'
# #     and create dummy vars from sex and embark_town. 
# #     '''
# #     df = df.drop_duplicates()
# #     df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
# #     df['embark_town'] = df.embark_town.fillna(value='Southampton')
# #     dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
# #     df = pd.concat([df, dummy_df], axis=1)
# #     return df
#     #CAN TIDY UP PRIOR WORK TO WITH THIS FUNCTION, FOR ACQUIRE.PREPARE REQUIREMENT...

# #double check for null/MISSING values, there are none
# telco_df.dropna()



# =============================================================================
# TRAIN, TEST, SPLIT
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Categorical variables are - Male | Female
#Continous Variables are - 59.4, 20, etc.

X = recombined.iloc[:, :-1]
y = recombined.iloc[:, -1] # sometimes this wants .values


def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test
train, validate, test = train_validate_test_split(recombined, 'churn', seed=SEED)
#80% train, 20% test

#WRITE A PREPARE FUNCTION FOR THIS----------------^^^^^^^
# create X & y version of train, where y is a series with just the target variable and X are all the features. 
# train2 = train.reset_index()
# #***Bc Bc index is 


X_train = train.drop(columns=['churn'])
y_train = train['churn']

X_validate = validate.drop(columns=['churn'])
y_validate = validate['churn']

X_test = test.drop(columns=['churn'])
y_test = test['churn']


classifier_rf = RandomForestClassifier()

model = classifier_rf.fit(X_train, y_train)
# =============================================================================
# FEATURE IMPORTANCE AND PERCENTAGE 
# =============================================================================
print(model.feature_importances_) #evaluates the importance or weight of each feature

feat_cols = list(X_train.columns)

feat_imp = list(model.feature_importances_)  #Lets visualize the feature importance

feat_imp = [round(i * 100, 2) for i in feat_imp]

feat_cols = pd.Series(feat_cols)
feat_imp = pd.Series(feat_imp)

feat_df = pd.concat([feat_cols, feat_imp],axis=1)

feat_df = feat_df.set_index(0)

feat_df.plot.bar()
#observations:
    #tenure, monthly charges, and total charges are significant indicators of churn
    #features that would make for a informative model
y_pred = model.predict(X_train)


# Estimate the probability of each species, using the training data.
y_pred_proba = model.predict_proba(X_train)

                                # Compute the Accuracy


print('Accuracy of random forest classifier on training set: {:.2f}'
     .format(model.score(X_train, y_train)))
#Accuracy 1.00 /// it appears the model is overfitting the data


                        # Create a confusion matrix
print(confusion_matrix(y_train, y_pred))
# [[2886    5]
#  [   7 1039]]

print(classification_report(y_train, y_pred))

#       precision    recall  f1-score   support

#           No       1.00      1.00      1.00      2891
#          Yes       1.00      0.99      0.99      1046

#     accuracy                           1.00      3937
#    macro avg       1.00      1.00      1.00      3937
# weighted avg       1.00      1.00      1.00      3937

                        # Validate Model
# Compute the accuracy of the model when run on the validate dataset.                        
print('Accuracy of random forest classifier on test set: {:.2f}'
     .format(model.score(X_validate, y_validate)))
# Accuracy of random forest classifier on test set: 0.77




# =============================================================================
#                             EXPLORE PHASE
# =============================================================================
import explore
cat_vars = list(dummy_df.columns)

num_vars = ['tenure', 'monthly_charges', 'total_charges']

explore.explore_univariate(dummy_df, cat_vars, num_vars)




# =============================================================================
# PREDICTIONS OUTPUT
# =============================================================================
# csv_df = pd.DataFrame()
# csv_df['CustomerID'] = test['customer_id']
# csv_df['Prediction'] = rf2.predict(X_test)
# csv_df = csv_df.reset_index().drop(columns='index')

# proba_df = pd.DataFrame(model.predict_proba(X_test))

# output_df = pd.concat([csv_df, proba_df], axis=1)
# output_df.head()



proba_test = model.predict_proba(X_test)
proba_df = pd.DataFrame(proba_test, columns=model.classes_.tolist())
proba_df.head()

reset_test = test.reset_index()

reset_test.head()



test_proba_df = pd.concat([reset_test, proba_df], axis=1)
test_proba_df.head()

test_proba_df['predicted'] = model.predict(X_test)
test_proba_df.head(20)


#write probability table to a csv called predictions
test_proba_df.to_csv('predictions.csv')