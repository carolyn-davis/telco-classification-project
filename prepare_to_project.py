#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:58:14 2021

@author: carolyndavis
"""

# =============================================================================
#                 PREPARE TO PROJECT 
# =============================================================================
from env import host, user, password
import pandas as pd
import numpy as np
import os


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
telco_df = telco_df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
#drop dupes bc of faulty SQL query

telco_df = telco_df.set_index('customer_id')   #sets index to unique categ/index 'customer id

telco_df.to_excel(r'/Users/carolyndavis/Codeup/classification-exercises\File telco_df.xlsx', index = False)
#^^^ Read it into a an excel file 

telco_df.head(10)


telco_df.columns

cols_to_drop = ['internet_service_type_id', 'payment_type_id']    #dropped all cols with service type id due to same value providing no indication of churn
# telco_df = telco_df.drop(columns='contract_type_id')
telco_df = telco_df.drop(cols_to_drop,axis=1)   #Now the og cols: internet_service_type_id, payment_type_id, and contract_type_id have been dropped


#Looking at the groupings for values within the cols// what's looking interesting?
for col in telco_df:
    print(col)
    print(telco_df[col].value_counts())
    
#contracttype looks normal, semi equal gender distribution
#partner: signif more without partner
#senior citizen- =
#dependents= more without
#tenure ...
#phone_service= way more with
#multilines = more without
#online_security= more without/
#online_backup= a lot of people with and without online backup
#device_protection= signif more without
#tech_support= pretty even with and without
#steaming tv= pretty even with and without
#streaming_movies= signific more cust are streaming than not
#paperless_billing= a lot of ppl are oaying sim bills/ with a few outliers
#monthly_charges=...
#total_charges = ...
#churn= 5174 not churn/1869 are churn
#internet_service_type= DSL 7043
#payment_type= credit car all 7043

#From this decided to drop the cols: 'internet_service_type', and 'payment_type'
#Same value for all records in telco_df

cols_to_drop = ['internet_service_type', 'payment_type']


col_to_drop = ['contract_type']
telco_df = telco_df.drop(col_to_drop,axis=1) 


y = telco_df['churn']
telco_df = telco_df.drop(["churn"], axis=1)

#Created dummy_df to format str vals in cols to numeric to perform stat testing
dummy_df = pd.get_dummies(telco_df[['gender', 'partner', 'dependents', 'phone_service',
                                    'multiple_lines', 'online_security', 'online_backup',
                                    'device_protection', 'tech_support', 'streaming_tv',
                                    'streaming_movies', 'paperless_billing']], dummy_na=False)
dummy_df.head()

#concatting the dummy_df with the telco_df 

# telco_df = pd.concat([telco_df, dummy_df], axis=1)
# telco_df.head(1)


# =============================================================================
#                                 CLEAN THE DATA
# =============================================================================

# def clean_data(df):
#     '''
#     This function will drop any duplicate observations, 
#     drop ['deck', 'embarked', 'class', 'age'], fill missing embark_town with 'Southampton'
#     and create dummy vars from sex and embark_town. 
#     '''
#     df = df.drop_duplicates()
#     df = df.drop(columns=['deck', 'embarked', 'class', 'age'])
#     df['embark_town'] = df.embark_town.fillna(value='Southampton')
#     dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
#     df = pd.concat([df, dummy_df], axis=1)
#     return df
    #CAN TIDY UP PRIOR WORK TO WITH THIS FUNCTION, FOR ACQUIRE.PREPARE REQUIREMENT...

#double check for null values, there are none
telco_df.dropna()



# =============================================================================
#                             EXPLORE
# =============================================================================
import explore
cat_vars = list(dummy_df.columns)

num_vars = ['tenure', 'monthly_charges', 'total_charges']

explore.explore_univariate(dummy_df, cat_vars, num_vars)




# =============================================================================
#                                 TRAIN, VALIDATE, SPLIT
# =============================================================================
from sklearn.model_selection import train_test_split

#Categorical variables are - Male | Female
#Continous Variables are - 59.4, 20, etc.


X = dummy_df.copy()

# you can add the cont. variables to this X (from telco df - ones like monthly charge)

Y = y.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)
#80% train, 20% test

#WRITE A PREPARE FUNCTION FOR THIS----------------^^^^^^^




