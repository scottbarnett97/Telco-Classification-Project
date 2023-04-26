import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

## this one is for after exploration and the final report

def final_prep_telco(df):
    '''
    This function will drop any duplicate observations,
    Clean up the total_charges
    drop(columns=['Unnamed: 0', 'payment_type_id', 'internet_service_type_id', 'gender', 'contract_type_id', 'senior_citizen', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'streaming_tv', 'streaming_movies', 'total_charges', 'contract_type'])
    and create dummy vars from 'partner','dependents','tech_support','paperless_billing','churn','contract_type','internet_service_type','payment_type' 
    Then it drops the unneded dummies, and corrects fomatting for internet_service_type_Fiber optic'
    '''
    df = df.drop_duplicates()
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    df = df.drop(columns=['Unnamed: 0', 'payment_type_id', 'internet_service_type_id', 'gender', 'contract_type_id', 'senior_citizen', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'streaming_tv', 'streaming_movies', 'total_charges', 'contract_type'])
    dummy_df = pd.get_dummies(df[['partner','dependents','tech_support','churn','internet_service_type','payment_type']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.rename(columns={'internet_service_type_Fiber optic': 'internet_service_type_fiber_optic'})
    df = df.drop(columns=['internet_service_type_None', 'payment_type_Credit card (automatic)','payment_type_Mailed check','tech_support_No internet service','internet_service_type_None'])
    return df

    
## this one is the initial prep for exploration
def prep_telco(df):
    '''
    This function will drop any duplicate observations, 
    drop [payment_type_id', 'internet_service_type_id','contract_type_id']
    and create dummy vars from 'gender','partner','dependents','tech_support','streaming_tv','streaming_movies'
                                ,'paperless_billing','churn','contract_type','internet_service_type','payment_type'. 
    '''
    df = df.drop_duplicates()
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])    
    dummy_df = pd.get_dummies(df[['gender','partner','dependents','tech_support','streaming_tv','streaming_movies', 'paperless_billing','churn','contract_type','internet_service_type','payment_type']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['Unnamed: 0','gender','partner','dependents','tech_support','streaming_tv','streaming_movies', 'paperless_billing','churn','contract_type','internet_service_type','payment_type'])
    return df

                 
def split_data(df,strat):
    '''
    Be sure to code it as train, validate, test = split_data(df,'column you want to stratify')
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[{strat}])
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate[{strat}])
    # This confirms and Validates my split.
    print(f'train -> {train.shape}, {round(train.shape[0]*100 / df.shape[0],2)}%')
    print(f'validate -> {validate.shape},{round(validate.shape[0]*100 / df.shape[0],2)}%')
    print(f'test -> {test.shape}, {round(test.shape[0]*100 / df.shape[0],2)}%')
    return train, validate, test

