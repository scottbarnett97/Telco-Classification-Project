# imports used in go here 
import warnings
warnings.filterwarnings("ignore")
# Tabular data friends:
import pandas as pd
import numpy as np
import math
# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns
# Sklearn stuff:

# Data acquisition
from pydataset import data
import scipy.stats as stats
import seaborn as sns
import numpy as np
import env
import os
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import acquire
import prepare
################################################# exploration data visuals and satatistical tests ######################################################## 


# Making a function to create a stack plot to explore feature to churn rates
def percentage_stacked_plot(train,columns_to_plot, super_title):  
    '''
    Prints a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)
    # create a figure
    fig = plt.figure(figsize=(14, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=18,  y=.95)
    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):
        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(train[column], train['churn']).apply(lambda x: x/x.sum()*100, axis=1)
        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['blue','salmon'])
        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='churn', fancybox=True)
        # set title and labels
        plt.ylabel("Percentage %")
        #ax.set_title('Proprtion of Observations by ' + column,
                     #fontsize=10, loc='center')
        ax.tick_params(rotation='auto')
        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
    plt.show()
                   
        
# lets make some layered histograms to evaluate the numerical cust. acct. features
def histogram_plots(train,columns_to_plot, super_title):
    '''
    Prints a histogram for each independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''
    # set number of rows and number of columns
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)
    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
    # loop to each demographic column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):
        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        # histograms for each class (normalized histogram)
        train[train['churn']=='No'][column].plot(kind='hist', ax=ax, density=True, 
                                                   alpha=0.5, color='blue', label='No')
        train[train['churn']=='Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                    alpha=0.5, color='salmon', label='Yes')
        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='churn', fancybox=True)
        # set title and labels
        plt.xlabel('Payment in $USD')
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')
        ax.tick_params(rotation='auto')
        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
    plt.show()
           
            
def get_charts_demographics(train):
    '''
    This creates in demographic churn charts
    '''
    # demographic column names
    demographic_columns = ['partner', 'dependents']
    # stacked plot of demographic columns
    percentage_stacked_plot(train,demographic_columns, 'Proportion of Demographic Features and Churn')
    
    
def get_charts_cat_account(train):
    '''
    This creates in catagorical account features vs churn charts
    '''
    # customer account column names
    account_columns = ['paperless_billing', 'payment_type']
    # stacked plot of catagorical customer account columns
    percentage_stacked_plot(train,account_columns, 'Proportion of Catagorical Customer Account Information and churn')
    
    
def get_charts_services(train):
    '''
    This creates in services churn charts
    '''
    # services column names
    services_columns = [ 'internet_service_type','tech_support']
                        
    # stacked plot of services columns
    percentage_stacked_plot(train,services_columns, 'Proportion of Services Information and churn')

    
def get_charts_num_account(train):
    '''
    chart evaluate the numerical cust. acct. features
    '''
    account_columns_numeric = ['tenure', 'monthly_charges', 'total_charges']
    histogram_plots(train,account_columns_numeric, 'Numerical Customer Account Information')   
    
    
    #################################################  satatistical tests ############################################## 
# function for chi^2 evaluation
def eval_results(p, alpha, group1, group2):
    '''
    this function will take in the p-value, alpha, and a name for the 2 variables 
    you are comparing (group 1 and group 2)
    '''
    if p < alpha:
        print(f'Since the p-value is less than alpha, there exists some relationship between {group1} and the {group2}.\n Therefore, we reject the Ho')
    else:
        print(f'Since the p-value is less than alpha, there is not a significant relationship between {group1} and {group2}.\n Therefore, we fail to reject the Ho')

    
    
def get_partner_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value
    It also evaluates the reslts to the null hypothisis
    '''
    group1='churn'
    group2='partner'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.partner)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)

    
def get_dependents_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value
    It also evaluates the reslts to the null hypothisis
    '''
    group1='churn'
    group2='dependents'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.dependents)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)

    
def get_paperless_billing_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value
    It also evaluates the reslts to the null hypothisis
    '''
    group1='churn'
    group2='paperless_billing'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.paperless_billing)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)  
    
    
def get_electronic_check_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value  
    It also evaluates the reslts to the null hypothisis

    '''
    group1='churn'
    group2='electronic_check'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.electronic_check)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)
    
    
def get_fiber_optic_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value
    It also evaluates the reslts to the null hypothisis

    '''
    group1='churn'
    group2='fiber_optic'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.fiber_optic)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)
    
    
def get_tech_support_Yes_chi(train):
    '''
    This conducts a chi squared test and reurns the chi^2 and p-value
    It also evaluates the reslts to the null hypothisis
    '''
    group1='churn'
    group2='tech_support_Yes'
    alpha = 0.05
    observed = pd.crosstab(train.churn, train.tech_support_Yes)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #print alpha
    print('alpha = 0.05')
    # print the chi2 value, formatted to a float with 4 digits. 
    print(f'chi^2 = {chi2:.4f}') 
    # print the p-value, formatted to a float with 4 digits. 
    print(f'p-value = {p:.4f}')
    eval_results(p, alpha, group1, group2)
    
    
def get_tenure_tstat(train):
    '''
    This conducts a T-test and reurns the t and p-value, as well as an evaluation fro rejectin the null hypothisis or not
    '''
    # One Sample T-Test, 2-tailed
    alpha = 0.05
    churn_sample = train[train.churn == 'Yes'].tenure
    overall_mean = train.tenure.mean()
    t, p = stats.ttest_1samp(churn_sample, overall_mean)
    t,p
    print(f't-score = {t}, p-value = {p}')
    # For a 2-tailed test, we take the p-value as is
    if p == alpha:
        print("Since the p-value is equal to alpha, we fail to reject Null Hypothisis")
    else:
        print("Since the p-value is not equal to alpha, we reject Null Hypothisis\n There is a significant relationship between churn and tenure ")
        
    
def get_monthly_charges_ttest(train):
    '''
    This conducts a T-test and reurns the t and p-value, as well as an evaluation fro rejectin the null hypothisis or not
    '''
    # one sample t-test, 1-tail
    alpha = 0.05
    churn_sample = train[train.churn == 'Yes'].monthly_charges
    overall_mean = train.monthly_charges.mean()
    t, p = stats.ttest_1samp(churn_sample, overall_mean)
    print(f't={t}, p/2 = {p/2}')
    # For a 1-tailed test, we evaluate p/2 < α and t > 0(to test if higher)
    if p/2 > alpha:
        print("We fail to reject Null Hypothisis")
    elif t < 0:
        print("We fail to reject Null Hypothisis")
    else:
        print("Since p/2 > alpha and t < 0 We reject Null Hypothisis\n  There is a significant relationship between churn and monthly_charges")
    
    
        ################################################# Modeling ############################################## 

        
        
        
#creating X,y
def get_xy():
    '''
    This function generates X and y for train, validate, and test
    '''
    # Acquiring data
    df = acquire.get_telco_data()
    # Running initial preperation for exploration
    df = prepare.final_prep_telco(df)
    # Split
    train, validate, test = prepare.split_data(df,'churn')
    # create X & y version of train, where y is a series with just the target variable and X are all the features.    
    X_train = train.drop(['churn_Yes','churn','customer_id','partner','dependents','tech_support','paperless_billing','internet_service_type','payment_type'], axis=1)
    y_train = train.churn_Yes 
    X_validate = validate.drop(['churn_Yes','churn','customer_id','partner','dependents','tech_support','paperless_billing','internet_service_type','payment_type'], axis=1)
    y_validate = validate.churn_Yes
    X_test = test.drop(['churn_Yes','churn','customer_id','partner','dependents','tech_support','paperless_billing','internet_service_type','payment_type'], axis=1)
    y_test = test.churn_Yes
    return X_train,y_train,X_validate,y_validate,X_test,y_test
X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()
        
        
        
        
        
# Creating Baselines
def get_baselines():
    '''
    this function returns a baseline for accuracy and recall
    '''
    baseline_prediction = y_train.mode()
    # Predict the majority class in the training set
    baseline_pred = [0] * len(y_train)
    accuracy = accuracy_score(y_train, baseline_pred)
    recall = recall_score(y_train, baseline_pred)
    baseline_results = {'Metric': ['Accuracy', 'Recall'], 'Score': [accuracy, recall]}
    baseline_df = pd.DataFrame(data=baseline_results)
    return baseline_df      
        
        
def create_models2(seed=123):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    This includes best fit hyperparamaenters                
    '''
    models2 = []
    models2.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=100)))
    models2.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models2.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=3,min_samples_split=4,random_state=seed)))
    models2.append(('random_forest', RandomForestClassifier(max_depth=3,random_state=seed)))
    return models2

def get_models():
    # create models list
    models = create_models2(seed=123)
    X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()
    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'accuracy', 'recall'])
    
    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train)
        
        # calculate training accuracy and recall
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_recall = recall_score(y_train, train_predictions)
        
        # make predictions with the validation data
        val_predictions = model.predict(X_validate)
        
        # calculate validation accuracy and recall
        val_accuracy = accuracy_score(y_validate, val_predictions)
        val_recall = recall_score(y_validate, val_predictions)
        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'accuracy': train_accuracy, 'recall': train_recall}, ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'accuracy': val_accuracy, 'recall': val_recall}, ignore_index=True)
        '''
        this section left in case I want to return to printed format rather than data frame
        # print classifier accuracy and recall
        print('Classifier: {}, Train Accuracy: {}, Train Recall: {}, Validation Accuracy: {}, Validation Recall: {}'.format(name, train_accuracy, train_recall, val_accuracy, val_recall))
        '''
    return results

def get_test_model():
    '''
    This will run the logistic regression model on the test set
    '''
    l= LogisticRegression(C=.1,random_state=123)
    l.fit(X_train, y_train)
    y_pred = l.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    '''
    #left here incase i want to go back to printed list, rather than df
    print('Logistic Regression')
    print(f'Accuracy on test: {round(accuracy*100,2)}')
    print(f'Recall on test: {round(recall*100,2)}')
    '''
    results_df = pd.DataFrame({'Model': 'Logistic Regression','Accuracy': [accuracy], 'Recall': [recall]})
    return results_df
    
  