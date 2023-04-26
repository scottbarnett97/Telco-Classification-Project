# Telco-Classification-Project
Classification Project

## Project description with goals

## Initial hypotheses and/or questions you have of the data, ideas


       'tenure', 'phone_service', 'multiple_lines', 'online_security',
       'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
       'streaming_movies', 'paperless_billing', 'monthly_charges',
       'total_charges', 'churn', 'contract_type', 'internet_service_type',
       'payment_type', 'gender_Male', 'partner_Yes', 'dependents_Yes',
       'tech_support_No internet service', 'tech_support_Yes',
       'streaming_tv_No internet service', 'streaming_tv_Yes',
       'streaming_movies_No internet service', 'streaming_movies_Yes',
       'paperless_billing_Yes', 'churn_Yes', 'contract_type_One year',
       'contract_type_Two year', 'internet_service_type_Fiber optic',
       'internet_service_type_None', 'payment_type_Credit card (automatic)',
       'payment_type_Electronic check', 'payment_type_Mailed check'],

## Data Dictionary

|Target|Datatype|Key|Definition
|:-------|:-------|:-------|:----------|
|churn|  int64| 1 = Yes <br /> 0 = No| has customer churned|

|Feature|Datatype|Key|Definition|
|:------- |:-------|:-------|:----------|
|customer_id                        | object | Unique   | identifier for each individual customer's account|
|senior_citizen                     | int64  | 1 = Yes  <br />0 = No    | is senior citizen|
|tenure                             | int64  | Months   | how long a customer has been utilizing telco services|
|monthly_charges                    | float64|  in USD  | how much a customer pays per month|
|total_charges                      | float64|  in USD  | how much a customer has paid since account opening|
|gender_Male                        | int64  | 1 = Male <br />0 = Female| gender|
|partner_Yes                        | int64  | 1 = Yes  <br />0 = No| has a significant other|
|has_dependents                     | int64  | 1 = Yes  <br />0 = No| has children|
|has_phone_service                  | int64  | 1 = Yes  <br />0 = No| has phone service with telco|
|multiple_lines_Yes                 | uint8  | 1 = Yes  <br />0 = No| has multiple phone lines|
|online_security_Yes                | uint8  | 1 = Yes  <br />0 = No| utilizes online security services|
|online_backup_Yes                  | uint8  | 1 = Yes  <br />0 = No| has online backup services via telco|
|device_protection_Yes              | uint8  | 1 = Yes  <br />0 = No| has device protection via telco|
|tech_support_Yes                   | uint8  | 1 = Yes  <br />0 = No| has technical support services with telco|
|streaming_tv_Yes                   | uint8  | 1 = Yes  <br />0 = No| has tv streaming capabilities with their account|
|streaming_movies_Yes               | uint8  | 1 = Yes  <br />0 = No| has movie streaming capabilities with their account|
|contract_type_One year             | uint8  | 1 = Yes  <br />0 = No| must renew their contract every year|
|contract_type_Two year             | uint8  | 1 = Yes  <br />0 = No| must renew their contract every two years|
|internet_service_type_Fiber optic  | uint8  | 1 = Yes  <br />0 = No| has fiber optic internet, 0: doesn't have fiber optic internet|
|internet_service_type_None         | uint8  | 1 = Yes  <br />0 = No| doesn't have internet service via telco|
|payment_type_Credit car (automatic)| uint8  | 1 = Yes  <br />0 = No| makes payments via automatic credit card transfer|
|payment_type_Electronic check      | uint8  | 1 = Yes  <br />0 = No| makes payments via electronic checks|
|payment_type_Mailed check          | uint8  | 1 = Yes  <br />0 = No| makes payments via mailed in checks|


