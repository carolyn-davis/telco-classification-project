TELCO Classification Project


Project Objectives/Goals :
Identify relevant cause of churn at telco. Develop a machine learning classification model to identify key drivers of churn at Telco. Produce deliverables to include a Jupyter Final Notebook outline process of model creation. Develop and produce a predictions CSV supporting classification outcome. 

Prepared Telco data was trained and tested on three machine learning classification models for the aim of identifying three key drivers of churn. 
Out of the three classification models tested in the project, RandomForest and KNN performed significantly better than the third model tested, Logistic Regression. With a test accuracy of 77% for predicting our target value CHURN, it my recommendation to use the RandomForest model. 
Further exploration into the feature importance found in this model presented as additional support for the recommendation presented. 

Key Takeaway: Provide some incentive to our monthly customers to decrease their average rate pf churn with Telco.

PLAN:
•	Imported the data from the SQL database: /telco_churn/
•	Developed the wrangle.py file to easily acquire SQL data later on
•	This function easily grabs the data and presents itself in PD data frame 
•	Devloped hypothesis for possible drivers of churn from skimming data
Hypothesis 1: There is a relationship between monthly charges and churn
Hypothesis 2: There is a relationship between tenure and churn.
•	Theses hypothesis were established to be tested later
•	Created/established the below data dictionary:

DATA DICTIONARY:
Customer_id: provides unique customer identifier for each TELCO customer
senior_citizen: Indicates if customer is a senior citizen (int) 
tenure: time customer has been with the company
monthly_charges: Dollar cost per month (float) 
total_charges: Dollar cost accumulated during tenure (float) 
dependents: Indicates if customer has dependents 
partner: has spouse/significant other on plan
gender_Male: Indicates if customer identifies as male (int)
 phone_service: Indicates if customer has at least 1 phone line (int) 
paperless_billing: Indicates if customer uses paperless billing (int)
 churn:  Indicates if customer has left the company (int) 
contract_type: Indicates telco plan/ two year
internet_service_type: Indicates if customer has DSL internet (int) 
gender: indicates whether customer is male or female
multiple_lines: indicates whether customer has multiple phone lines for their Telco plan
payment_type: indicates what type of payment uses to pay telco charges for service
streaming_movies: indicates whether customer has streaming movie service in Telco contract
streaming_tv: indicates whether customer has streaming tv service in Telco contract
tech_support: indicates whether customer has tech_support as a service in their contract
device_protection: indicates whether customer has device protection as a service in their contract
online_backup: indicates whether customer has online backup as a service in their contract
online_security: indicates whether customer has online security as a service in their contract

ACQUIRE: 
•	Using the wrangle.py data I uploaded the  data into a datafram in my prepare_to_project.py file 
•	Functions stored in wrangle.py were used to retrieve data from SQL database

PREPARATION
•	Due to duplicates being imported from the SQL database, the duplicates were dropped and final dataframe was compared to SQL database for integrity
•	Once the correct amount of records was achieved, and duplicates dopped the ‘customer_id’ was set to be the new index and the new dataframe was written to csv for future retrieval
•	The churn and quantifiable columns(total_charges, monthly_charges, tenuire were  then extracted from the telco_df and set aside.  These columns were copied then dropped. 
•	The resulting dataframes: cat_df/quant_df/churn_df were summarized and visualized in a series of bar plots to identify significant categories correlated to churn
•	After the categories were visualized with a pair plot, three indicators of churn became apparent: tenure, monthly_charges, total_charges. These three categories were connected to drivers churn.
•	The MVP was decided upon, look further into ‘monthly_charges’ as a driver of churn 
•	The data was then put into 3 allocated categories of train/test,/validate. 

EXPLORATION:
•	With the MVP of monthly_charges established, the two hypotheses were tested.
•	Hypothesis one: There is a relationship between monthly_charges and churn.
•	Hypothesis Null: There is no relationship between monthly charges and churn.
-With the performance of CHI square stat test, the null hypothesis was rejected.
•	Hypothesis Two: There is a relationship between tenure and churn.
•	Hypothesis Null: There is no relationship between tenure and churn.
-With the performance of the CHI square test the null hypothesis was rejected.
•	These features indicated higher possible propensities of churn 

MODELING/EVALUATION:
•	The baseline accuracy was established with the train data and had an accuracy of 73.43%
•	Three models were chosen to train, validate, test: RandomForest, KNN, and Logistic Regression 
•	The data was then fitted, transformed, and evaluated to each of the three models. 
•	RandomForest and KNN performed best with accuracies of 77% and 76% respectfully. Logistic regression performed. These both beat the previously established baseline, but ultimately RandomForest was chosen for its accuracy on the dataset.
•	The recommendation was then established; Telco should utilize the RandomForest model to identify/attend to the monthly_to_month churners, due to monthly charges and its correlation to churn. Furthermore, with more time, tenure should additionally be explored. The three features initially identified /tenure/monthly_charges/total_charges should be further explored to attend to major drivers of chuirn.
•	The predictions were then made on this model and translated to a CSV file. See github telco_classification_repo,

