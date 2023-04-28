################################################################################
###SECOND SCRIPT - MACHINE LEARNING ALGOS, ENTIRE AND CLUSTERS, ENTIRE POOLED###
################################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS

This is the second script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

In this script, I fit and predict using Machine Learning algorithms on the 
entire dataset in the pooled specification as well as in the five clusters
derived from it.

Namely, I fit and predict with Linear Regressions and Random Forest,
and on the latter I also compute the Shapley Values and SHAP Feature
Importances (Mean Absolute Shapley Values, MASVs).

To simplify the computations, Shapley Values have been computed 
on either smaller forests (100 trees instead of 1000) or forests
with trees slightly less deep than the optimal. Differences in terms
of predictive performances were in the order of 0.02 R2, hence 
making the calculated SVs generalizable to the optimal case too.

On the Linear Regressions, I extract the estimated coefficients, and 
obtain the Absolute Coefficients (ACs), comparable with the MASVs.
'''

def linreg_train_test(X_train, y_train, X_test, y_test):
    
    lineareg = LinearRegression()
    
    X_const_train = sm.add_constant(X_train, has_constant = 'add')
    
    X_const_test = sm.add_constant(X_test, has_constant = 'add')
    
    lineareg_fitted = lineareg.fit(X_const_train, y_train)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_const_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_const_train)

    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()  
    
    lineareg_yhat_train_round = np.round(lineareg_yhat_train)
        
    Test_R2 = r2_score(y_test, lineareg_yhat_test)
    
    Train_R2 = r2_score(y_train, lineareg_yhat_train)

    list_of_results = [Mse_lineareg_test, Mse_lineareg_train, Test_R2, Train_R2, lineareg_fitted]
    
    return list_of_results

def RandomForest(X_train, y_train, if_bootstrap,
                 optim, n_trees, n_max_feats, 
                 n_max_depth, n_min_sample_leaf, 
                 n_cv, X_test = None,
                 y_test = None):
        
    if optim == True:
        
        rf = RandomForestRegressor(bootstrap = if_bootstrap)

        pruning_dict = {'n_estimators':n_trees,
                'max_features': n_max_feats,
                'max_depth':n_max_depth,
                'min_samples_leaf':n_min_sample_leaf
                }
        
        rf_regr_optim = GridSearchCV(rf, 
                        pruning_dict, 
                        cv = n_cv, 
                        n_jobs = -1,
                        scoring = 'neg_mean_squared_error')
        
    else:
        
        rf_regr_optim = RandomForestRegressor(n_estimators = n_trees[0],
                                              max_features = n_max_feats[0],
                                              max_depth = n_max_depth[0])
        
    rf_regr_fitted = rf_regr_optim.fit(X_train, y_train)
        
    best_rf = rf_regr_fitted.best_estimator_
    
    yhat_train = best_rf.predict(X_train)

    Train_MSE = ((yhat_train - y_train)**2).mean() 
    
    results_from_cv = rf_regr_fitted.cv_results_
    
    if X_test is None and y_test is None:
        
        print('No out of sample accuracy was computed')
    
    else:
        
        yhat_test = best_rf.predict(X_test)

        Test_MSE = ((yhat_test - y_test)**2).mean() 
        
        Train_R2 = r2_score(y_train, yhat_train)
        
        Test_R2 = r2_score(y_test, yhat_test)
        
    list_of_results = [rf_regr_fitted, best_rf, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2]
    
    return list_of_results

#######################################
###IMPORTING TRAIN AND TEST CLUSTERS###
#######################################
    
for i in [0,1,2,3,4]:
    
    Train_name = "train_cluster_" + str(i) 
    
    Test_name = "test_cluster_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i)] = pd.read_csv(path_test)
    
#####################################
###IMPORTING ENTIRE POOLED DATASET###
#####################################

train_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_train.csv')

test_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_test.csv')

train_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

test_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

y_train_datacomplete = train_datacomplete['dvisit']

y_test_datacomplete = test_datacomplete['dvisit']

X_train_datacomplete = train_datacomplete.drop(['dvisit', 'pid', 'syear', 'hid'], axis = 1)

X_test_datacomplete = test_datacomplete.drop(['dvisit',  'pid', 'syear', 'hid'], axis = 1)

##################################
###MACHINE LEARNING IN CLUSTERS###
##################################

#######################
#######CLUSTER 0#######
#######################
   

#######################
###LINEAR REGRESSION###
#######################    

const_in_test_0 = []

for i in list(Test_cluster_0):
        
    if Test_cluster_0[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_0.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0.drop(i, axis = 1, inplace = True)
        
len(const_in_test_0)

const_in_train_0 = []

for i in list(Train_cluster_0):
        
    if Train_cluster_0[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_0.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0.drop(i, axis = 1, inplace = True)
        
len(const_in_train_0)

Y_Test_cluster_0 = Test_cluster_0['dvisit']

X_Test_cluster_0 = Test_cluster_0.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_0 = Train_cluster_0['dvisit']

X_Train_cluster_0 = Train_cluster_0.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

X_Train_cluster_0.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                    'disabled': 'Disability',
                                    'eversmoke': 'Ever Smoked',
                                    'masmoke': 'MA Smoke',
                                    'bmiimp2': 'BMI',
                                    'insured': 'Insurance',
                                    'alone': 'Whether no relationship',
                                    'hhincap': 'Log equivalised income',
                                    'gender': 'Gender',
                                    'age': 'Age',
                                    'PsyScaleimp2': 'Psychological Scale',
                                    'PhyScaleimp2': 'Physiological Scale',
                                    'BA': 'Education: whether Bachelor Degree',
                                    'MA': 'Education: whether Master Degree or higher',
                                    'lower': 'Education: whether in school',
                                    'FT': 'Employment: whether Full-Time',
                                    'NO': 'Employment: whether Unemployed',
                                    'PT': 'Employment: whether Part-Time',
                                    'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_Test_cluster_0.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                   'disabled': 'Disability',
                                   'eversmoke': 'Ever Smoked',
                                   'masmoke': 'MA Smoke',
                                   'bmiimp2': 'BMI',
                                   'insured': 'Insurance',
                                   'alone': 'Whether no relationship',
                                   'hhincap': 'Log equivalised income',
                                   'gender': 'Gender',
                                   'age': 'Age',
                                   'PsyScaleimp2': 'Psychological Scale',
                                   'PhyScaleimp2': 'Physiological Scale',
                                   'BA': 'Education: whether Bachelor Degree',
                                   'MA': 'Education: whether Master Degree or higher',
                                   'lower': 'Education: whether in school',
                                   'FT': 'Employment: whether Full-Time',
                                   'NO': 'Employment: whether Unemployed',
                                   'PT': 'Employment: whether Part-Time',
                                   'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

linreg_cluster_0 = linreg_train_test(X_train = X_Train_cluster_0, 
                                     y_train = Y_Train_cluster_0, 
                                     X_test = X_Test_cluster_0, 
                                     y_test = Y_Test_cluster_0)

linreg_cluster_0[2] 

#Test R2 = 0.16113538587722898

linreg_cluster_0[3] 

#Train R2 = 0.15825560915978898

linreg_cluster_0[0] 

#Test MSE = 12.68911623701252

linreg_cluster_0[1] 

#Train MSE = 12.315493969607301

##########################################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION, CLUSTER 0###
##########################################################

X_Train_cluster_0_c = sm.add_constant(X_Train_cluster_0)

lr_train_clsuter_0 = sm.OLS(Y_Train_cluster_0, X_Train_cluster_0_c)

res_0 = lr_train_clsuter_0.fit()

coefs_cluster_0_df = pd.read_html(res_0.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

print(res_0.summary())

#coefs_cluster_0_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_0_pooled.csv')

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_0 = RandomForest(X_train = X_Train_cluster_0, 
                            y_train = Y_Train_cluster_0, 
                            if_bootstrap = True,
                            optim = True, 
                            n_trees = [1000], 
                            n_max_feats = [3], 
                            n_max_depth = [12], 
                            n_min_sample_leaf = [1], 
                            n_cv = 4, 
                            X_test = X_Test_cluster_0,
                            y_test = Y_Test_cluster_0)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


RF_cluster_0[1]

RF_cluster_0[5]

#Test R2 = 0.20191529308110923

RF_cluster_0[6]

#Train R2 = 0.5942523013462404

RF_cluster_0[3]

#Test MSE = 12.07225748062577

RF_cluster_0[4]

#Train MSE = 5.936461698265119
'''
###############################
##SHAPLEY VALUES IN CLUSTER 0##
###############################

start_time = time.time()

explainer_cluster_0 = shap.TreeExplainer(RF_cluster_0[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_cluster_0 = explainer_cluster_0.shap_values(X_Test_cluster_0)

shap_values_df_cluster_0 = pd.DataFrame(shap_values_cluster_0)

shap_values_df_cluster_0.columns = list(X_Test_cluster_0)

shap.summary_plot(shap_values_cluster_0, X_Test_cluster_0)

shap.summary_plot(shap_values_cluster_0, X_Test_cluster_0, plot_type = "bar")

shap_values_df_cluster_0.to_csv(dest_path + 'SVs_marginal_cluster0_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

#######################
#######CLUSTER 1#######
#######################
    

#######################
###LINEAR REGRESSION###
#######################    

const_in_test_1 = []

for i in list(Test_cluster_1):
        
    if Test_cluster_1[i].nunique() == 1:
        
        const_in_test_1.append(i)
            
        Train_cluster_1.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1.drop(i, axis = 1, inplace = True)
        
len(const_in_test_1)

const_in_train_1 = []

for i in list(Train_cluster_1):
        
    if Train_cluster_1[i].nunique() == 1:
        
        const_in_train_1.append(i)
            
        Train_cluster_1.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1.drop(i, axis = 1, inplace = True)
        
len(const_in_train_1)

Y_Test_cluster_1 = Test_cluster_1['dvisit']

X_Test_cluster_1 = Test_cluster_1.drop(['syear', 'pid', 'hid', 'dvisit', 'SE', 'Pension'], axis = 1)

Y_Train_cluster_1 = Train_cluster_1['dvisit']

X_Train_cluster_1 = Train_cluster_1.drop(['syear', 'pid', 'hid', 'dvisit', 'SE', 'Pension'], axis = 1)

X_Train_cluster_1.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                    'disabled': 'Disability',
                                    'eversmoke': 'Ever Smoked',
                                    'masmoke': 'MA Smoke',
                                    'bmiimp2': 'BMI',
                                    'insured': 'Insurance',
                                    'alone': 'Whether no relationship',
                                    'hhincap': 'Log equivalised income',
                                    'gender': 'Gender',
                                    'age': 'Age',
                                    'PsyScaleimp2': 'Psychological Scale',
                                    'PhyScaleimp2': 'Physiological Scale',
                                    'BA': 'Education: whether Bachelor Degree',
                                    'MA': 'Education: whether Master Degree or higher',
                                    'lower': 'Education: whether in school',
                                    'FT': 'Employment: whether Full-Time',
                                    'NO': 'Employment: whether Unemployed',
                                    'PT': 'Employment: whether Part-Time',
                                    'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_Test_cluster_1.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                   'disabled': 'Disability',
                                   'eversmoke': 'Ever Smoked',
                                   'masmoke': 'MA Smoke',
                                   'bmiimp2': 'BMI',
                                   'insured': 'Insurance',
                                   'alone': 'Whether no relationship',
                                   'hhincap': 'Log equivalised income',
                                   'gender': 'Gender',
                                   'age': 'Age',
                                   'PsyScaleimp2': 'Psychological Scale',
                                   'PhyScaleimp2': 'Physiological Scale',
                                   'BA': 'Education: whether Bachelor Degree',
                                   'MA': 'Education: whether Master Degree or higher',
                                   'lower': 'Education: whether in school',
                                   'FT': 'Employment: whether Full-Time',
                                   'NO': 'Employment: whether Unemployed',
                                   'PT': 'Employment: whether Part-Time',
                                   'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

linreg_cluster_1 = linreg_train_test(X_train = X_Train_cluster_1, 
                                     y_train = Y_Train_cluster_1, 
                                     X_test = X_Test_cluster_1, 
                                     y_test = Y_Test_cluster_1)

linreg_cluster_1[2] 

#Test R2 = 0.13930591770495704

linreg_cluster_1[3] 

#Train R2 = 0.14995308758157944

linreg_cluster_1[0] 

#Test MSE = 20.41105260248074

linreg_cluster_1[1] 

#Train MSE = 20.219402437612256

##########################################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION, CLUSTER 1###
##########################################################

X_Train_cluster_1_c = sm.add_constant(X_Train_cluster_1)

lr_train_clsuter_1 = sm.OLS(Y_Train_cluster_1, X_Train_cluster_1_c)

res_1 = lr_train_clsuter_1.fit()

coefs_cluster_1_df = pd.read_html(res_1.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

print(res_1.summary())

#coefs_cluster_1_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_1_pooled.csv')

###################
###RANDOM FOREST###
###################


start_time = time.time()

RF_cluster_1 = RandomForest(X_train = X_Train_cluster_1, 
                            y_train = Y_Train_cluster_1, 
                            if_bootstrap = True,
                            optim = True, 
                            n_trees = [1000], 
                            n_max_feats = [2], 
                            n_max_depth = [25], 
                            n_min_sample_leaf = [1], 
                            n_cv = 4, 
                            X_test = X_Test_cluster_1,
                            y_test = Y_Test_cluster_1)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_1[1]

RF_cluster_1[5]

#Test R2 = 0.24787713648306164

RF_cluster_1[6]

#Train R2 = 0.8594717620894001

RF_cluster_1[3]

#Test MSE = 17.836324945836203

RF_cluster_1[4]

#Train MSE = 3.3426355118202133

'''
###############################
##SHAPLEY VALUES IN CLUSTER 1##
###############################

start_time = time.time()

explainer_cluster_1 = shap.TreeExplainer(RF_cluster_1[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_cluster_1 = explainer_cluster_1.shap_values(X_Test_cluster_1)

shap_values_df_cluster_1 = pd.DataFrame(shap_values_cluster_1)

shap_values_df_cluster_1.columns = list(X_Test_cluster_1)

shap.summary_plot(shap_values_cluster_1, X_Test_cluster_1)

shap.summary_plot(shap_values_cluster_1, X_Test_cluster_1, plot_type = "bar")

shap_values_df_cluster_1.to_csv(dest_path + 'SVs_marginal_cluster1_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

#######################
#######CLUSTER 2#######
#######################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_2 = []

for i in list(Test_cluster_2):
        
    if Test_cluster_2[i].nunique() == 1:
        
        const_in_test_2.append(i)
            
        Train_cluster_2.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2.drop(i, axis = 1, inplace = True)
        
len(const_in_test_2)

const_in_train_2 = []

for i in list(Train_cluster_2):
        
    if Train_cluster_2[i].nunique() == 1:
        
        const_in_train_2.append(i)
            
        Train_cluster_2.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2.drop(i, axis = 1, inplace = True)
        
len(const_in_train_2)

Y_Test_cluster_2 = Test_cluster_2['dvisit']

X_Test_cluster_2 = Test_cluster_2.drop(['syear', 'pid', 'hid', 'dvisit', 'SE', 'Pension'], axis = 1)

Y_Train_cluster_2 = Train_cluster_2['dvisit']

X_Train_cluster_2 = Train_cluster_2.drop(['syear', 'pid', 'hid', 'dvisit', 'SE', 'Pension'], axis = 1)

X_Train_cluster_2.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                    'disabled': 'Disability',
                                    'eversmoke': 'Ever Smoked',
                                    'masmoke': 'MA Smoke',
                                    'bmiimp2': 'BMI',
                                    'insured': 'Insurance',
                                    'alone': 'Whether no relationship',
                                    'hhincap': 'Log equivalised income',
                                    'gender': 'Gender',
                                    'age': 'Age',
                                    'PsyScaleimp2': 'Psychological Scale',
                                    'PhyScaleimp2': 'Physiological Scale',
                                    'BA': 'Education: whether Bachelor Degree',
                                    'MA': 'Education: whether Master Degree or higher',
                                    'lower': 'Education: whether in school',
                                    'FT': 'Employment: whether Full-Time',
                                    'NO': 'Employment: whether Unemployed',
                                    'PT': 'Employment: whether Part-Time',
                                    'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_Test_cluster_2.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                   'disabled': 'Disability',
                                   'eversmoke': 'Ever Smoked',
                                   'masmoke': 'MA Smoke',
                                   'bmiimp2': 'BMI',
                                   'insured': 'Insurance',
                                   'alone': 'Whether no relationship',
                                   'hhincap': 'Log equivalised income',
                                   'gender': 'Gender',
                                   'age': 'Age',
                                   'PsyScaleimp2': 'Psychological Scale',
                                   'PhyScaleimp2': 'Physiological Scale',
                                   'BA': 'Education: whether Bachelor Degree',
                                   'MA': 'Education: whether Master Degree or higher',
                                   'lower': 'Education: whether in school',
                                   'FT': 'Employment: whether Full-Time',
                                   'NO': 'Employment: whether Unemployed',
                                   'PT': 'Employment: whether Part-Time',
                                   'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

linreg_cluster_2 = linreg_train_test(X_train = X_Train_cluster_2, 
                                     y_train = Y_Train_cluster_2, 
                                     X_test = X_Test_cluster_2, 
                                     y_test = Y_Test_cluster_2)

linreg_cluster_2[2] 

#Test R2 = 0.1399121136832342

linreg_cluster_2[3] 

#Train R2 = 0.14513052836095885

linreg_cluster_2[0] 

#Test MSE = 7.706827356199911

linreg_cluster_2[1] 

#Train MSE = 7.006297383240808

###############################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION###
###############################################

X_Train_cluster_2_c = sm.add_constant(X_Train_cluster_2)

lr_train_clsuter_2 = sm.OLS(Y_Train_cluster_2, X_Train_cluster_2_c)

res_2 = lr_train_clsuter_2.fit()

coefs_cluster_2_df = pd.read_html(res_2.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

#coefs_cluster_2_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_2_pooled.csv')

print(res_2.summary())

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_2 = RandomForest(X_train = X_Train_cluster_2, 
                            y_train = Y_Train_cluster_2, 
                            if_bootstrap = True,
                            optim = True, 
                            n_trees = [1000], 
                            n_max_feats = [6], 
                            n_max_depth = [11], 
                            n_min_sample_leaf = [1], 
                            n_cv = 4, 
                            X_test = X_Test_cluster_2,
                            y_test = Y_Test_cluster_2)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_2[1]

RF_cluster_2[5]

#Test R2 = 0.18971402628037992

RF_cluster_2[6]

#Train R2 = 0.4343917790958166

RF_cluster_2[3]

#Test MSE = 7.260576747976133

RF_cluster_2[4]

#Train MSE = 4.635584179257864

'''
###############################
##SHAPLEY VALUES IN CLUSTER 2##
###############################


start_time = time.time()

explainer_cluster_2 = shap.TreeExplainer(RF_cluster_2[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_cluster_2 = explainer_cluster_2.shap_values(X_Test_cluster_2)

shap_values_df_cluster_2 = pd.DataFrame(shap_values_cluster_2)

shap_values_df_cluster_2.columns = list(X_Test_cluster_2)

shap.summary_plot(shap_values_cluster_2, X_Test_cluster_2)

shap.summary_plot(shap_values_cluster_2, X_Test_cluster_2, plot_type = "bar")

shap_values_df_cluster_2.to_csv(dest_path + 'SVs_marginal_cluster2_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

#######################
#######CLUSTER 3#######
#######################
   
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_3 = []

for i in list(Test_cluster_3):
        
    if Test_cluster_3[i].nunique() == 1:
        
        const_in_test_3.append(i)
            
        Train_cluster_3.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3.drop(i, axis = 1, inplace = True)
        
len(const_in_test_3)

const_in_train_3 = []

for i in list(Train_cluster_3):
        
    if Train_cluster_3[i].nunique() == 1:
        
        const_in_train_3.append(i)
            
        Train_cluster_3.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_3 = Test_cluster_3['dvisit']

X_Test_cluster_3 = Test_cluster_3.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_3 = Train_cluster_3['dvisit']

X_Train_cluster_3 = Train_cluster_3.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

X_Train_cluster_3.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                    'disabled': 'Disability',
                                    'eversmoke': 'Ever Smoked',
                                    'masmoke': 'MA Smoke',
                                    'bmiimp2': 'BMI',
                                    'insured': 'Insurance',
                                    'alone': 'Whether no relationship',
                                    'hhincap': 'Log equivalised income',
                                    'gender': 'Gender',
                                    'age': 'Age',
                                    'PsyScaleimp2': 'Psychological Scale',
                                    'PhyScaleimp2': 'Physiological Scale',
                                    'BA': 'Education: whether Bachelor Degree',
                                    'MA': 'Education: whether Master Degree or higher',
                                    'lower': 'Education: whether in school',
                                    'FT': 'Employment: whether Full-Time',
                                    'NO': 'Employment: whether Unemployed',
                                    'PT': 'Employment: whether Part-Time',
                                    'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_Test_cluster_3.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                   'disabled': 'Disability',
                                   'eversmoke': 'Ever Smoked',
                                   'masmoke': 'MA Smoke',
                                   'bmiimp2': 'BMI',
                                   'insured': 'Insurance',
                                   'alone': 'Whether no relationship',
                                   'hhincap': 'Log equivalised income',
                                   'gender': 'Gender',
                                   'age': 'Age',
                                   'PsyScaleimp2': 'Psychological Scale',
                                   'PhyScaleimp2': 'Physiological Scale',
                                   'BA': 'Education: whether Bachelor Degree',
                                   'MA': 'Education: whether Master Degree or higher',
                                   'lower': 'Education: whether in school',
                                   'FT': 'Employment: whether Full-Time',
                                   'NO': 'Employment: whether Unemployed',
                                   'PT': 'Employment: whether Part-Time',
                                   'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

linreg_cluster_3 = linreg_train_test(X_train = X_Train_cluster_3, 
                                     y_train = Y_Train_cluster_3, 
                                     X_test = X_Test_cluster_3, 
                                     y_test = Y_Test_cluster_3)

linreg_cluster_3[2] 

#Test R2 = 0.12365066461593832

linreg_cluster_3[3] 

#Train R2 = 0.1417260948132404

linreg_cluster_3[0] 

#Test MSE = 10.935079336216534

linreg_cluster_3[1] 

#Train MSE = 9.401358602683965

###############################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION###
###############################################

X_Train_cluster_3_c = sm.add_constant(X_Train_cluster_3)

lr_train_clsuter_3 = sm.OLS(Y_Train_cluster_3, X_Train_cluster_3_c)

res_3 = lr_train_clsuter_3.fit()

coefs_cluster_3_df = pd.read_html(res_3.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

print(res_3.summary())

#coefs_cluster_3_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_3_pooled.csv')

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_3 = RandomForest(X_train = X_Train_cluster_3, 
                            y_train = Y_Train_cluster_3, 
                            if_bootstrap = True,
                            optim = True, 
                            n_trees = [1000],
                            n_max_feats = [2], 
                            n_max_depth = [21], 
                            n_min_sample_leaf = [1], 
                            n_cv = 4, 
                            X_test = X_Test_cluster_3,
                            y_test = Y_Test_cluster_3)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


RF_cluster_3[1]

RF_cluster_3[5]

#Test R2 = 0.1724548109820735

RF_cluster_3[6]

#Train R2 = 0.7904734114334874

RF_cluster_3[3]

#Test MSE = 10.326101625043709

RF_cluster_3[4]

#Train MSE = 2.295111833188233

'''
###############################
##SHAPLEY VALUES IN CLUSTER 3##
###############################

start_time = time.time()

explainer_cluster_3 = shap.TreeExplainer(RF_cluster_3[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_cluster_3 = explainer_cluster_3.shap_values(X_Test_cluster_3)

shap_values_df_cluster_3 = pd.DataFrame(shap_values_cluster_3)

shap_values_df_cluster_3.columns = list(X_Test_cluster_3)

shap.summary_plot(shap_values_cluster_3, X_Test_cluster_3)

shap.summary_plot(shap_values_cluster_3, X_Test_cluster_3, plot_type = "bar")

shap_values_df_cluster_3.to_csv(dest_path + 'SVs_marginal_cluster3_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

#######################
#######CLUSTER 4#######
#######################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_4 = []

for i in list(Test_cluster_4):
        
    if Test_cluster_4[i].nunique() == 1:
        
        const_in_test_4.append(i)
            
        Train_cluster_4.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4.drop(i, axis = 1, inplace = True)
        
len(const_in_test_4)

const_in_train_4 = []

for i in list(Train_cluster_4):
        
    if Train_cluster_4[i].nunique() == 1:
        
        const_in_train_4.append(i)
            
        Train_cluster_4.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4.drop(i, axis = 1, inplace = True)
        
len(const_in_train_4)

Y_Test_cluster_4 = Test_cluster_4['dvisit']

X_Test_cluster_4 = Test_cluster_4.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_4 = Train_cluster_4['dvisit']

X_Train_cluster_4 = Train_cluster_4.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

X_Train_cluster_4.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                    'disabled': 'Disability',
                                    'eversmoke': 'Ever Smoked',
                                    'masmoke': 'MA Smoke',
                                    'bmiimp2': 'BMI',
                                    'insured': 'Insurance',
                                    'alone': 'Whether no relationship',
                                    'hhincap': 'Log equivalised income',
                                    'gender': 'Gender',
                                    'age': 'Age',
                                    'PsyScaleimp2': 'Psychological Scale',
                                    'PhyScaleimp2': 'Physiological Scale',
                                    'BA': 'Education: whether Bachelor Degree',
                                    'MA': 'Education: whether Master Degree or higher',
                                    'lower': 'Education: whether in school',
                                    'FT': 'Employment: whether Full-Time',
                                    'NO': 'Employment: whether Unemployed',
                                    'PT': 'Employment: whether Part-Time',
                                    'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_Test_cluster_4.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                   'disabled': 'Disability',
                                   'eversmoke': 'Ever Smoked',
                                   'masmoke': 'MA Smoke',
                                   'bmiimp2': 'BMI',
                                   'insured': 'Insurance',
                                   'alone': 'Whether no relationship',
                                   'hhincap': 'Log equivalised income',
                                   'gender': 'Gender',
                                   'age': 'Age',
                                   'PsyScaleimp2': 'Psychological Scale',
                                   'PhyScaleimp2': 'Physiological Scale',
                                   'BA': 'Education: whether Bachelor Degree',
                                   'MA': 'Education: whether Master Degree or higher',
                                   'lower': 'Education: whether in school',
                                   'FT': 'Employment: whether Full-Time',
                                   'NO': 'Employment: whether Unemployed',
                                   'PT': 'Employment: whether Part-Time',
                                   'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

linreg_cluster_4 = linreg_train_test(X_train = X_Train_cluster_4, 
                                     y_train = Y_Train_cluster_4, 
                                     X_test = X_Test_cluster_4, 
                                     y_test = Y_Test_cluster_4)

linreg_cluster_4[2] 

#Test R2 = 0.13328573277030964

linreg_cluster_4[3] 

#Train R2 = 0.13432361191228148

linreg_cluster_4[0] 

#Test MSE = 5.414851068589415

linreg_cluster_4[1] 

#Train MSE = 6.606466269499825

###############################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION###
###############################################

X_Train_cluster_4_c = sm.add_constant(X_Train_cluster_4)

lr_train_clsuter_4 = sm.OLS(Y_Train_cluster_4, X_Train_cluster_4_c)

res_4 = lr_train_clsuter_4.fit()

coefs_cluster_4_df = pd.read_html(res_4.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

print(res_4.summary())

#coefs_cluster_4_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_4_pooled.csv')

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_4 = RandomForest(X_train = X_Train_cluster_4, 
                            y_train = Y_Train_cluster_4, 
                            if_bootstrap = True,
                            optim = True, 
                            n_trees = [1000], 
                            n_max_feats = [2], 
                            n_max_depth = [9], 
                            n_min_sample_leaf = [1], 
                            n_cv = 4, 
                            X_test = X_Test_cluster_4,
                            y_test = Y_Test_cluster_4)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


RF_cluster_4[1]


RF_cluster_4[5]

#Test R2 = 0.13448969627028495

#In this case, with only 100 trees, was even performing better:
#Test R2 = 0.13900010204046997

RF_cluster_4[6]

#Train R2 = 0.436603843256672

RF_cluster_4[3]

#Test MSE = 5.407329232050111

RF_cluster_4[4]

#Train MSE = 4.299594810611236

'''
###############################
##SHAPLEY VALUES IN CLUSTER 4##
###############################

start_time = time.time()

explainer_cluster_4 = shap.TreeExplainer(RF_cluster_4[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_cluster_4 = explainer_cluster_4.shap_values(X_Test_cluster_4)

shap_values_df_cluster_4 = pd.DataFrame(shap_values_cluster_4)

shap_values_df_cluster_4.columns = list(X_Test_cluster_4)

shap.summary_plot(shap_values_cluster_4, X_Test_cluster_4)

shap.summary_plot(shap_values_cluster_4, X_Test_cluster_4, plot_type = "bar")

shap_values_df_cluster_4.to_csv(dest_path + 'SVs_marginal_cluster4_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

#Consideration: The fact that the improvements via Random Forest appears to be larger,
#smaller or not present across the different clusters, suggests that
#the ml improvement is no deterministically associated to the pooling
#procedure, but rather to the clustering itself.

#For full comparison between the two algorithms, therefore, I compute and compare
#the Weighted Average Test R2s. 

############################################
###WEIGHTED AVERAGE TEST R2 RF vs. LINREG###
############################################

#Here I compute the Weighted Average Test R2 across the clusters for both the
#Random Forest and the Linear Regression. Weights are given by the clusters
#relative sizes.

#Random Forest:
    
weight_rf_0 = RF_cluster_0[5] * (len(X_Train_cluster_0) + len(X_Test_cluster_0)) / (len(X_train_datacomplete) + len(X_test_datacomplete))

weight_rf_1 = RF_cluster_1[5] * (len(X_Train_cluster_1) + len(X_Test_cluster_1)) / (len(X_train_datacomplete) + len(X_test_datacomplete)) 

weight_rf_2 = RF_cluster_2[5] * (len(X_Train_cluster_2) + len(X_Test_cluster_2)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

weight_rf_3 = RF_cluster_3[5] * (len(X_Train_cluster_3) + len(X_Test_cluster_3)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

weight_rf_4 = RF_cluster_4[5] * (len(X_Train_cluster_4) + len(X_Test_cluster_4)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

watr_rf =  weight_rf_0 + weight_rf_1 + weight_rf_2 + weight_rf_3 + weight_rf_4  

#0.19797828

#Linear Regression:

weight_linreg_0 = linreg_cluster_0[2] * (len(X_Train_cluster_0) + len(X_Test_cluster_0)) / (len(X_train_datacomplete) + len(X_test_datacomplete))

weight_linreg_1 = linreg_cluster_1[2] * (len(X_Train_cluster_1) + len(X_Test_cluster_1)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

weight_linreg_2 = linreg_cluster_2[2] * (len(X_Train_cluster_2) + len(X_Test_cluster_2)) / (len(X_train_datacomplete) + len(X_test_datacomplete)) 

weight_linreg_3 = linreg_cluster_3[2] * (len(X_Train_cluster_3) + len(X_Test_cluster_3)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

weight_linreg_4 = linreg_cluster_4[2] * (len(X_Train_cluster_4) + len(X_Test_cluster_4)) / (len(X_train_datacomplete) + len(X_test_datacomplete))  

watr_linreg =  weight_linreg_0 + weight_linreg_1 + weight_linreg_2 + weight_linreg_3 + weight_linreg_4  

#0.14463914000000003

#((0.19797828 / 0.14463914000000003) - 1)*100

#36.88%

#########################################
####MACHINE LEARNING ON ENTIRE POOLED####
#########################################

X_train_datacomplete.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                       'disabled': 'Disability',
                                       'eversmoke': 'Ever Smoked',
                                       'masmoke': 'MA Smoke',
                                       'bmiimp2': 'BMI',
                                       'insured': 'Insurance',
                                       'alone': 'Whether no relationship',
                                       'hhincap': 'Equivalised income',
                                       'gender': 'Gender',
                                       'age': 'Age',
                                       'PsyScaleimp2': 'Psychological Scale',
                                       'PhyScaleimp2': 'Physiological Scale',
                                       'BA': 'Education: whether Bachelor Degree',
                                       'MA': 'Education: whether Master Degree or higher',
                                       'lower': 'Education: whether in school',
                                       'FT': 'Employment: whether Full-Time',
                                       'NO': 'Employment: whether Unemployed',
                                       'PT': 'Employment: whether Part-Time',
                                       'VTraining': 'Employment: whether Vocational Training'}, inplace = True)
        
X_test_datacomplete.rename(columns = {'selfahealthimp': 'Self-Rated Health',
                                       'disabled': 'Disability',
                                       'eversmoke': 'Ever Smoked',
                                       'masmoke': 'MA Smoke',
                                       'bmiimp2': 'BMI',
                                       'insured': 'Insurance',
                                       'alone': 'Whether no relationship',
                                       'hhincap': 'Equivalised income',
                                       'gender': 'Gender',
                                       'age': 'Age',
                                       'PsyScaleimp2': 'Psychological Scale',
                                       'PhyScaleimp2': 'Physiological Scale',
                                       'BA': 'Education: whether Bachelor Degree',
                                       'MA': 'Education: whether Master Degree or higher',
                                       'lower': 'Education: whether in school',
                                       'FT': 'Employment: whether Full-Time',
                                       'NO': 'Employment: whether Unemployed',
                                       'PT': 'Employment: whether Part-Time',
                                       'VTraining': 'Employment: whether Vocational Training'}, inplace = True)

#####################################
###LINEAR REGRESSION ENTIRE POOLED###
#####################################

linreg_pooled = linreg_train_test(X_train = X_train_datacomplete, 
                                  y_train = y_train_datacomplete, 
                                  X_test = X_test_datacomplete,
                                  y_test = y_test_datacomplete)  

MSE_Test_linreg_pooled = linreg_pooled[0]

#11.868934408781767

MSE_Train_linreg_pooled = linreg_pooled[1]

#11.99300022136143

R2_Test_linreg_pooled = linreg_pooled[2]

#0.1799823416689863

R2_Train_linreg_pooled = linreg_pooled[3]

#0.17851863952923697

#X_Train_pooled_c = sm.add_constant(X_train_datacomplete)
#lr_train_1 = sm.OLS(y_train_datacomplete, X_Train_pooled_c)
#res = lr_train_1.fit()
#print(res.summary())


###################################
####RANDOM FOREST ENTIRE POOLED####
###################################

start_time = time.time()

RF_pooled = RandomForest(X_train = X_train_datacomplete, 
                         y_train = y_train_datacomplete, 
                         if_bootstrap = True,
                         optim = True, 
                         n_trees = [1000], 
                         n_max_feats = [7], 
                         n_max_depth = [23], 
                         n_min_sample_leaf = [1], 
                         n_cv = 4, 
                         X_test = X_test_datacomplete,
                         y_test = y_test_datacomplete)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


Optimal_RF = RF_pooled[1]

Test_MSE_RF_pooled = RF_pooled[3]

#11.113193649403135

Train_MSE_RF_pooled = RF_pooled[4]

#2.405798960522129

Test_R2_RF_pooled = RF_pooled[5]

#0.23219602374582227

Train_R2_RF_pooled  = RF_pooled[6]

#0.8352106256457235

#Keep in mind that TreeSHAP is (upper bounded) quadratic in the maximum depth
#of each tree. 

'''
################################
##SHAPLEY VALUES ENTIRE POOLED##
################################

start_time = time.time()

explainer_pooled = shap.TreeExplainer(RF_pooled[1])

dest_path = 'C:\\Some\\Local\\Path\\' 

shap_values_pooled = explainer_pooled.shap_values(X_test_datacomplete)

shap_values_df_pooled = pd.DataFrame(shap_values_pooled)

shap_values_df_pooled.columns = list(X_test_datacomplete)

shap.summary_plot(shap_values_pooled, X_test_datacomplete)

shap.summary_plot(shap_values_pooled, X_test_datacomplete, plot_type = "bar")

shap_values_df_pooled.to_csv(dest_path + 'SVs_marginal_pooled_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Runtime was 7606.736219406128 seconds.

dest_path = 'C:\\Some\\Local\\Path\\'

shap_values_df_pooled = pd.read_csv(dest_path + 'SVs_marginal_pooled_df.csv')
'''

'''
##########################################
##EXTRACTING THE MASVs FROM THE CLUSTERS##
##########################################

SVs_marginal_cluster0_pooled_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster0_pooled_df.csv')

SVs_marginal_cluster1_pooled_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster1_pooled_df.csv')

SVs_marginal_cluster2_pooled_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster2_pooled_df.csv')

SVs_marginal_cluster3_pooled_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster3_pooled_df.csv')

SVs_marginal_cluster4_pooled_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster4_pooled_df.csv')

SVs_marginal_cluster0_pooled_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

SVs_marginal_cluster1_pooled_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

SVs_marginal_cluster2_pooled_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

SVs_marginal_cluster3_pooled_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

SVs_marginal_cluster4_pooled_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

SVs_marginal_cluster0_pooled_df.loc['MASV'] = SVs_marginal_cluster0_pooled_df.abs().mean()

SVs_marginal_cluster1_pooled_df.loc['MASV'] = SVs_marginal_cluster1_pooled_df.abs().mean()

SVs_marginal_cluster2_pooled_df.loc['MASV'] = SVs_marginal_cluster2_pooled_df.abs().mean()

SVs_marginal_cluster3_pooled_df.loc['MASV'] = SVs_marginal_cluster3_pooled_df.abs().mean()

SVs_marginal_cluster4_pooled_df.loc['MASV'] = SVs_marginal_cluster4_pooled_df.abs().mean()

Only_masvs_cluster0 = SVs_marginal_cluster0_pooled_df.loc['MASV']

Only_masvs_cluster1 = SVs_marginal_cluster1_pooled_df.loc['MASV']

Only_masvs_cluster2 = SVs_marginal_cluster2_pooled_df.loc['MASV']

Only_masvs_cluster3 = SVs_marginal_cluster3_pooled_df.loc['MASV']

Only_masvs_cluster4 = SVs_marginal_cluster4_pooled_df.loc['MASV']

Only_masvs_cluster0_t = Only_masvs_cluster0.T

Only_masvs_cluster1_t = Only_masvs_cluster1.T

Only_masvs_cluster2_T = Only_masvs_cluster2.T

Only_masvs_cluster3_t = Only_masvs_cluster3.T

Only_masvs_cluster4_t = Only_masvs_cluster4.T

Only_masvs_cluster0_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster0_pooled_df_adapted.csv')

Only_masvs_cluster1_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster1_pooled_df_adapted.csv')

Only_masvs_cluster2_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster1_pooled_df_adapted.csv')

Only_masvs_cluster3_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster3_pooled_df_adapted.csv')

Only_masvs_cluster4_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster4_pooled_df_adapted.csv')
'''


