####################################################################################
###FOURTH SCRIPT - MACHINE LEARNING ALGOS, ENTIRE AND CLUSTERS, TRANSFORMED POOLED##
####################################################################################


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

This is the fourth script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

In this script, I fit and predict using Machine Learning algorithms on the 
entire dataset in the Transformed pooled specification, as well as in the 
three clusters derived from it.

Namely, I fit and predict with Linear Regressions and Random Forest,
and on the latter we also compute the Shapley Values and SHAP Feature
Importances (Mean Absolute Shapley Values, MASVs).

To simplify the computations, Shapley Values have been computed 
on either smaller forests (100 trees instead of 1000) or forests
with trees slightly less deep than the optimal. Differences in terms
of predictive performances were in the order of 0.02 R2, hence 
making the calculated SVs generalizable to the optimal case too.

On the Linear Regressions, we extract the estimated coefficients, and 
obtain the Absolute Coefficients (ACs).
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

########################################################
###IMPORTING TRAIN AND TEST ENTIRE TRANSFORMED POOLED###
########################################################

X_train_datacomplete_MD_stand_1 = pd.read_csv('C:\\Some\\Local\\Path\\X_train_datacomplete_MD_stand_1.csv')

X_test_datacomplete_MD_stand_1 = pd.read_csv('C:\\Some\\Local\\Path\\X_test_datacomplete_MD_stand_1.csv')

y_train_datacomplete_PD_y = pd.read_csv('C:\\Some\\Local\\Path\\y_train_datacomplete_PD_y.csv')

y_test_datacomplete_PD_y = pd.read_csv('C:\\Some\\Local\\Path\\y_test_datacomplete_PD_y.csv')

for i in [X_train_datacomplete_MD_stand_1, X_test_datacomplete_MD_stand_1,
          y_train_datacomplete_PD_y, y_test_datacomplete_PD_y]:
    
    i.drop(["Unnamed: 0"], axis = 1, inplace = True)


#######################################
###IMPORTING TRAIN AND TEST CLUSTERS###
#######################################
    
for i in [0,1,2]:
    
    Train_name = "train_cluster_mundlaked_" + str(i) 
    
    Test_name = "test_cluster_mundlaked_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i)] = pd.read_csv(path_test)
    
    
for i in [0,1,2]: 
    
    globals()["Train_cluster_" + str(i)].drop(['pid', 'hid', 'cluster', 'pid.1',
                                               'cluster.1','cluster.2', 
                                               'Group Mean Doctor Visits', 
                                               'Group-Demeaned Doctor Visits'], axis = 1, inplace = True)
        
        
    globals()["Test_cluster_" + str(i)].drop(['pid', 'hid', 'cluster', 'pid.1',
                                              'cluster.1','cluster.2', 
                                              'Group Mean Doctor Visits', 
                                              'Group-Demeaned Doctor Visits'], axis = 1, inplace = True)
    
#####################################
####LINEAR REGRESSION IN CLUSTERS####
#####################################
 
#############
##CLUSTER 0##
#############

y_Train_cluster_0 = Train_cluster_0['Doctor Visits']

X_Train_cluster_0 = Train_cluster_0.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_0 = Test_cluster_0['Doctor Visits']

X_Test_cluster_0 = Test_cluster_0.drop(['Doctor Visits'], axis = 1)

linreg_cluster_0 = linreg_train_test(X_train = X_Train_cluster_0, 
                                     y_train = y_Train_cluster_0, 
                                     X_test = X_Test_cluster_0,
                                     y_test = y_Test_cluster_0)    


#Mse_lineareg_test, Mse_lineareg_train, Test_R2, Train_R2, lineareg_fitted

MSE_Test_linreg_0 =linreg_cluster_0[0]

#25.50072773633491

MSE_Train_linreg_0 = linreg_cluster_0[1]

#23.006691250367815

R2_Test_linreg_0 = linreg_cluster_0[2]

#0.1278113307666746

R2_Train_linreg_0 = linreg_cluster_0[3]

#0.1279742395218958

##########################################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION, CLUSTER 0###
##########################################################

X_Train_cluster_0.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                    'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                    'P[D]_disabled': 'Avg. Disability',
                                    'M[D]_disabled': 'Avg. (dev.) Disability',
                                    'P[D]_eversmoke': 'Avg. Ever Smoked',
                                    'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                    'P[D]_masmoke': 'Avg. MA Smoke',
                                    'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                    'P[D]_bmiimp2': 'Avg. BMI',
                                    'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                    'P[D]_insured': 'Avg. Insurance',
                                    'M[D]_insured': 'Avg. (dev.) Insurance',
                                    'P[D]_alone': 'Avg. Whether no relationship',
                                    'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                    'P[D]_hhincap': 'Avg. Log equivalised income',
                                    'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                    'P[D]_gender': 'Avg. Gender',
                                    'P[D]_age': 'Avg. Age',
                                    'M[D]_age': 'Avg. (dev.) Age',
                                    'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                    'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                    'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                    'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                    'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                    'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                    'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                    'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                    'P[D]_lower': 'Avg. Education: whether in school',
                                    'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                    'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                    'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                    'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                    'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                    'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                    'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                    'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                    'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Test_cluster_0.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                    'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                    'P[D]_disabled': 'Avg. Disability',
                                    'M[D]_disabled': 'Avg. (dev.) Disability',
                                    'P[D]_eversmoke': 'Avg. Ever Smoked',
                                    'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                    'P[D]_masmoke': 'Avg. MA Smoke',
                                    'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                    'P[D]_bmiimp2': 'Avg. BMI',
                                    'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                    'P[D]_insured': 'Avg. Insurance',
                                    'M[D]_insured': 'Avg. (dev.) Insurance',
                                    'P[D]_alone': 'Avg. Whether no relationship',
                                    'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                    'P[D]_hhincap': 'Avg. Log equivalised income',
                                    'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                    'P[D]_gender': 'Avg. Gender',
                                    'P[D]_age': 'Avg. Age',
                                    'M[D]_age': 'Avg. (dev.) Age',
                                    'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                    'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                    'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                    'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                    'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                    'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                    'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                    'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                    'P[D]_lower': 'Avg. Education: whether in school',
                                    'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                    'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                    'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                    'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                    'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                    'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                    'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                    'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                    'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Train_cluster_0_c = sm.add_constant(X_Train_cluster_0)

lr_train_clsuter_0 = sm.OLS(y_Train_cluster_0, X_Train_cluster_0_c)

res_0 = lr_train_clsuter_0.fit()

coefs_cluster_0_df = pd.read_html(res_0.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

#coefs_cluster_0_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_0_mundlak.csv')

print(res_0.summary())

#############
##CLUSTER 1##
#############

y_Train_cluster_1 = Train_cluster_1['Doctor Visits']

X_Train_cluster_1 = Train_cluster_1.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_1 = Test_cluster_1['Doctor Visits']

X_Test_cluster_1 = Test_cluster_1.drop(['Doctor Visits'], axis = 1)

linreg_cluster_1 = linreg_train_test(X_train = X_Train_cluster_1, 
                                     y_train = y_Train_cluster_1, 
                                     X_test = X_Test_cluster_1,
                                     y_test = y_Test_cluster_1)    

MSE_Test_linreg_1 = linreg_cluster_1[0]

#6.278332449477104

MSE_Train_linreg_1 = linreg_cluster_1[1]

#7.052784934793279

R2_Test_linreg_1 = linreg_cluster_1[2]

#0.13030754228385877

R2_Train_linreg_1 = linreg_cluster_1[3]

#0.1272258049290831

##########################################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION, CLUSTER 1###
##########################################################

X_Train_cluster_1.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                    'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                    'P[D]_disabled': 'Avg. Disability',
                                    'M[D]_disabled': 'Avg. (dev.) Disability',
                                    'P[D]_eversmoke': 'Avg. Ever Smoked',
                                    'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                    'P[D]_masmoke': 'Avg. MA Smoke',
                                    'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                    'P[D]_bmiimp2': 'Avg. BMI',
                                    'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                    'P[D]_insured': 'Avg. Insurance',
                                    'M[D]_insured': 'Avg. (dev.) Insurance',
                                    'P[D]_alone': 'Avg. Whether no relationship',
                                    'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                    'P[D]_hhincap': 'Avg. Log equivalised income',
                                    'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                    'P[D]_gender': 'Avg. Gender',
                                    'P[D]_age': 'Avg. Age',
                                    'M[D]_age': 'Avg. (dev.) Age',
                                    'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                    'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                    'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                    'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                    'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                    'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                    'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                    'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                    'P[D]_lower': 'Avg. Education: whether in school',
                                    'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                    'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                    'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                    'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                    'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                    'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                    'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                    'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                    'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Test_cluster_1.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                   'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                   'P[D]_disabled': 'Avg. Disability',
                                   'M[D]_disabled': 'Avg. (dev.) Disability',
                                   'P[D]_eversmoke': 'Avg. Ever Smoked',
                                   'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                   'P[D]_masmoke': 'Avg. MA Smoke',
                                   'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                   'P[D]_bmiimp2': 'Avg. BMI',
                                   'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                   'P[D]_insured': 'Avg. Insurance',
                                   'M[D]_insured': 'Avg. (dev.) Insurance',
                                   'P[D]_alone': 'Avg. Whether no relationship',
                                   'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                   'P[D]_hhincap': 'Avg. Log equivalised income',
                                   'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                   'P[D]_gender': 'Avg. Gender',
                                   'P[D]_age': 'Avg. Age',
                                   'M[D]_age': 'Avg. (dev.) Age',
                                   'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                   'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                   'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                   'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                   'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                   'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                   'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                   'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                   'P[D]_lower': 'Avg. Education: whether in school',
                                   'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                   'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                   'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                   'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                   'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                   'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                   'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                   'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                   'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Train_cluster_1_c = sm.add_constant(X_Train_cluster_1)

lr_train_clsuter_1 = sm.OLS(y_Train_cluster_1, X_Train_cluster_1_c)

res_1 = lr_train_clsuter_1.fit()

coefs_cluster_1_df = pd.read_html(res_1.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

#coefs_cluster_1_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_1_mundlak.csv')

print(res_1.summary())

#############
##CLUSTER 2##
#############

y_Train_cluster_2 = Train_cluster_2['Doctor Visits']

X_Train_cluster_2 = Train_cluster_2.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_2 = Test_cluster_2['Doctor Visits']

X_Test_cluster_2 = Test_cluster_2.drop(['Doctor Visits'], axis = 1)

linreg_cluster_2 = linreg_train_test(X_train = X_Train_cluster_2, 
                                     y_train = y_Train_cluster_2, 
                                     X_test = X_Test_cluster_2,
                                     y_test = y_Test_cluster_2)    

MSE_Test_linreg_2 = linreg_cluster_2[0]

#7.862936559425069

MSE_Train_linreg_2 = linreg_cluster_2[1]

#7.665988170639644

R2_Test_linreg_2 = linreg_cluster_2[2]

#0.10620819795470848

R2_Train_linreg_2 = linreg_cluster_2[3]

#0.11419055246686816

#Finite numbers suggest no harmful multicollinearity.

##########################################################
###EXTRACTING COEFFICIENTS LINEAR REGRESSION, CLUSTER 2###
##########################################################

X_Train_cluster_2.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                    'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                    'P[D]_disabled': 'Avg. Disability',
                                    'M[D]_disabled': 'Avg. (dev.) Disability',
                                    'P[D]_eversmoke': 'Avg. Ever Smoked',
                                    'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                    'P[D]_masmoke': 'Avg. MA Smoke',
                                    'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                    'P[D]_bmiimp2': 'Avg. BMI',
                                    'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                    'P[D]_insured': 'Avg. Insurance',
                                    'M[D]_insured': 'Avg. (dev.) Insurance',
                                    'P[D]_alone': 'Avg. Whether no relationship',
                                    'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                    'P[D]_hhincap': 'Avg. Log equivalised income',
                                    'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                    'P[D]_gender': 'Avg. Gender',
                                    'P[D]_age': 'Avg. Age',
                                    'M[D]_age': 'Avg. (dev.) Age',
                                    'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                    'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                    'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                    'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                    'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                    'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                    'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                    'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                    'P[D]_lower': 'Avg. Education: whether in school',
                                    'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                    'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                    'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                    'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                    'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                    'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                    'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                    'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                    'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Test_cluster_2.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                   'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                   'P[D]_disabled': 'Avg. Disability',
                                   'M[D]_disabled': 'Avg. (dev.) Disability',
                                   'P[D]_eversmoke': 'Avg. Ever Smoked',
                                   'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                   'P[D]_masmoke': 'Avg. MA Smoke',
                                   'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                   'P[D]_bmiimp2': 'Avg. BMI',
                                   'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                   'P[D]_insured': 'Avg. Insurance',
                                   'M[D]_insured': 'Avg. (dev.) Insurance',
                                   'P[D]_alone': 'Avg. Whether no relationship',
                                   'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                   'P[D]_hhincap': 'Avg. Log equivalised income',
                                   'M[D]_hhincap': 'Avg. (dev.) Log equivalised income',
                                   'P[D]_gender': 'Avg. Gender',
                                   'P[D]_age': 'Avg. Age',
                                   'M[D]_age': 'Avg. (dev.) Age',
                                   'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                   'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                   'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                   'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                   'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                   'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                   'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                   'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                   'P[D]_lower': 'Avg. Education: whether in school',
                                   'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                   'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                   'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                   'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                   'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                   'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                   'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                   'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                   'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)

X_Train_cluster_2_c = sm.add_constant(X_Train_cluster_2)

lr_train_clsuter_2 = sm.OLS(y_Train_cluster_2, X_Train_cluster_2_c)

res_2 = lr_train_clsuter_2.fit()

coefs_cluster_2_df = pd.read_html(res_2.summary().tables[1].as_html(), header = 0, index_col = 0)[0]

#coefs_cluster_2_df.to_csv('C:\\Some\\Local\\Path\\coefs_cluster_2_mundlak.csv')

#Take home:

#1) The Linear Regressions are healthy, with no indications of multicollinearity.
#2) Across the three clusters, similar R2, but very different MSEs.

##################################
####RANDOM FORESTS IN CLUSTERS####
##################################

###############
###CLUSTER 0###
###############

start_time = time.time()

RF_cluster_0 = RandomForest(X_train = X_Train_cluster_0, 
                            y_train = y_Train_cluster_0, 
                            if_bootstrap = True,
                            optim = True,
                            n_trees = [1000], 
                            n_max_feats = [9], 
                            n_max_depth = [35],  
                            n_min_sample_leaf = [1],
                            n_cv = 4,
                            X_test = X_Test_cluster_0,
                            y_test = y_Test_cluster_0)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

#rf_regr_fitted, best_rf, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2

Optimal_RF_0 = RF_cluster_0[1]

Test_MSE_RF_0 = RF_cluster_0[3]

#20.897605390729083

Train_MSE_RF_0 = RF_cluster_0[4]

#2.6430975762822126

Test_R2_RF_0 = RF_cluster_0[5]

#0.2852496279965845

Train_R2_RF_0 = RF_cluster_0[6]

#0.8998183115993053

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

shap_values_df_cluster_0.to_csv(dest_path + 'SVs_marginal_cluster0_mundlaked_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

'''
###############
###CLUSTER 1###
###############


start_time = time.time()

RF_cluster_1 = RandomForest(X_train = X_Train_cluster_1, 
                            y_train = y_Train_cluster_1, 
                            if_bootstrap = True,
                            optim = True,
                            n_trees = [1000], 
                            n_max_feats = [9], 
                            n_max_depth = [23],  
                            n_min_sample_leaf = [1],
                            n_cv = 4,
                            X_test = X_Test_cluster_1,
                            y_test = y_Test_cluster_1)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

Optimal_RF_1 = RF_cluster_1[1]

Test_MSE_RF_1 = RF_cluster_1[3]

#5.865294497308775

Train_MSE_RF_1 = RF_cluster_1[4]

#1.4841174120616956

Test_R2_RF_1 = RF_cluster_1[5]

#0.1875227335216575

Train_R2_RF_1 = RF_cluster_1[6]

#0.8163421411997381

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

shap_values_df_cluster_1.to_csv(dest_path + 'SVs_marginal_cluster1_mundlaked_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

###############
###CLUSTER 2###
###############

start_time = time.time()

RF_cluster_2 = RandomForest(X_train = X_Train_cluster_2, 
                            y_train = y_Train_cluster_2, 
                            if_bootstrap = True,
                            optim = True,
                            n_trees = [1000], 
                            n_max_feats = [9], 
                            n_max_depth = [23],  
                            n_min_sample_leaf = [1],
                            n_cv = 4,
                            X_test = X_Test_cluster_2,
                            y_test = y_Test_cluster_2)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

Optimal_RF_2 = RF_cluster_2[1]

Test_MSE_RF_2 = RF_cluster_2[3]

#7.39696604611439

Train_MSE_RF_2 = RF_cluster_2[4]

#1.7303651750441966

Test_R2_RF_2 = RF_cluster_2[5]

#0.15917576568774217

Train_R2_RF_2 = RF_cluster_2[6]

#0.8000552850307137

'''
###############################
##SHAPLEY VALUES IN CLUSTER 2##
###############################

start_time = time.time()

explainer_cluster_2 = shap.TreeExplainer(RF_cluster_1[1])

dest_path = 'C:\\Some\\Local\\Path\\'
    
shap_values_cluster_2 = explainer_cluster_2.shap_values(X_Test_cluster_2)

shap_values_df_cluster_2 = pd.DataFrame(shap_values_cluster_2)

shap_values_df_cluster_2.columns = list(X_Test_cluster_2)

shap.summary_plot(shap_values_cluster_2, X_Test_cluster_2)

shap.summary_plot(shap_values_cluster_2, X_Test_cluster_2, plot_type = "bar")

shap_values_df_cluster_2.to_csv(dest_path + 'SVs_marginal_cluster2_mundlaked_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
'''

############################################
###WEIGHTED AVERAGE TEST R2 RF vs. LINREG###
############################################

#Here I compute the Weighted Average Test R2 across the clusters for both the
#Random Forest and the Linear Regression. Weights are given by the clusters
#relative sizes.

#Random Forest:
    
weight_rf_0 = RF_cluster_0[5] * (len(X_Train_cluster_0) + len(X_Test_cluster_0)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1))

weight_rf_1 = RF_cluster_1[5] * (len(X_Train_cluster_1) + len(X_Test_cluster_1)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1))  

weight_rf_2 = RF_cluster_2[5] * (len(X_Train_cluster_2) + len(X_Test_cluster_2)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1)) 
 
watr_rf =  weight_rf_0 + weight_rf_1 + weight_rf_2 

#0.2058097717051096

#Linear Regression:

weight_linreg_0 = linreg_cluster_0[2] * (len(X_Train_cluster_0) + len(X_Test_cluster_0)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1))

weight_linreg_1 = linreg_cluster_1[2] * (len(X_Train_cluster_1) + len(X_Test_cluster_1)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1))    

weight_linreg_2 = linreg_cluster_2[2] * (len(X_Train_cluster_2) + len(X_Test_cluster_2)) / (len(X_train_datacomplete_MD_stand_1) + len(X_test_datacomplete_MD_stand_1))

watr_linreg =  weight_linreg_0 + weight_linreg_1 + weight_linreg_2 

#0.12169874352313656

#((watr_rf / watr_linreg) - 1)*100

#69.1141303081592

############################################################
###LINEAR REGRESSION ON ENTIRE TRANSFORMED POOLED DATASET###
############################################################

X_train_datacomplete_MD_stand_1.drop(['pid','hid'], axis = 1, inplace = True)

X_test_datacomplete_MD_stand_1.drop(['pid','hid'], axis = 1, inplace = True)

X_train_datacomplete_MD_stand_1.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                                  'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                                  'P[D]_disabled': 'Avg. Disability',
                                                  'M[D]_disabled': 'Avg. (dev.) Disability',
                                                  'P[D]_eversmoke': 'Avg. Ever Smoked',
                                                  'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                                  'P[D]_masmoke': 'Avg. MA Smoke',
                                                  'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                                  'P[D]_bmiimp2': 'Avg. BMI',
                                                  'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                                  'P[D]_insured': 'Avg. Insurance',
                                                  'M[D]_insured': 'Avg. (dev.) Insurance',
                                                  'P[D]_alone': 'Avg. Whether no relationship',
                                                  'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                                  'P[D]_hhincap': 'Avg. Equivalised income',
                                                  'M[D]_hhincap': 'Avg. (dev.) Equivalised income',
                                                  'P[D]_gender': 'Avg. Gender',
                                                  'P[D]_age': 'Avg. Age',
                                                  'M[D]_age': 'Avg. (dev.) Age',
                                                  'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                                  'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                                  'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                                  'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                                  'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                                  'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                                  'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                                  'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                                  'P[D]_lower': 'Avg. Education: whether in school',
                                                  'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                                  'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                                  'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                                  'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                                  'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                                  'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                                  'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                                  'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                                  'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)
        
X_test_datacomplete_MD_stand_1.rename(columns = {'P[D]_selfahealthimp': 'Avg. Self-Rated Health',
                                                 'M[D]_selfahealthimp': 'Avg. (dev.) Self-Rated Health',
                                                 'P[D]_disabled': 'Avg. Disability',
                                                 'M[D]_disabled': 'Avg. (dev.) Disability',
                                                 'P[D]_eversmoke': 'Avg. Ever Smoked',
                                                 'M[D]_eversmoke': 'Avg. (dev.) Ever Smoked',
                                                 'P[D]_masmoke': 'Avg. MA Smoke',
                                                 'M[D]_masmoke': 'Avg. (dev.) MA Smoke',
                                                 'P[D]_bmiimp2': 'Avg. BMI',
                                                 'M[D]_bmiimp2': 'Avg. (dev.) BMI',
                                                 'P[D]_insured': 'Avg. Insurance',
                                                 'M[D]_insured': 'Avg. (dev.) Insurance',
                                                 'P[D]_alone': 'Avg. Whether no relationship',
                                                 'M[D]_alone': 'Avg. (dev.) Whether no relationship',
                                                 'P[D]_hhincap': 'Avg. Equivalised income',
                                                 'M[D]_hhincap': 'Avg. (dev.) Equivalised income',
                                                 'P[D]_gender': 'Avg. Gender',
                                                 'P[D]_age': 'Avg. Age',
                                                 'M[D]_age': 'Avg. (dev.) Age',
                                                 'P[D]_PsyScaleimp2': 'Avg. Psychological Scale',
                                                 'M[D]_PsyScaleimp2': 'Avg. (dev.) Psychological Scale',
                                                 'P[D]_PhyScaleimp2': 'Avg. Physiological Scale',
                                                 'M[D]_PhyScaleimp2': 'Avg. (dev.) Physiological Scale',
                                                 'P[D]_BA': 'Avg. Education: whether Bachelor Degree',
                                                 'M[D]_BA': 'Avg. (dev.) Education: whether Bachelor Degree',
                                                 'P[D]_MA': 'Avg. Education: whether Master Degree or higher',
                                                 'M[D]_MA': 'Avg. (dev.) Education: whether Master Degree or higher',
                                                 'P[D]_lower': 'Avg. Education: whether in school',
                                                 'M[D]_lower': 'Avg. (dev.) Education: whether in school',
                                                 'P[D]_FT': 'Avg. Employment: whether Full-Time',
                                                 'M[D]_FT': 'Avg. (dev.) Employment: whether Full-Time',
                                                 'P[D]_NO': 'Avg. Employment: whether Unemployed',
                                                 'M[D]_NO': 'Avg. (dev.) Employment: whether Unemployed',
                                                 'P[D]_PT': 'Avg. Employment: whether Part-Time',
                                                 'M[D]_PT': 'Avg. (dev.) Employment: whether Part-Time',
                                                 'P[D]_VTraining': 'Avg. Employment: whether Vocational Training',
                                                 'M[D]_VTraining': 'Avg. (dev.) Employment: whether Vocational Training'}, inplace = True)


#The term "Mundlak" is used since this strategy to work on panel data
#is inspired to Mundlak, Y. (1978). "On the pooling of time series and 
#cross section data", Econometrica, 69–85.

y_train_mundlak = y_train_datacomplete_PD_y['Doctor Visits']

y_test_mundlak = y_test_datacomplete_PD_y['Doctor Visits']


linreg_mundlak_all = linreg_train_test(X_train = X_train_datacomplete_MD_stand_1, 
                                       y_train = y_train_mundlak, 
                                       X_test = X_test_datacomplete_MD_stand_1,
                                       y_test = y_test_mundlak)  

MSE_Test_linreg_mundlak_all = linreg_mundlak_all[0]

#11.894782080064704

MSE_Train_linreg_mundlak_all = linreg_mundlak_all[1]

#11.959464016543391

R2_Test_linreg_mundlak_all = linreg_mundlak_all[2]

#0.17819654134785046 

R2_Train_linreg_mundlak_all = linreg_mundlak_all[3]

#0.18081576007042388

#############################
###MULTICOLLINEARITY CHECK###
#############################

#X_train_datacomplete_MD_stand_1_c = sm.add_constant(X_train_datacomplete_MD_stand_1)

#lr_train = sm.OLS(y_train_mundlak, X_train_datacomplete_MD_stand_1_c)

#res = lr_train.fit()

#print(res.summary())

#######################################
####RANDOM FOREST ON ENTIRE DATASET####
#######################################

start_time = time.time()

RF_mundlak_all = RandomForest(X_train = X_train_datacomplete_MD_stand_1, 
                              y_train = y_train_mundlak, 
                              if_bootstrap = True,
                              optim = True,
                              n_trees = [1000], 
                              n_max_feats = [9], 
                              n_max_depth = [11],  
                              n_min_sample_leaf = [1],
                              n_cv = 4,
                              X_test = X_test_datacomplete_MD_stand_1,
                              y_test = y_test_mundlak)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

Optimal_RF_mundlak_all = RF_mundlak_all[1]

#Above hyperparmaters - 9 and 11 - with 3000 trees made it worse.

Test_MSE_RF_mundlak_all = RF_mundlak_all[3]

#11.61253675478991

Train_MSE_RF_mundlak_all = RF_mundlak_all[4]

#8.547779313387153

Test_R2_RF_mundlak_all = RF_mundlak_all[5]

#0.19769670393494265

Train_R2_RF_mundlak_all = RF_mundlak_all[6]

#0.41450502378395204

'''
#################################
##SHAPLEY VALUES ENTIRE MUNDLAK##
#################################

start_time = time.time()

explainer_mundlak_all = shap.TreeExplainer(RF_mundlak_all[1])

#09/06/2022, 10:12. To simplify the matter, I compute them on the Test Set.
#Also, using 100 trees and max depth only 9. Hopefully, despite the many more
#variables, the computation time should not explode.

dest_path = 'C:\\Some\\Local\\Path\\'

shap_values_mundlak_all = explainer_mundlak_all.shap_values(X_test_datacomplete_MD_stand_1)

shap_values_df_mundlak_all = pd.DataFrame(shap_values_mundlak_all)

shap_values_df_mundlak_all.columns = list(X_test_datacomplete_MD_stand_1)

shap.summary_plot(shap_values_mundlak_all, X_test_datacomplete_MD_stand_1)

shap.summary_plot(shap_values_mundlak_all, X_test_datacomplete_MD_stand_1, plot_type = "bar")

shap_values_df_mundlak_all.to_csv(dest_path + 'SVs_marginal_mundlak_all_df.csv')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Runtime was 471.19058442115784 seconds with 100 trees and max depth = 9
'''




'''
##########################################
##EXTRACTING THE MASVs FROM THE CLUSTERS##
##########################################

#Needed only for cluster0 and cluster2 (already done for cluster1).

#SVs_marginal_cluster0_mundlaked_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster0_mundlaked_df.csv')

#SVs_marginal_cluster1_mundlaked_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster1_mundlaked_df.csv')

#SVs_marginal_cluster2_mundlaked_df = pd.read_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster2_mundlaked_df.csv')

#SVs_marginal_cluster0_mundlaked_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

#SVs_marginal_cluster1_mundlaked_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

#SVs_marginal_cluster2_mundlaked_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

#SVs_marginal_cluster0_mundlaked_df.loc['MASV'] = SVs_marginal_cluster0_mundlaked_df.abs().mean()

#SVs_marginal_cluster1_mundlaked_df.loc['MASV'] = SVs_marginal_cluster1_mundlaked_df.abs().mean()

#SVs_marginal_cluster2_mundlaked_df.loc['MASV'] = SVs_marginal_cluster2_mundlaked_df.abs().mean()

#Only_masvs_cluster0 = SVs_marginal_cluster0_mundlaked_df.loc['MASV']

#Only_masvs_cluster1 = SVs_marginal_cluster1_mundlaked_df.loc['MASV']

#Only_masvs_cluster2 = SVs_marginal_cluster2_mundlaked_df.loc['MASV']

#Only_masvs_cluster0_t = Only_masvs_cluster0.T

#Only_masvs_cluster1_t = Only_masvs_cluster1.T

#Only_masvs_cluster2_t = Only_masvs_cluster2.T

#Only_masvs_cluster0_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster0_mundlaked_df_adapted.csv')

#Only_masvs_cluster1_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster1_mundlaked_df_adapted.csv')

#Only_masvs_cluster2_t.to_csv('C:\\Some\\Local\\Path\\SVs_marginal_cluster2_mundlaked_df_adapted.csv')
'''


