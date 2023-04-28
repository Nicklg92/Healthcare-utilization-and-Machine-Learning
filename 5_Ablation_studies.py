######################################
###FIFTH SCRIPTS - ABLATION STUDIES###
######################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS

This is the fifth script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

Aim of this script is to perform robustness checks for
the main results in the paper.

In particular, I compute the Weighted Average Test R2 across clusters under two specifications:
one in which I ablate Self-Rated Health, and another where I ablate both Disability
and Physiological Score. 

I am going to do it in the clusters under both specifications 
(the five from the Pooled and the three from the Transformed Pooled).

In the Transformed Pooled one, when ablating one variable I ablate both the
average and deviation from average.

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

########################################
##1) CLUSTERS FROM POOLED SPECIFCATION##
########################################
    
for i in [0,1,2,3,4]:
    
    Train_name = "train_cluster_" + str(i) 
    
    Test_name = "test_cluster_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i)] = pd.read_csv(path_test)
    
###################################
##1.1) ABLATING SELF-RATED HEALTH##
###################################
    
for i in [0,1,2,3,4]:
    
    globals()["Train_cluster_" + str(i) + "_no_srh"] = globals()["Train_cluster_" + str(i)].drop(['selfahealthimp'], axis = 1)
    
    globals()["Test_cluster_" + str(i) + "_no_srh"] = globals()["Test_cluster_" + str(i)].drop(['selfahealthimp'], axis = 1)
    
##################################
###1.1.1) CLUSTER 0 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_0 = []

for i in list(Test_cluster_0_no_srh):
        
    if Test_cluster_0_no_srh[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_0_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_test_0)

const_in_train_0 = []

for i in list(Train_cluster_0_no_srh):
        
    if Train_cluster_0_no_srh[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_0_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_train_0)

Y_Test_cluster_0_no_srh = Test_cluster_0_no_srh['dvisit']

X_Test_cluster_0_no_srh = Test_cluster_0_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_0_no_srh = Train_cluster_0_no_srh['dvisit']

X_Train_cluster_0_no_srh = Train_cluster_0_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_0_no_srh = linreg_train_test(X_train = X_Train_cluster_0_no_srh, 
                                            y_train = Y_Train_cluster_0_no_srh, 
                                            X_test = X_Test_cluster_0_no_srh, 
                                            y_test = Y_Test_cluster_0_no_srh)


linreg_cluster_0_no_srh[2] 

#Test R2 = 0.14159065269441695

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_0_no_srh = RandomForest(X_train = X_Train_cluster_0_no_srh, 
                                   y_train = Y_Train_cluster_0_no_srh, 
                                   if_bootstrap = True,
                                   optim = True, 
                                   n_trees = [1000], 
                                   n_max_feats = [5], 
                                   n_max_depth = [14],  
                                   n_min_sample_leaf = [1], 
                                   n_cv = 4, 
                                   X_test = X_Test_cluster_0_no_srh,
                                   y_test = Y_Test_cluster_0_no_srh)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_0_no_srh[1]

RF_cluster_0_no_srh[5]

#Test R2 = 0.17235296142970546

RF_cluster_0_no_srh[3]

#Test MSE = 12.519433170536178

##################################
###1.1.2) CLUSTER 1 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_1 = []

for i in list(Test_cluster_1_no_srh):
        
    if Test_cluster_1_no_srh[i].nunique() == 1:
        
        const_in_test_1.append(i)
            
        Train_cluster_1_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_test_1)

const_in_train_1 = []

for i in list(Train_cluster_1_no_srh):
        
    if Train_cluster_1_no_srh[i].nunique() == 1:
        
        const_in_train_1.append(i)
            
        Train_cluster_1_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_train_1)

Y_Test_cluster_1_no_srh = Test_cluster_1_no_srh['dvisit']

X_Test_cluster_1_no_srh = Test_cluster_1_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_1_no_srh = Train_cluster_1_no_srh['dvisit']

X_Train_cluster_1_no_srh = Train_cluster_1_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_1_no_srh = linreg_train_test(X_train = X_Train_cluster_1_no_srh, 
                                            y_train = Y_Train_cluster_1_no_srh, 
                                            X_test = X_Test_cluster_1_no_srh, 
                                            y_test = Y_Test_cluster_1_no_srh)


linreg_cluster_1_no_srh[2] 

#Test R2 = 0.1244049958578688

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_1_no_srh = RandomForest(X_train = X_Train_cluster_1_no_srh, 
                                   y_train = Y_Train_cluster_1_no_srh, 
                                   if_bootstrap = True,
                                   optim = True, 
                                   n_trees = [1000], 
                                   n_max_feats = [5], 
                                   n_max_depth = [16],  
                                   n_min_sample_leaf = [1], 
                                   n_cv = 4, 
                                   X_test = X_Test_cluster_1_no_srh,
                                   y_test = Y_Test_cluster_1_no_srh)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_1_no_srh[1]

RF_cluster_1_no_srh[5]

#Test R2 = 0.20689975706114283

RF_cluster_1_no_srh[3]

#Test MSE = 18.80808885603105

##################################
###1.1.3) CLUSTER 2 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_2 = []

for i in list(Test_cluster_2_no_srh):
        
    if Test_cluster_2_no_srh[i].nunique() == 1:
        
        const_in_test_2.append(i)
            
        Train_cluster_2_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_test_2)

const_in_train_2 = []

for i in list(Train_cluster_2_no_srh):
        
    if Train_cluster_2_no_srh[i].nunique() == 1:
        
        const_in_train_2.append(i)
            
        Train_cluster_2_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_train_2)

Y_Test_cluster_2_no_srh = Test_cluster_2_no_srh['dvisit']

X_Test_cluster_2_no_srh = Test_cluster_2_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_2_no_srh = Train_cluster_2_no_srh['dvisit']

X_Train_cluster_2_no_srh = Train_cluster_2_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_2_no_srh = linreg_train_test(X_train = X_Train_cluster_2_no_srh, 
                                            y_train = Y_Train_cluster_2_no_srh, 
                                            X_test = X_Test_cluster_2_no_srh, 
                                            y_test = Y_Test_cluster_2_no_srh)


linreg_cluster_2_no_srh[2] 

#Test R2 = 0.10694636831349713

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_2_no_srh = RandomForest(X_train = X_Train_cluster_2_no_srh, 
                                   y_train = Y_Train_cluster_2_no_srh, 
                                   if_bootstrap = True,
                                   optim = True, 
                                   n_trees = [1000], 
                                   n_max_feats = [5], 
                                   n_max_depth = [16],    
                                   n_min_sample_leaf = [1], 
                                   n_cv = 4, 
                                   X_test = X_Test_cluster_2_no_srh,
                                   y_test = Y_Test_cluster_2_no_srh)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_2_no_srh[1]

RF_cluster_2_no_srh[5]

#Test R2 = 0.14397289892171183

RF_cluster_2_no_srh[3]

#Test MSE = 7.670440643561074

##################################
###1.1.4) CLUSTER 3 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_3 = []

for i in list(Test_cluster_3_no_srh):
        
    if Test_cluster_3_no_srh[i].nunique() == 1:
        
        const_in_test_3.append(i)
            
        Train_cluster_3_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_test_3)

const_in_train_3 = []

for i in list(Train_cluster_3_no_srh):
        
    if Train_cluster_3_no_srh[i].nunique() == 1:
        
        const_in_train_3.append(i)
            
        Train_cluster_3_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_3_no_srh = Test_cluster_3_no_srh['dvisit']

X_Test_cluster_3_no_srh = Test_cluster_3_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_3_no_srh = Train_cluster_3_no_srh['dvisit']

X_Train_cluster_3_no_srh = Train_cluster_3_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

linreg_cluster_3_no_srh = linreg_train_test(X_train = X_Train_cluster_3_no_srh, 
                                            y_train = Y_Train_cluster_3_no_srh, 
                                            X_test = X_Test_cluster_3_no_srh, 
                                            y_test = Y_Test_cluster_3_no_srh)


linreg_cluster_3_no_srh[2] 

#Test R2 = 0.10132972850576571

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_3_no_srh = RandomForest(X_train = X_Train_cluster_3_no_srh, 
                                   y_train = Y_Train_cluster_3_no_srh, 
                                   if_bootstrap = True,
                                   optim = True, 
                                   n_trees = [1000], 
                                   n_max_feats = [3], 
                                   n_max_depth = [16], 
                                   n_min_sample_leaf = [1], 
                                   n_cv = 4, 
                                   X_test = X_Test_cluster_3_no_srh,
                                   y_test = Y_Test_cluster_3_no_srh)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_3_no_srh[1]

RF_cluster_3_no_srh[5]

#Test R2 = 0.13493411267190636

RF_cluster_3_no_srh[3]

#Test MSE = 10.794284570138442

##################################
###1.1.5) CLUSTER 4 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_4 = []

for i in list(Test_cluster_4_no_srh):
        
    if Test_cluster_4_no_srh[i].nunique() == 1:
        
        const_in_test_4.append(i)
            
        Train_cluster_4_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_test_4)

const_in_train_4 = []

for i in list(Train_cluster_4_no_srh):
        
    if Train_cluster_4_no_srh[i].nunique() == 1:
        
        const_in_train_4.append(i)
            
        Train_cluster_4_no_srh.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4_no_srh.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_4_no_srh = Test_cluster_4_no_srh['dvisit']

X_Test_cluster_4_no_srh = Test_cluster_4_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_4_no_srh = Train_cluster_4_no_srh['dvisit']

X_Train_cluster_4_no_srh = Train_cluster_4_no_srh.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

linreg_cluster_4_no_srh = linreg_train_test(X_train = X_Train_cluster_4_no_srh, 
                                            y_train = Y_Train_cluster_4_no_srh, 
                                            X_test = X_Test_cluster_4_no_srh, 
                                            y_test = Y_Test_cluster_4_no_srh)


linreg_cluster_4_no_srh[2] 

#Test R2 = 0.1051046686399425

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_4_no_srh = RandomForest(X_train = X_Train_cluster_4_no_srh, 
                                   y_train = Y_Train_cluster_4_no_srh, 
                                   if_bootstrap = True,
                                   optim = True, 
                                   n_trees = [1000], 
                                   n_max_feats = [2], 
                                   n_max_depth = [12], 
                                   n_min_sample_leaf = [1], 
                                   n_cv = 4, 
                                   X_test = X_Test_cluster_4_no_srh,
                                   y_test = Y_Test_cluster_4_no_srh)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_4_no_srh[1]

RF_cluster_4_no_srh[5]

#Test R2 = 0.10681849289124257

RF_cluster_4_no_srh[3]

#Test MSE = 5.580206789108319

###################################################
##1.2) ABLATING DISABILITY AND PHSYILOGICAL SCALE##
###################################################

for i in [0,1,2,3,4]:
    
    globals()["Train_cluster_" + str(i) + "_no_dis_no_physc"] = globals()["Train_cluster_" + str(i)].drop(['disabled', 'PhyScaleimp2'], axis = 1)
    
    globals()["Test_cluster_" + str(i) + "_no_dis_no_physc"] = globals()["Test_cluster_" + str(i)].drop(['disabled', 'PhyScaleimp2'], axis = 1)
    
##################################
###1.2.1) CLUSTER 0 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_0 = []

for i in list(Test_cluster_0_no_dis_no_physc):
        
    if Test_cluster_0_no_dis_no_physc[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_0_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_test_0)

const_in_train_0 = []

for i in list(Train_cluster_0_no_dis_no_physc):
        
    if Train_cluster_0_no_dis_no_physc[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_0_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_0_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_train_0)

Y_Test_cluster_0_no_dis_no_physc = Test_cluster_0_no_dis_no_physc['dvisit']

X_Test_cluster_0_no_dis_no_physc = Test_cluster_0_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_0_no_dis_no_physc = Train_cluster_0_no_dis_no_physc['dvisit']

X_Train_cluster_0_no_dis_no_physc = Train_cluster_0_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_0_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_0_no_dis_no_physc, 
                                                     y_train = Y_Train_cluster_0_no_dis_no_physc, 
                                                     X_test = X_Test_cluster_0_no_dis_no_physc, 
                                                     y_test = Y_Test_cluster_0_no_dis_no_physc)


linreg_cluster_0_no_dis_no_physc[2] 

#Test R2 = 0.1396546634719964

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_0_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_0_no_dis_no_physc, 
                                            y_train = Y_Train_cluster_0_no_dis_no_physc, 
                                            if_bootstrap = True,
                                            optim = True, 
                                            n_trees = [1000], 
                                            n_max_feats = [3], 
                                            n_max_depth = [13], 
                                            n_min_sample_leaf = [1], 
                                            n_cv = 4, 
                                            X_test = X_Test_cluster_0_no_dis_no_physc,
                                            y_test = Y_Test_cluster_0_no_dis_no_physc)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_0_no_dis_no_physc[1]

RF_cluster_0_no_dis_no_physc[5]

#Test R2 = 0.17611644369400736

RF_cluster_0_no_dis_no_physc[3]

#Test MSE = 12.462504718550393

##################################
###1.2.2) CLUSTER 1 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_1 = []

for i in list(Test_cluster_1_no_dis_no_physc):
        
    if Test_cluster_1_no_dis_no_physc[i].nunique() == 1:
        
        const_in_test_1.append(i)
            
        Train_cluster_1_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_test_1)

const_in_train_1 = []

for i in list(Train_cluster_1_no_dis_no_physc):
        
    if Train_cluster_1_no_dis_no_physc[i].nunique() == 1:
        
        const_in_train_1.append(i)
            
        Train_cluster_1_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_train_1)

Y_Test_cluster_1_no_dis_no_physc = Test_cluster_1_no_dis_no_physc['dvisit']

X_Test_cluster_1_no_dis_no_physc = Test_cluster_1_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_1_no_dis_no_physc = Train_cluster_1_no_dis_no_physc['dvisit']

X_Train_cluster_1_no_dis_no_physc = Train_cluster_1_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_1_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_1_no_dis_no_physc, 
                                                     y_train = Y_Train_cluster_1_no_dis_no_physc, 
                                                     X_test = X_Test_cluster_1_no_dis_no_physc, 
                                                     y_test = Y_Test_cluster_1_no_dis_no_physc)


linreg_cluster_1_no_dis_no_physc[2] 

#Test R2 = 0.1236694404313442

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_1_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_1_no_dis_no_physc, 
                                            y_train = Y_Train_cluster_1_no_dis_no_physc, 
                                            if_bootstrap = True,
                                            optim = True, 
                                            n_trees = [1000], 
                                            n_max_feats = [5], 
                                            n_max_depth = [16],  
                                            n_min_sample_leaf = [1], 
                                            n_cv = 4, 
                                            X_test = X_Test_cluster_1_no_dis_no_physc,
                                            y_test = Y_Test_cluster_1_no_dis_no_physc)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_1_no_dis_no_physc[1]

RF_cluster_1_no_dis_no_physc[5]

#Test R2 = 0.20325840936672757

RF_cluster_1_no_dis_no_physc[3]

#Test MSE = 18.894442115410325

##################################
###1.2.3) CLUSTER 2 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_2 = []

for i in list(Test_cluster_2_no_dis_no_physc):
        
    if Test_cluster_2_no_dis_no_physc[i].nunique() == 1:
        
        const_in_test_2.append(i)
            
        Train_cluster_2_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_test_2)

const_in_train_2 = []

for i in list(Train_cluster_2_no_dis_no_physc):
        
    if Train_cluster_2_no_dis_no_physc[i].nunique() == 1:
        
        const_in_train_2.append(i)
            
        Train_cluster_2_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_train_2)

Y_Test_cluster_2_no_dis_no_physc = Test_cluster_2_no_dis_no_physc['dvisit']

X_Test_cluster_2_no_dis_no_physc = Test_cluster_2_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_2_no_dis_no_physc = Train_cluster_2_no_dis_no_physc['dvisit']

X_Train_cluster_2_no_dis_no_physc = Train_cluster_2_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)


linreg_cluster_2_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_2_no_dis_no_physc, 
                                                     y_train = Y_Train_cluster_2_no_dis_no_physc, 
                                                     X_test = X_Test_cluster_2_no_dis_no_physc, 
                                                     y_test = Y_Test_cluster_2_no_dis_no_physc)


linreg_cluster_2_no_dis_no_physc[2] 

#Test R2 = 0.12112644357344982

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_2_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_2_no_dis_no_physc, 
                                            y_train = Y_Train_cluster_2_no_dis_no_physc, 
                                            if_bootstrap = True,
                                            optim = True, 
                                            n_trees = [1000], 
                                            n_max_feats = [4], 
                                            n_max_depth = [12],    
                                            n_min_sample_leaf = [1], 
                                            n_cv = 4, 
                                            X_test = X_Test_cluster_2_no_dis_no_physc,
                                            y_test = Y_Test_cluster_2_no_dis_no_physc)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_2_no_dis_no_physc[1]

RF_cluster_2_no_dis_no_physc[5]

#Test R2 = 0.16537287096773357

RF_cluster_2_no_dis_no_physc[3]

#Test MSE = 7.47868594894205

##################################
###1.2.4) CLUSTER 3 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_3 = []

for i in list(Test_cluster_3_no_dis_no_physc):
        
    if Test_cluster_3_no_dis_no_physc[i].nunique() == 1:
        
        const_in_test_3.append(i)
            
        Train_cluster_3_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_test_3)

const_in_train_3 = []

for i in list(Train_cluster_3_no_dis_no_physc):
        
    if Train_cluster_3_no_dis_no_physc[i].nunique() == 1:
        
        const_in_train_3.append(i)
            
        Train_cluster_3_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_3_no_dis_no_physc = Test_cluster_3_no_dis_no_physc['dvisit']

X_Test_cluster_3_no_dis_no_physc = Test_cluster_3_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_3_no_dis_no_physc = Train_cluster_3_no_dis_no_physc['dvisit']

X_Train_cluster_3_no_dis_no_physc = Train_cluster_3_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

linreg_cluster_3_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_3_no_dis_no_physc, 
                                                     y_train = Y_Train_cluster_3_no_dis_no_physc, 
                                                     X_test = X_Test_cluster_3_no_dis_no_physc, 
                                                     y_test = Y_Test_cluster_3_no_dis_no_physc)


linreg_cluster_3_no_dis_no_physc[2] 

#Test R2 = 0.09849201432654375

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_3_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_3_no_dis_no_physc, 
                                            y_train = Y_Train_cluster_3_no_dis_no_physc, 
                                            if_bootstrap = True,
                                            optim = True, 
                                            n_trees = [1000], 
                                            n_max_feats = [3], 
                                            n_max_depth = [15], 
                                            n_min_sample_leaf = [1], 
                                            n_cv = 4, 
                                            X_test = X_Test_cluster_3_no_dis_no_physc,
                                            y_test = Y_Test_cluster_3_no_dis_no_physc)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_3_no_dis_no_physc[1]

RF_cluster_3_no_dis_no_physc[5]

#Test R2 = 0.13536714642433134

RF_cluster_3_no_dis_no_physc[3]

#Test MSE = 10.788881178766038

##################################
###1.1.5) CLUSTER 4 WITHOUT SRH###
##################################
    
#######################
###LINEAR REGRESSION###
#######################    

const_in_test_4 = []

for i in list(Test_cluster_4_no_dis_no_physc):
        
    if Test_cluster_4_no_dis_no_physc[i].nunique() == 1:
        
        const_in_test_4.append(i)
            
        Train_cluster_4_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_test_4)

const_in_train_4 = []

for i in list(Train_cluster_4_no_dis_no_physc):
        
    if Train_cluster_4_no_dis_no_physc[i].nunique() == 1:
        
        const_in_train_4.append(i)
            
        Train_cluster_4_no_dis_no_physc.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4_no_dis_no_physc.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_4_no_dis_no_physc = Test_cluster_4_no_dis_no_physc['dvisit']

X_Test_cluster_4_no_dis_no_physc = Test_cluster_4_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_4_no_dis_no_physc = Train_cluster_4_no_dis_no_physc['dvisit']

X_Train_cluster_4_no_dis_no_physc = Train_cluster_4_no_dis_no_physc.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

linreg_cluster_4_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_4_no_dis_no_physc, 
                                                     y_train = Y_Train_cluster_4_no_dis_no_physc, 
                                                     X_test = X_Test_cluster_4_no_dis_no_physc, 
                                                     y_test = Y_Test_cluster_4_no_dis_no_physc)


linreg_cluster_4_no_dis_no_physc[2] 

#Test R2 = 0.12669110957553908

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_4_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_4_no_dis_no_physc, 
                                            y_train = Y_Train_cluster_4_no_dis_no_physc, 
                                            if_bootstrap = True,
                                            optim = True, 
                                            n_trees = [1000], 
                                            n_max_feats = [2], 
                                            n_max_depth = [12], 
                                            n_min_sample_leaf = [1], 
                                            n_cv = 4, 
                                            X_test = X_Test_cluster_4_no_dis_no_physc,
                                            y_test = Y_Test_cluster_4_no_dis_no_physc)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_cluster_4_no_dis_no_physc[1]

RF_cluster_4_no_dis_no_physc[5]

#Test R2 = 0.13028011370215098

RF_cluster_4_no_dis_no_physc[3]

#Test MSE = 5.4336288598850535


########################################
########################################
########################################

####################################################
##2) CLUSTERS FROM TRANSFORMED POOLED SPECIFCATION##
####################################################

for i in [0,1,2]:
    
    Train_name = "train_cluster_mundlaked_" + str(i) 
    
    Test_name = "test_cluster_mundlaked_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\'  + Test_name + '.csv'

    globals()["Train_cluster_" + str(i) +'_mundlak'] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i) +'_mundlak'] = pd.read_csv(path_test)
    
for i in [0,1,2]: 
    
    globals()["Train_cluster_" + str(i) +'_mundlak'].drop(['pid', 'hid', 'cluster', 'pid.1',
                                                           'cluster.1','cluster.2', 
                                                           'Group Mean Doctor Visits', 
                                                           'Group-Demeaned Doctor Visits'], axis = 1, inplace = True)
        
        
    globals()["Test_cluster_" + str(i) +'_mundlak'].drop(['pid', 'hid', 'cluster', 'pid.1',
                                                          'cluster.1','cluster.2', 
                                                          'Group Mean Doctor Visits', 
                                                          'Group-Demeaned Doctor Visits'], axis = 1, inplace = True)
    
    
###########################################
##2.1) ABLATING THE TWO SELF RATED HEALTH##
###########################################
    
for i in [0,1,2]:
    
    globals()["Train_cluster_" + str(i) +'_mundlak_no_srh'] = globals()["Train_cluster_" + str(i) +'_mundlak'].drop(['P[D]_selfahealthimp','M[D]_selfahealthimp'], axis = 1)
    
    globals()["Test_cluster_" + str(i) +'_mundlak_no_srh'] = globals()["Test_cluster_" + str(i) +'_mundlak'].drop(['P[D]_selfahealthimp','M[D]_selfahealthimp'], axis = 1)
    
##################################
###2.1.1) CLUSTER 0 WITHOUT SRH###
##################################
    
#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_0_mundlak_no_srh = Train_cluster_0_mundlak_no_srh['Doctor Visits']

X_Train_cluster_0_mundlak_no_srh = Train_cluster_0_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_0_mundlak_no_srh = Test_cluster_0_mundlak_no_srh['Doctor Visits']

X_Test_cluster_0_mundlak_no_srh = Test_cluster_0_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

linreg_cluster_0_mundlak_no_srh = linreg_train_test(X_train = X_Train_cluster_0_mundlak_no_srh, 
                                                    y_train = y_Train_cluster_0_mundlak_no_srh, 
                                                    X_test = X_Test_cluster_0_mundlak_no_srh,
                                                    y_test = y_Test_cluster_0_mundlak_no_srh)    

R2_Test_linreg_0_mundlak_no_srh = linreg_cluster_0_mundlak_no_srh[2]

#0.10935876566067393

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_0_mundlak_no_srh = RandomForest(X_train = X_Train_cluster_0_mundlak_no_srh, 
                                           y_train = y_Train_cluster_0_mundlak_no_srh, 
                                           if_bootstrap = True,
                                           optim = True,
                                           n_trees = [1000], 
                                           n_max_feats = [7], 
                                           n_max_depth = [33],  
                                           n_min_sample_leaf = [1],
                                           n_cv = 4,
                                           X_test = X_Test_cluster_0_mundlak_no_srh,
                                           y_test = y_Test_cluster_0_mundlak_no_srh)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_0_mundlak_no_srh[1]
    
RF_cluster_0_mundlak_no_srh[5]

#Test R2 = 0.2645109595998085

RF_cluster_0_mundlak_no_srh[3]

#Test MSE = 21.503954859663615

##################################
###2.1.2) CLUSTER 1 WITHOUT SRH###
##################################

#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_1_mundlak_no_srh = Train_cluster_1_mundlak_no_srh['Doctor Visits']

X_Train_cluster_1_mundlak_no_srh = Train_cluster_1_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_1_mundlak_no_srh = Test_cluster_1_mundlak_no_srh['Doctor Visits']

X_Test_cluster_1_mundlak_no_srh = Test_cluster_1_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

linreg_cluster_1_mundlak_no_srh = linreg_train_test(X_train = X_Train_cluster_1_mundlak_no_srh, 
                                                    y_train = y_Train_cluster_1_mundlak_no_srh, 
                                                    X_test = X_Test_cluster_1_mundlak_no_srh,
                                                    y_test = y_Test_cluster_1_mundlak_no_srh)    

R2_Test_linreg_1_mundlak_no_srh = linreg_cluster_1_mundlak_no_srh[2]

#0.10531208429417416

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_1_mundlak_no_srh = RandomForest(X_train = X_Train_cluster_1_mundlak_no_srh, 
                                           y_train = y_Train_cluster_1_mundlak_no_srh, 
                                           if_bootstrap = True,
                                           optim = True,
                                           n_trees = [1000], 
                                           n_max_feats = [5], 
                                           n_max_depth = [25],  
                                           n_min_sample_leaf = [1],
                                           n_cv = 4,
                                           X_test = X_Test_cluster_1_mundlak_no_srh,
                                           y_test = y_Test_cluster_1_mundlak_no_srh)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_1_mundlak_no_srh[1]

RF_cluster_1_mundlak_no_srh[5]

#Test R2 = 0.1644365014909993
    
RF_cluster_1_mundlak_no_srh[3]

#Test MSE = 7.682897334567331


##################################
###2.1.3) CLUSTER 2 WITHOUT SRH###
##################################

#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_2_mundlak_no_srh = Train_cluster_2_mundlak_no_srh['Doctor Visits']

X_Train_cluster_2_mundlak_no_srh = Train_cluster_2_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_2_mundlak_no_srh = Test_cluster_2_mundlak_no_srh['Doctor Visits']

X_Test_cluster_2_mundlak_no_srh = Test_cluster_2_mundlak_no_srh.drop(['Doctor Visits'], axis = 1)

linreg_cluster_2_mundlak_no_srh = linreg_train_test(X_train = X_Train_cluster_2_mundlak_no_srh, 
                                                    y_train = y_Train_cluster_2_mundlak_no_srh, 
                                                    X_test = X_Test_cluster_2_mundlak_no_srh,
                                                    y_test = y_Test_cluster_2_mundlak_no_srh)    

R2_Test_linreg_2_mundlak_no_srh = linreg_cluster_2_mundlak_no_srh[2]

#0.07935197623988866

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_2_mundlak_no_srh = RandomForest(X_train = X_Train_cluster_2_mundlak_no_srh, 
                                           y_train = y_Train_cluster_2_mundlak_no_srh, 
                                           if_bootstrap = True,
                                           optim = True,
                                           n_trees = [1000], 
                                           n_max_feats = [5], 
                                           n_max_depth = [25],   
                                           n_min_sample_leaf = [1], 
                                           n_cv = 4,
                                           X_test = X_Test_cluster_2_mundlak_no_srh,
                                           y_test = y_Test_cluster_2_mundlak_no_srh)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_1_mundlak_no_srh[1]

RF_cluster_2_mundlak_no_srh[5]

#Test R2 = 0.1266735269076108
    
RF_cluster_2_mundlak_no_srh[3]

#Test MSE = 7.682897334567331

################################################
##2.2) ABLATING THE TWO DISABLED AND PHYSCALES##
################################################
    
for i in [0,1,2]:
    
    globals()["Train_cluster_" + str(i) +'_mundlak_no_dis_no_physc'] = globals()["Train_cluster_" + str(i) +'_mundlak'].drop(['P[D]_disabled', 'M[D]_disabled', 'P[D]_PsyScaleimp2', 'M[D]_PsyScaleimp2'], axis = 1)
    
    globals()["Test_cluster_" + str(i) +'_mundlak_no_dis_no_physc'] = globals()["Test_cluster_" + str(i) +'_mundlak'].drop(['P[D]_disabled', 'M[D]_disabled', 'P[D]_PsyScaleimp2', 'M[D]_PsyScaleimp2'], axis = 1)
    
#####################################################
###2.2.1) CLUSTER 0 WITHOUT DISABLED AND PHYSCALES###
#####################################################
    
#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_0_mundlak_no_dis_no_physc = Train_cluster_0_mundlak_no_dis_no_physc['Doctor Visits']

X_Train_cluster_0_mundlak_no_dis_no_physc = Train_cluster_0_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_0_mundlak_no_dis_no_physc = Test_cluster_0_mundlak_no_dis_no_physc['Doctor Visits']

X_Test_cluster_0_mundlak_no_dis_no_physc = Test_cluster_0_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

linreg_cluster_0_mundlak_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_0_mundlak_no_dis_no_physc, 
                                                             y_train = y_Train_cluster_0_mundlak_no_dis_no_physc, 
                                                             X_test = X_Test_cluster_0_mundlak_no_dis_no_physc,
                                                             y_test = y_Test_cluster_0_mundlak_no_dis_no_physc)    

R2_Test_linreg_0_mundlak_no_dis_no_physc = linreg_cluster_0_mundlak_no_dis_no_physc[2]

#0.11296288426827439

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_0_mundlak_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_0_mundlak_no_dis_no_physc, 
                                                    y_train = y_Train_cluster_0_mundlak_no_dis_no_physc, 
                                                    if_bootstrap = True,
                                                    optim = True,
                                                    n_trees = [1000], 
                                                    n_max_feats = [5], 
                                                    n_max_depth = [33],  
                                                    n_min_sample_leaf = [1],
                                                    n_cv = 4,
                                                    X_test = X_Test_cluster_0_mundlak_no_dis_no_physc,
                                                    y_test = y_Test_cluster_0_mundlak_no_dis_no_physc)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_0_mundlak_no_dis_no_physc[1]

RF_cluster_0_mundlak_no_dis_no_physc[5]

#Test R2 = 0.2693269781535932

RF_cluster_0_mundlak_no_dis_no_physc[3]

#Test MSE = 21.363145901412466

#####################################################
###2.2.2) CLUSTER 1 WITHOUT DISABLED AND PHYSCALES###
#####################################################

#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_1_mundlak_no_dis_no_physc = Train_cluster_1_mundlak_no_dis_no_physc['Doctor Visits']

X_Train_cluster_1_mundlak_no_dis_no_physc = Train_cluster_1_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_1_mundlak_no_dis_no_physc = Test_cluster_1_mundlak_no_dis_no_physc['Doctor Visits']

X_Test_cluster_1_mundlak_no_dis_no_physc = Test_cluster_1_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

linreg_cluster_1_mundlak_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_1_mundlak_no_dis_no_physc, 
                                                             y_train = y_Train_cluster_1_mundlak_no_dis_no_physc, 
                                                             X_test = X_Test_cluster_1_mundlak_no_dis_no_physc,
                                                             y_test = y_Test_cluster_1_mundlak_no_dis_no_physc)    

R2_Test_linreg_1_mundlak_no_dis_no_physc = linreg_cluster_1_mundlak_no_dis_no_physc[2]

#0.11588263813593724

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_1_mundlak_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_1_mundlak_no_dis_no_physc, 
                                                    y_train = y_Train_cluster_1_mundlak_no_dis_no_physc, 
                                                    if_bootstrap = True,
                                                    optim = True,
                                                    n_trees = [1000], 
                                                    n_max_feats = [5], 
                                                    n_max_depth = [25],   
                                                    n_min_sample_leaf = [1], 
                                                    n_cv = 4,
                                                    X_test = X_Test_cluster_1_mundlak_no_dis_no_physc,
                                                    y_test = y_Test_cluster_1_mundlak_no_dis_no_physc)
 
end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_1_mundlak_no_dis_no_physc[1]
    
RF_cluster_1_mundlak_no_dis_no_physc[5]

#Test R2 = 0.1851560899080501

RF_cluster_1_mundlak_no_dis_no_physc[3]

#Test MSE = 5.882379359047929

#####################################################
###2.2.3) CLUSTER 2 WITHOUT DISABLED AND PHYSCALES###
#####################################################

#########################
####LINEAR REGRESSION####
#########################
    
y_Train_cluster_2_mundlak_no_dis_no_physc = Train_cluster_2_mundlak_no_dis_no_physc['Doctor Visits']

X_Train_cluster_2_mundlak_no_dis_no_physc = Train_cluster_2_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

y_Test_cluster_2_mundlak_no_dis_no_physc = Test_cluster_2_mundlak_no_dis_no_physc['Doctor Visits']

X_Test_cluster_2_mundlak_no_dis_no_physc = Test_cluster_2_mundlak_no_dis_no_physc.drop(['Doctor Visits'], axis = 1)

linreg_cluster_2_mundlak_no_dis_no_physc = linreg_train_test(X_train = X_Train_cluster_2_mundlak_no_dis_no_physc, 
                                                             y_train = y_Train_cluster_2_mundlak_no_dis_no_physc, 
                                                             X_test = X_Test_cluster_2_mundlak_no_dis_no_physc,
                                                             y_test = y_Test_cluster_2_mundlak_no_dis_no_physc)    

R2_Test_linreg_2_mundlak_no_dis_no_physc = linreg_cluster_2_mundlak_no_dis_no_physc[2]

#0.10162898800762021

###################
###RANDOM FOREST###
###################

start_time = time.time()

RF_cluster_2_mundlak_no_dis_no_physc = RandomForest(X_train = X_Train_cluster_2_mundlak_no_dis_no_physc, 
                                                    y_train = y_Train_cluster_2_mundlak_no_dis_no_physc, 
                                                    if_bootstrap = True,
                                                    optim = True,
                                                    n_trees = [1000], 
                                                    n_max_feats = [5], 
                                                    n_max_depth = [24],   
                                                    n_min_sample_leaf = [1], 
                                                    n_cv = 4,
                                                    X_test = X_Test_cluster_2_mundlak_no_dis_no_physc,
                                                    y_test = y_Test_cluster_2_mundlak_no_dis_no_physc)

end_time = time.time()

print("Runtime was " + str(end_time - start_time) + " seconds")

RF_cluster_2_mundlak_no_dis_no_physc[1]

RF_cluster_2_mundlak_no_dis_no_physc[5]    
    
#Test R2 = 0.16749100712496123

RF_cluster_2_mundlak_no_dis_no_physc[3]

#Test MSE = 7.323814540643527

###############################################
###CALCULATIONS OF WEIGHTED AVERAGE TEST R2s###
###############################################

#From previous scripts, we know that the entire dataset has 208903 individuals.

########################################
##5 CLUSTERS FROM POOLED SPECIFICATION##
########################################

####################################
##ABLATION OF SELF-ASSESSED HEALTH##
####################################

#Random Forest:
    
weight_rf_0_no_srh = RF_cluster_0_no_srh[5] * (len(X_Train_cluster_0_no_srh) + len(X_Test_cluster_0_no_srh)) / 208903

weight_rf_1_no_srh = RF_cluster_1_no_srh[5] * (len(X_Train_cluster_1_no_srh) + len(X_Test_cluster_1_no_srh)) / 208903 

weight_rf_2_no_srh = RF_cluster_2_no_srh[5] * (len(X_Train_cluster_2_no_srh) + len(X_Test_cluster_2_no_srh)) / 208903

weight_rf_3_no_srh = RF_cluster_3_no_srh[5] * (len(X_Train_cluster_3_no_srh) + len(X_Test_cluster_3_no_srh)) / 208903 

weight_rf_4_no_srh = RF_cluster_4_no_srh[5] * (len(X_Train_cluster_4_no_srh) + len(X_Test_cluster_4_no_srh)) / 208903
 
watr_rf_no_srh =  weight_rf_0_no_srh + weight_rf_1_no_srh + weight_rf_2_no_srh + weight_rf_3_no_srh + weight_rf_4_no_srh

#0.16036119639484825

#Linear Regression:

weight_linreg_0_no_srh = linreg_cluster_0_no_srh[2] * (len(X_Train_cluster_0_no_srh) + len(X_Test_cluster_0_no_srh)) / 208903

weight_linreg_1_no_srh = linreg_cluster_1_no_srh[2] * (len(X_Train_cluster_1_no_srh) + len(X_Test_cluster_1_no_srh)) / 208903  

weight_linreg_2_no_srh = linreg_cluster_2_no_srh[2] * (len(X_Train_cluster_2_no_srh) + len(X_Test_cluster_2_no_srh)) / 208903

weight_linreg_3_no_srh = linreg_cluster_3_no_srh[2] * (len(X_Train_cluster_3_no_srh) + len(X_Test_cluster_3_no_srh)) / 208903  

weight_linreg_4_no_srh = linreg_cluster_4_no_srh[2] * (len(X_Train_cluster_4_no_srh) + len(X_Test_cluster_4_no_srh)) / 208903

watr_linreg_no_srh =  weight_linreg_0_no_srh + weight_linreg_1_no_srh + weight_linreg_2_no_srh + weight_linreg_3_no_srh + weight_linreg_4_no_srh

#0.1159589758570034

#########################################################
##ABLATION OF DISABILITY STATUS AND PHYSIOLOGICAL SCORE##
#########################################################

#Random Forest:
    
weight_rf_0_no_dis_no_physc = RF_cluster_0_no_dis_no_physc[5] * (len(X_Train_cluster_0_no_dis_no_physc) + len(X_Test_cluster_0_no_dis_no_physc)) / 208903

weight_rf_1_no_dis_no_physc = RF_cluster_1_no_dis_no_physc[5] * (len(X_Train_cluster_1_no_dis_no_physc) + len(X_Test_cluster_1_no_dis_no_physc)) / 208903 

weight_rf_2_no_dis_no_physc = RF_cluster_2_no_dis_no_physc[5] * (len(X_Train_cluster_2_no_dis_no_physc) + len(X_Test_cluster_2_no_dis_no_physc)) / 208903

weight_rf_3_no_dis_no_physc = RF_cluster_3_no_dis_no_physc[5] * (len(X_Train_cluster_3_no_dis_no_physc) + len(X_Test_cluster_3_no_dis_no_physc)) / 208903 

weight_rf_4_no_dis_no_physc = RF_cluster_4_no_dis_no_physc[5] * (len(X_Train_cluster_4_no_dis_no_physc) + len(X_Test_cluster_4_no_dis_no_physc)) / 208903
 
watr_rf_no_dis_no_physc =  weight_rf_0_no_dis_no_physc + weight_rf_1_no_dis_no_physc + weight_rf_2_no_dis_no_physc + weight_rf_3_no_dis_no_physc + weight_rf_4_no_dis_no_physc

#0.16746198983878843

#Linear Regression:

weight_linreg_0_no_dis_no_physc = linreg_cluster_0_no_dis_no_physc[2] * (len(X_Train_cluster_0_no_dis_no_physc) + len(X_Test_cluster_0_no_dis_no_physc)) / 208903

weight_linreg_1_no_dis_no_physc = linreg_cluster_1_no_dis_no_physc[2] * (len(X_Train_cluster_1_no_dis_no_physc) + len(X_Test_cluster_1_no_dis_no_physc)) / 208903  

weight_linreg_2_no_dis_no_physc = linreg_cluster_2_no_dis_no_physc[2] * (len(X_Train_cluster_2_no_dis_no_physc) + len(X_Test_cluster_2_no_dis_no_physc)) / 208903

weight_linreg_3_no_dis_no_physc = linreg_cluster_3_no_dis_no_physc[2] * (len(X_Train_cluster_3_no_dis_no_physc) + len(X_Test_cluster_3_no_dis_no_physc)) / 208903  

weight_linreg_4_no_dis_no_physc = linreg_cluster_4_no_dis_no_physc[2] * (len(X_Train_cluster_4_no_dis_no_physc) + len(X_Test_cluster_4_no_dis_no_physc)) / 208903

watr_linreg_no_dis_no_physc =  weight_linreg_0_no_dis_no_physc + weight_linreg_1_no_dis_no_physc + weight_linreg_2_no_dis_no_physc + weight_linreg_3_no_dis_no_physc + weight_linreg_4_no_dis_no_physc

#0.11987283447425522

##############################################################
##3 CLUSTERS FROM TRANSFORMED POOLED SPECIFICATION (MUNDLAK)##
##############################################################

####################################
##ABLATION OF SELF-ASSESSED HEALTH##
####################################

#Random Forest:
    
weight_rf_0_mundlak_no_srh = RF_cluster_0_mundlak_no_srh[5] * (len(X_Train_cluster_0_mundlak_no_srh) + len(X_Test_cluster_0_mundlak_no_srh)) / 208903

weight_rf_1_mundlak_no_srh = RF_cluster_1_mundlak_no_srh[5] * (len(X_Train_cluster_1_mundlak_no_srh) + len(X_Test_cluster_1_mundlak_no_srh)) / 208903 

weight_rf_2_mundlak_no_srh = RF_cluster_2_mundlak_no_srh[5] * (len(X_Train_cluster_2_mundlak_no_srh) + len(X_Test_cluster_2_mundlak_no_srh)) / 208903
 
watr_rf_mundlak_no_srh =  weight_rf_0_mundlak_no_srh + weight_rf_1_mundlak_no_srh + weight_rf_2_mundlak_no_srh 

#0.17982203274226846

#Linear Regression:

weight_linreg_0_mundlak_no_srh = linreg_cluster_0_mundlak_no_srh[2] * (len(X_Train_cluster_0_mundlak_no_srh) + len(X_Test_cluster_0_mundlak_no_srh)) / 208903

weight_linreg_1_mundlak_no_srh = linreg_cluster_1_mundlak_no_srh[2] * (len(X_Train_cluster_1_mundlak_no_srh) + len(X_Test_cluster_1_mundlak_no_srh)) / 208903  

weight_linreg_2_mundlak_no_srh = linreg_cluster_2_mundlak_no_srh[2] * (len(X_Train_cluster_2_mundlak_no_srh) + len(X_Test_cluster_2_mundlak_no_srh)) / 208903

watr_linreg_mundlak_no_srh =  weight_linreg_0_mundlak_no_srh + weight_linreg_1_mundlak_no_srh + weight_linreg_2_mundlak_no_srh 

#0.09793978156621037

#########################################################
##ABLATION OF DISABILITY STATUS AND PHYSIOLOGICAL SCORE##
#########################################################

#Random Forest:
    
weight_rf_0_mundlak_no_dis_no_physc = RF_cluster_0_mundlak_no_dis_no_physc[5] * (len(X_Train_cluster_0_mundlak_no_dis_no_physc) + len(X_Test_cluster_0_mundlak_no_dis_no_physc)) / 208903

weight_rf_1_mundlak_no_dis_no_physc = RF_cluster_1_mundlak_no_dis_no_physc[5] * (len(X_Train_cluster_1_mundlak_no_dis_no_physc) + len(X_Test_cluster_1_mundlak_no_dis_no_physc)) / 208903 

weight_rf_2_mundlak_no_dis_no_physc = RF_cluster_2_mundlak_no_dis_no_physc[5] * (len(X_Train_cluster_2_mundlak_no_dis_no_physc) + len(X_Test_cluster_2_mundlak_no_dis_no_physc)) / 208903
 
watr_rf_mundlak_no_dis_no_physc =  weight_rf_0_mundlak_no_dis_no_physc + weight_rf_1_mundlak_no_dis_no_physc + weight_rf_2_mundlak_no_dis_no_physc 

#0.20312030271662418

#Linear Regression:

weight_linreg_0_mundlak_no_dis_no_physc = linreg_cluster_0_mundlak_no_dis_no_physc[2] * (len(X_Train_cluster_0_mundlak_no_dis_no_physc) + len(X_Test_cluster_0_mundlak_no_dis_no_physc)) / 208903

weight_linreg_1_mundlak_no_dis_no_physc = linreg_cluster_1_mundlak_no_dis_no_physc[2] * (len(X_Train_cluster_1_mundlak_no_dis_no_physc) + len(X_Test_cluster_1_mundlak_no_dis_no_physc)) / 208903  

weight_linreg_2_mundlak_no_dis_no_physc = linreg_cluster_2_mundlak_no_dis_no_physc[2] * (len(X_Train_cluster_2_mundlak_no_dis_no_physc) + len(X_Test_cluster_2_mundlak_no_dis_no_physc)) / 208903

watr_linreg_mundlak_no_dis_no_physc =  weight_linreg_0_mundlak_no_dis_no_physc + weight_linreg_1_mundlak_no_dis_no_physc + weight_linreg_2_mundlak_no_dis_no_physc 

#0.11038352786634506

