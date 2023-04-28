##################################################
###SIXTH SCRIPT - POISSON AND NEGATIVE BINOMIAL###
##################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PoissonRegressor
from scipy.stats import pearsonr

np.random.seed(1123581321)

'''
COMMENTS

This is the sixth script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

Aim of this script is to perform robustness checks for
the main results in the paper.

In particular, in this script, I fit and predict healthcare utilization using
Poisson and Negative Binomial Regressions.

Healthcare utilization is indeed measured as Number of Doctor Visits in the last
three months, which consists in a count variable. 

For this kind of variables, using a simple Linear Regression can lead to 
negative predictions, and it is by construction heteroskedastic.

A Poisson Regression fixes the above. Moreover, even better a Negative
Binomial also allows to take into account cases where the count variable
has large 0 count, hence I fit and predict also with it.

Overall, in terms of predictive accuracy, already observing that the amount
of negative (nonsensical) prediction by a Linear Regression is low would
represent a solid argument in favour of its use (despite the heteroskedasticity).
'''

##########################
###IMPORTING ALL POOLED###
##########################

train_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_train.csv')

test_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_test.csv')

datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed.csv')

train_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

test_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

#############################
###IMPORTING ALL MUNDLAKED###
#############################

X_datacomplete_MD_stand_1 = pd.read_csv('C:\\Some\\Local\\Path\\X_datacomplete_MD_stand_1.csv')

X_train_datacomplete_MD_stand_1 = pd.read_csv('C:\\Some\\Local\\Path\\X_train_datacomplete_MD_stand_1.csv')

X_test_datacomplete_MD_stand_1 = pd.read_csv('C:\\Some\\Local\\Path\\X_test_datacomplete_MD_stand_1.csv')

y_datacomplete_PD_y = pd.read_csv('C:\\Some\\Local\\Path\\y_datacomplete_PD_y.csv')

y_test_datacomplete_PD_y = pd.read_csv('C:\\Some\\Local\\Path\\y_test_datacomplete_PD_y.csv')

y_train_datacomplete_PD_y = pd.read_csv('C:\\Some\\Local\\Path\\y_train_datacomplete_PD_y.csv')

####################################
###IMPORTING CLUSTERS FROM POOLED###
####################################
    
for i in [0,1,2,3,4]:
    
    Train_name = "train_cluster_" + str(i) 
    
    Test_name = "test_cluster_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i)] = pd.read_csv(path_test)

#######################################
###IMPORTING CLUSTERS FROM MUNDLAKED###
#######################################
    
for i in [0,1,2]:
    
    Train_name = "train_cluster_mundlaked_" + str(i) 
    
    Test_name = "test_cluster_mundlaked_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_MD_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_MD_" + str(i)] = pd.read_csv(path_test)

del Test_name, Train_name, i, path_test, path_train

##################################################
###ROBUSTNESS CHECK 1: USING POISSON REGRESSION###
##################################################

def linreg_train_test(X_train, y_train, X_test, y_test):
    
    lineareg = LinearRegression()
    
    X_const_train = sm.add_constant(X_train, has_constant = 'add')
    
    X_const_test = sm.add_constant(X_test, has_constant = 'add')
    
    lineareg_fitted = lineareg.fit(X_const_train, y_train)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_const_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_const_train)

    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()  
                    
    Train_R2 = r2_score(y_train, lineareg_yhat_train)
        
    Test_R2 = r2_score(y_test, lineareg_yhat_test)
    
    #Adding the Test R2 with rounded negatives.
    
    lineareg_yhat_test_nonegs = lineareg_yhat_test.clip(0)
    
    Test_R2_nonegs = r2_score(y_test, lineareg_yhat_test_nonegs)
    
    list_of_results = [Mse_lineareg_test, Mse_lineareg_train, Test_R2, Train_R2, Test_R2_nonegs, lineareg_yhat_test, lineareg_yhat_train]
    
    return list_of_results

PR_reg = PoissonRegressor(alpha = 0, fit_intercept = True)


#####################
##PR ON POOLED DATA##
#####################

y_train_datacomplete = train_datacomplete['dvisit']

y_test_datacomplete = test_datacomplete['dvisit']

X_train_datacomplete = train_datacomplete.drop(['dvisit', 'pid', 'syear', 'hid'], axis = 1)

X_test_datacomplete = test_datacomplete.drop(['dvisit',  'pid', 'syear', 'hid'], axis = 1)

X_train_datacomplete.rename(columns = {'selfahealthimp': 'Self-Rated Health',
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
        
X_test_datacomplete.rename(columns = {'selfahealthimp': 'Self-Rated Health',
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

PR_reg_pooled = PR_reg.fit(X_train_datacomplete, y_train_datacomplete)

pred_test_PR_reg_pooled = PR_reg_pooled.predict(X_test_datacomplete)

PR_reg_pooled = PR_reg.fit(X_train_datacomplete, y_train_datacomplete)

pred_test_PR_reg_pooled = PR_reg_pooled.predict(X_test_datacomplete)

#let's check both the deviance and the R2.

#dev2_test_PR_reg_pooled = PR_reg_pooled.score(X_test_datacomplete, y_test_datacomplete)

#0.25481669362052406

r2_test_PR_reg_pooled = r2_score(y_test_datacomplete, pred_test_PR_reg_pooled)

#0.18561514115507738 - replicated on 01/07/2022, 11:39. 

#r2_test_PR_reg_pooled_1 = pearsonr(y_test_datacomplete, pred_test_PR_reg_pooled)[0]**2

#0.18583537314342657.

#For a nice explanation of R2 and Deviance:
#https://bookdown.org/egarpor/SSS2-UC3M/logreg-deviance.html

###########################################
##NEGATIVE BINOMIAL ON ENTIRE POOLED DATA##
###########################################

#It is not exactly a piece of cake. The main reference article is:
#https://timeseriesreasoning.com/contents/negative-binomial-regression-model/

#It, in turns, is built on Cameron and Trivedi's book "Regression Analysis of 
#Count Data".

#Here described a bit more in detail, in the other cases will directly write the result.

#Slightly modified by me for easiness. 

#1) STEP 1: Configure and fit the Poisson regression model on the training data set.

X_train_datacomplete_c = sm.add_constant(X_train_datacomplete)

X_test_datacomplete_c = sm.add_constant(X_test_datacomplete)

poisson_training_results_pooled = sm.GLM(y_train_datacomplete, 
                                         X_train_datacomplete_c, 
                                         family = sm.families.Poisson()).fit()

print(poisson_training_results_pooled.summary())

#2) STEP 2: Fit the auxiliary OLS regression model 
#on the data set and use the fitted model to get the value of α.

#Such value α is the coefficient of linear regression where variable are
#the estimated mu parameters of in the previous Poisson, and the dependent variable
#is given by the term defined in AUX_OLS_DEP_pooled.

BB_LAMBDA_pooled = poisson_training_results_pooled.mu

AUX_OLS_DEP_pooled = ((y_train_datacomplete - BB_LAMBDA_pooled)**2 - y_train_datacomplete ) / BB_LAMBDA_pooled 

aux_reg_pooled = sm.OLS(AUX_OLS_DEP_pooled , BB_LAMBDA_pooled).fit()

print(aux_reg_pooled.summary())

#Indeed significant at 0.001 and = 1.0501 with AUX_OLS_DEP_pooled = ((y_train_datacomplete - BB_LAMBDA_pooled )**2 - BB_LAMBDA_pooled ) / BB_LAMBDA_pooled 

#3) STEP 3: Now that we know that this α is statistically different from 0 (meaning
#that we can reject the hypothesis variance = mean, basis of the Poisson Regression) we can
#supply its value into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set.

#https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.NegativeBinomial.html

nb2_training_results_pooled = sm.GLM(y_train_datacomplete, 
                                     X_train_datacomplete_c,
                                     family = sm.families.NegativeBinomial(alpha = aux_reg_pooled.params[0])).fit()

print(nb2_training_results_pooled.summary())

#4) STEP 4: Let’s make test predictions using our trained NB2 model.

nb2_predictions_pooled = nb2_training_results_pooled.get_prediction(X_test_datacomplete_c)

predictions_pooled_summary_frame = nb2_predictions_pooled.summary_frame()

print(predictions_pooled_summary_frame)

#5) Me: computing Test R2

r2_test_NB_reg_pooled = r2_score(y_test_datacomplete, predictions_pooled_summary_frame['mean'])

#0.1767097266908647

###################################################################
##CHECKING QUOTA OF NEGATIVE PREDICTIONS VIA OLS ON ENTIRE POOLED##
###################################################################

#Given the comparable values of the R2 produced by the three regressions 
#(Linear, Poisson, and Negative Binomial), it is interesting to observe
#how many negative values is the first incorrectly producing!

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

R2_test_linreg_nonegs_pooled = linreg_pooled[4]

#0.1805694876613707

R2_Train_linreg_pooled = linreg_pooled[3]

#0.17851863952923697 

test_pred_neg_pctg_pooled = (np.sum(linreg_pooled[-2] < 0) / len(X_test_datacomplete))*100

#2.1110073957061823

train_pred_neg_pctg_pooled = (np.sum(linreg_pooled[-1] < 0) / len(X_train_datacomplete))*100

#2.192410334964876

#Excellent news, validating OLS procedure.

#Even more in detail, I can check "how negative" are these negative:
#are we talking about massive blunders (e.g., -12) or simple -0.001?

#We look at the average values of these negative predictions.

test_pred_neg_pctg_pooled_df = pd.DataFrame(linreg_pooled[-2])

test_pred_neg_pctg_pooled_df_1 = test_pred_neg_pctg_pooled_df[(test_pred_neg_pctg_pooled_df < 0).all(1)]

test_pred_neg_pctg_pooled_df_1.mean()

#-0.241691, basically all 0s. If I were to round them, they would be 0s.

train_pred_neg_pctg_pooled_df = pd.DataFrame(linreg_pooled[-1])

train_pred_neg_pctg_pooled_df_1 = train_pred_neg_pctg_pooled_df[(train_pred_neg_pctg_pooled_df < 0).all(1)]

train_pred_neg_pctg_pooled_df_1.mean()

#-0.241678

#All the above operations are now repeated also on the entire Transformed Pooled
#dataset, as well as on the 5 clusters from the entire Pooled and the 3 from
#the entire Transformed Pooled. Since the operations are the same, to ease up
#the script, comments are not repeated.

#################################
##PR ON TRANSFORMED POOLED DATA##
#################################

X_train_datacomplete_MD_stand_1.drop(['pid','hid', 'Unnamed: 0'], axis = 1, inplace = True)

X_test_datacomplete_MD_stand_1.drop(['pid','hid', 'Unnamed: 0'], axis = 1, inplace = True)

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


y_train_mundlak = y_train_datacomplete_PD_y['Doctor Visits']

y_test_mundlak = y_test_datacomplete_PD_y['Doctor Visits']

PR_reg_mundlak = PR_reg.fit(X_train_datacomplete_MD_stand_1, y_train_mundlak)

pred_test_PR_reg_mundlak = PR_reg_mundlak.predict(X_test_datacomplete_MD_stand_1)

#let's check both the deviance and the R2.

r2_test_PR_reg_mundlak = r2_score(y_test_mundlak, pred_test_PR_reg_mundlak)

#0.18657211625671954

r2_test_PR_reg_mundlak_1 = pearsonr(y_test_mundlak, pred_test_PR_reg_mundlak)[0]**2

#0.18658521669907077

#######################################################
##NEGATIVE BINOMIAL ON ENTIRE TRANSFORMED POOLED DATA##
#######################################################

X_train_datacomplete_MD_stand_1_c = sm.add_constant(X_train_datacomplete_MD_stand_1)

X_test_datacomplete_MD_stand_1_c = sm.add_constant(X_test_datacomplete_MD_stand_1)

poisson_training_results_mundlak = sm.GLM(y_train_mundlak, 
                                          X_train_datacomplete_MD_stand_1_c, 
                                          family = sm.families.Poisson()).fit()

print(poisson_training_results_mundlak.summary())

BB_LAMBDA_mundlak = poisson_training_results_mundlak.mu

AUX_OLS_DEP_mundlak = ((y_train_mundlak - BB_LAMBDA_mundlak)**2 - y_train_mundlak) / BB_LAMBDA_mundlak

aux_reg_mundlak = sm.OLS(AUX_OLS_DEP_mundlak, BB_LAMBDA_mundlak).fit()

print(aux_reg_mundlak.summary())

#Indeed signficant at 0.001

nb2_training_results_mundlak = sm.GLM(y_train_mundlak, X_train_datacomplete_MD_stand_1_c,
                                     family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_mundlak.summary())

nb2_predictions_mundlak = nb2_training_results_mundlak.get_prediction(X_test_datacomplete_MD_stand_1_c)

predictions_mundlak_summary_frame = nb2_predictions_mundlak.summary_frame()

print(predictions_mundlak_summary_frame)

r2_test_NB_reg_mundlak = r2_score(y_test_mundlak, predictions_mundlak_summary_frame['mean'])

#0.18205858797138508

###############################################################################
##CHECKING QUOTA OF NEGATIVE PREDICTIONS VIA OLS ON ENTIRE TRANSFORMED POOLED##
###############################################################################

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

R2_test_linreg_nonegs_mundlak_all = linreg_mundlak_all[4]

#0.1791376699459859

R2_Train_linreg_mundlak_all = linreg_mundlak_all[3]

#0.18081576007042388

test_pred_neg_pctg_mundlak = (np.sum(linreg_mundlak_all[-2] < 0) / len(X_test_datacomplete_MD_stand_1))*100

#2.2282855843565255

train_pred_neg_pctg_mundlak = (np.sum(linreg_mundlak_all[-1] < 0) / len(X_train_datacomplete_MD_stand_1))*100

#2.2869520470075755

test_pred_neg_pctg_mundlak_df = pd.DataFrame(linreg_mundlak_all[-2])

test_pred_neg_pctg_mundlak_df_1 = test_pred_neg_pctg_mundlak_df[(test_pred_neg_pctg_mundlak_df < 0).all(1)]

test_pred_neg_pctg_mundlak_df_1.mean()

#-0.299988

train_pred_neg_pctg_mundlak_df = pd.DataFrame(linreg_mundlak_all[-1])

train_pred_neg_pctg_mundlak_df_1 = train_pred_neg_pctg_mundlak_df[(train_pred_neg_pctg_mundlak_df < 0).all(1)]

train_pred_neg_pctg_mundlak_df_1.mean()

#-0.260114

#################################
##PR ON CLUSTERS ON POOLED DATA##
#################################

######################
##CLUSTER 0 - POOLED##
######################

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

PR_reg_cluster_0 = PR_reg.fit(X_Train_cluster_0, Y_Train_cluster_0)

pred_test_PR_reg_cluster_0 = PR_reg_cluster_0.predict(X_Test_cluster_0)

r2_test_PR_reg_cluster_0 = r2_score(Y_Test_cluster_0, pred_test_PR_reg_cluster_0)

linreg_cluster_0 = linreg_train_test(X_train = X_Train_cluster_0, 
                                     y_train = Y_Train_cluster_0, 
                                     X_test = X_Test_cluster_0,
                                     y_test = Y_Test_cluster_0) 

MSE_Test_linreg_cluster_0 = linreg_cluster_0[0]

#12.68911623701252

MSE_Train_linreg_cluster_0 = linreg_cluster_0[1]

#12.315493969607301

R2_Test_linreg_cluster_0 = linreg_cluster_0[2]

#0.16113538587722898

R2_test_linreg_nonegs_cluster_0 = linreg_cluster_0[4]

#0.16299837069804768

R2_Train_linreg_cluster_0 = linreg_cluster_0[3]

#0.15825560915978898

test_pred_neg_pctg_cluster_0 = (np.sum(linreg_cluster_0[-2] < 0) / len(X_Test_cluster_0))*100

#4.411764705882353

train_pred_neg_pctg_cluster_0 = (np.sum(linreg_cluster_0[-1] < 0) / len(X_Train_cluster_0))*100

#4.226682408500591

test_pred_neg_pctg_cluster_0_df = pd.DataFrame(linreg_cluster_0[-2])

test_pred_neg_pctg_cluster_0_df_1 = test_pred_neg_pctg_cluster_0_df[(test_pred_neg_pctg_cluster_0_df < 0).all(1)]

test_pred_neg_pctg_cluster_0_df_1.mean()

#-0.411113

train_pred_neg_pctg_cluster_0_df = pd.DataFrame(linreg_cluster_0[-1])

train_pred_neg_pctg_cluster_0_df_1 = train_pred_neg_pctg_cluster_0_df[(train_pred_neg_pctg_cluster_0_df < 0).all(1)]

train_pred_neg_pctg_cluster_0_df_1.mean()

#-0.41467

#Negative Binomial

X_Train_cluster_0_c = sm.add_constant(X_Train_cluster_0)

X_Test_cluster_0_c = sm.add_constant(X_Test_cluster_0)

poisson_training_results_cluster_0 = sm.GLM(Y_Train_cluster_0, 
                                            X_Train_cluster_0_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_0.summary())

BB_LAMBDA_cluster_0 = poisson_training_results_cluster_0.mu

AUX_OLS_DEP_cluster_0 = ((Y_Train_cluster_0 - BB_LAMBDA_cluster_0)**2 - Y_Train_cluster_0) / BB_LAMBDA_cluster_0

aux_reg_cluster_0 = sm.OLS(AUX_OLS_DEP_cluster_0, BB_LAMBDA_cluster_0).fit()

print(aux_reg_cluster_0.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_0 = sm.GLM(Y_Train_cluster_0, 
                                        X_Train_cluster_0_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_0.summary())

nb2_predictions_cluster_0 = nb2_training_results_cluster_0.get_prediction(X_Test_cluster_0_c)

predictions_cluster_0_summary_frame = nb2_predictions_cluster_0.summary_frame()

print(predictions_cluster_0_summary_frame)

r2_test_NB_reg_cluster_0 = r2_score(Y_Test_cluster_0, predictions_cluster_0_summary_frame['mean'])

#0.19270709236449568 

######################
##CLUSTER 1 - POOLED##
######################

const_in_test_1 = []

for i in list(Test_cluster_1):
        
    if Test_cluster_1[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_1.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1.drop(i, axis = 1, inplace = True)
        
len(const_in_test_1)

const_in_train_1 = []

for i in list(Train_cluster_1):
        
    if Train_cluster_1[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_1.drop(i, axis = 1, inplace = True)
            
        Test_cluster_1.drop(i, axis = 1, inplace = True)
        
len(const_in_train_1)

Y_Test_cluster_1 = Test_cluster_1['dvisit']

X_Test_cluster_1 = Test_cluster_1.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_1 = Train_cluster_1['dvisit']

X_Train_cluster_1 = Train_cluster_1.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

PR_reg_cluster_1 = PR_reg.fit(X_Train_cluster_1, Y_Train_cluster_1)

pred_test_PR_reg_cluster_1 = PR_reg_cluster_1.predict(X_Test_cluster_1)

r2_test_PR_reg_cluster_1 = r2_score(Y_Test_cluster_1, pred_test_PR_reg_cluster_1)

#0.13694216523246028

linreg_cluster_1 = linreg_train_test(X_train = X_Train_cluster_1, 
                                     y_train = Y_Train_cluster_1, 
                                     X_test = X_Test_cluster_1,
                                     y_test = Y_Test_cluster_1) 

MSE_Test_linreg_cluster_1 = linreg_cluster_1[0]

#20.41105260248074

MSE_Train_linreg_cluster_1 = linreg_cluster_1[1]

#20.219402437612256

R2_Test_linreg_cluster_1 = linreg_cluster_1[2]

#0.13930591770495704

R2_test_linreg_nonegs_cluster_1 = linreg_cluster_1[4]

#0.1395175034859727

R2_Train_linreg_cluster_1 = linreg_cluster_1[3]

#0.14995308758157944

test_pred_neg_pctg_cluster_1 = (np.sum(linreg_cluster_1[-2] < 0) / len(X_Test_cluster_1))*100

#0.5308880308880308

train_pred_neg_pctg_cluster_1 = (np.sum(linreg_cluster_1[-1] < 0) / len(X_Train_cluster_1))*100

#0.6153623398248027

test_pred_neg_pctg_cluster_1_df = pd.DataFrame(linreg_cluster_1[-2])

test_pred_neg_pctg_cluster_1_df_1 = test_pred_neg_pctg_cluster_1_df[(test_pred_neg_pctg_cluster_1_df < 0).all(1)]

test_pred_neg_pctg_cluster_1_df_1.mean()

#-0.336539

train_pred_neg_pctg_cluster_1_df = pd.DataFrame(linreg_cluster_1[-1])

train_pred_neg_pctg_cluster_1_df_1 = train_pred_neg_pctg_cluster_1_df[(train_pred_neg_pctg_cluster_1_df < 0).all(1)]

train_pred_neg_pctg_cluster_1_df_1.mean()

#-0.329198

#Negative Binomial

X_Train_cluster_1_c = sm.add_constant(X_Train_cluster_1)

X_Test_cluster_1_c = sm.add_constant(X_Test_cluster_1)

poisson_training_results_cluster_1 = sm.GLM(Y_Train_cluster_1, 
                                            X_Train_cluster_1_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_1.summary())

BB_LAMBDA_cluster_1 = poisson_training_results_cluster_1.mu

AUX_OLS_DEP_cluster_1 = ((Y_Train_cluster_1 - BB_LAMBDA_cluster_1)**2 - Y_Train_cluster_1) / BB_LAMBDA_cluster_1

aux_reg_cluster_1 = sm.OLS(AUX_OLS_DEP_cluster_1, BB_LAMBDA_cluster_1).fit()

print(aux_reg_cluster_1.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_1 = sm.GLM(Y_Train_cluster_1, 
                                        X_Train_cluster_1_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_1.summary())

nb2_predictions_cluster_1 = nb2_training_results_cluster_1.get_prediction(X_Test_cluster_1_c)

predictions_cluster_1_summary_frame = nb2_predictions_cluster_1.summary_frame()

print(predictions_cluster_1_summary_frame)

r2_test_NB_reg_cluster_1 = r2_score(Y_Test_cluster_1, predictions_cluster_1_summary_frame['mean'])

#0.13191369495609384

######################
##CLUSTER 2 - POOLED##
######################

const_in_test_2 = []

for i in list(Test_cluster_2):
        
    if Test_cluster_2[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_2.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2.drop(i, axis = 1, inplace = True)
        
len(const_in_test_2)

const_in_train_2 = []

for i in list(Train_cluster_2):
        
    if Train_cluster_2[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_2.drop(i, axis = 1, inplace = True)
            
        Test_cluster_2.drop(i, axis = 1, inplace = True)
        
len(const_in_train_2)

Y_Test_cluster_2 = Test_cluster_2['dvisit']

X_Test_cluster_2 = Test_cluster_2.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

Y_Train_cluster_2 = Train_cluster_2['dvisit']

X_Train_cluster_2 = Train_cluster_2.drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE'], axis = 1)

PR_reg_cluster_2 = PR_reg.fit(X_Train_cluster_2, Y_Train_cluster_2)

pred_test_PR_reg_cluster_2 = PR_reg_cluster_2.predict(X_Test_cluster_2)

r2_test_PR_reg_cluster_2 = r2_score(Y_Test_cluster_2, pred_test_PR_reg_cluster_2)

#0.15527026000850985

linreg_cluster_2 = linreg_train_test(X_train = X_Train_cluster_2, 
                                     y_train = Y_Train_cluster_2, 
                                     X_test = X_Test_cluster_2,
                                     y_test = Y_Test_cluster_2) 

MSE_Test_linreg_cluster_2 = linreg_cluster_2[0]

#7.706827356199911

MSE_Train_linreg_cluster_2 = linreg_cluster_2[1]

#7.006297383240808

R2_Test_linreg_cluster_2 = linreg_cluster_2[2]

#0.1399121136832342

R2_test_linreg_nonegs_cluster_2 = linreg_cluster_2[4]

#0.14035210448853364

R2_Train_linreg_cluster_2 = linreg_cluster_2[3]

#0.14513052836095885

test_pred_neg_pctg_cluster_2 = (np.sum(linreg_cluster_2[-2] < 0) / len(X_Test_cluster_2))*100

#2.1679046467438243

train_pred_neg_pctg_cluster_2 = (np.sum(linreg_cluster_2[-1] < 0) / len(X_Train_cluster_2))*100

#2.163772998186059

test_pred_neg_pctg_cluster_2_df = pd.DataFrame(linreg_cluster_2[-2])

test_pred_neg_pctg_cluster_2_df_1 = test_pred_neg_pctg_cluster_2_df[(test_pred_neg_pctg_cluster_2_df < 0).all(1)]

test_pred_neg_pctg_cluster_2_df_1.mean()

#-0.125863

train_pred_neg_pctg_cluster_2_df = pd.DataFrame(linreg_cluster_2[-1])

train_pred_neg_pctg_cluster_2_df_1 = train_pred_neg_pctg_cluster_2_df[(train_pred_neg_pctg_cluster_2_df < 0).all(1)]

train_pred_neg_pctg_cluster_2_df_1.mean()

#-0.135986

#Negative Binomial

X_Train_cluster_2_c = sm.add_constant(X_Train_cluster_2)

X_Test_cluster_2_c = sm.add_constant(X_Test_cluster_2)

poisson_training_results_cluster_2 = sm.GLM(Y_Train_cluster_2, 
                                            X_Train_cluster_2_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_2.summary())

BB_LAMBDA_cluster_2 = poisson_training_results_cluster_2.mu

AUX_OLS_DEP_cluster_2 = ((Y_Train_cluster_2 - BB_LAMBDA_cluster_2)**2 - Y_Train_cluster_2) / BB_LAMBDA_cluster_2

aux_reg_cluster_2 = sm.OLS(AUX_OLS_DEP_cluster_2, BB_LAMBDA_cluster_2).fit()

print(aux_reg_cluster_2.summary())

#Indeed significant at 0.001

nb2_training_results_cluster_2 = sm.GLM(Y_Train_cluster_2, 
                                        X_Train_cluster_2_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_2.summary())

nb2_predictions_cluster_2 = nb2_training_results_cluster_2.get_prediction(X_Test_cluster_2_c)

predictions_cluster_2_summary_frame = nb2_predictions_cluster_2.summary_frame()

print(predictions_cluster_2_summary_frame)

r2_test_NB_reg_cluster_2 = r2_score(Y_Test_cluster_2, predictions_cluster_2_summary_frame['mean'])

#0.15248861197333363

######################
##CLUSTER 3 - POOLED##
######################

const_in_test_3 = []

for i in list(Test_cluster_3):
        
    if Test_cluster_3[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_3.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3.drop(i, axis = 1, inplace = True)
        
len(const_in_test_3)

const_in_train_3 = []

for i in list(Train_cluster_3):
        
    if Train_cluster_3[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_3.drop(i, axis = 1, inplace = True)
            
        Test_cluster_3.drop(i, axis = 1, inplace = True)
        
len(const_in_train_3)

Y_Test_cluster_3 = Test_cluster_3['dvisit']

X_Test_cluster_3 = Test_cluster_3.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_3 = Train_cluster_3['dvisit']

X_Train_cluster_3 = Train_cluster_3.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

PR_reg_cluster_3 = PR_reg.fit(X_Train_cluster_3, Y_Train_cluster_3)

pred_test_PR_reg_cluster_3 = PR_reg_cluster_3.predict(X_Test_cluster_3)

r2_test_PR_reg_cluster_3 = r2_score(Y_Test_cluster_3, pred_test_PR_reg_cluster_3)

#0.13492775587848294

linreg_cluster_3 = linreg_train_test(X_train = X_Train_cluster_3, 
                                     y_train = Y_Train_cluster_3, 
                                     X_test = X_Test_cluster_3,
                                     y_test = Y_Test_cluster_3) 

MSE_Test_linreg_cluster_3 = linreg_cluster_3[0]

#10.935079336216534

MSE_Train_linreg_cluster_3 = linreg_cluster_3[1]

#9.401358602683965

R2_Test_linreg_cluster_3 = linreg_cluster_3[2]

#0.12365066461593832

R2_test_linreg_nonegs_cluster_3 = linreg_cluster_3[4]

#0.12388259709678817

R2_Train_linreg_cluster_3 = linreg_cluster_3[3]

#0.1417260948132404

test_pred_neg_pctg_cluster_3 = (np.sum(linreg_cluster_3[-2] < 0) / len(X_Test_cluster_3))*100

#0.9787444389520514

train_pred_neg_pctg_cluster_3 = (np.sum(linreg_cluster_3[-1] < 0) / len(X_Train_cluster_3))*100

#0.8453210737060655

test_pred_neg_pctg_cluster_3_df = pd.DataFrame(linreg_cluster_3[-2])

test_pred_neg_pctg_cluster_3_df_1 = test_pred_neg_pctg_cluster_3_df[(test_pred_neg_pctg_cluster_3_df < 0).all(1)]

test_pred_neg_pctg_cluster_3_df_1.mean()

#-0.201113

train_pred_neg_pctg_cluster_3_df = pd.DataFrame(linreg_cluster_3[-1])

train_pred_neg_pctg_cluster_3_df_1 = train_pred_neg_pctg_cluster_3_df[(train_pred_neg_pctg_cluster_3_df < 0).all(1)]

train_pred_neg_pctg_cluster_3_df_1.mean()

#-0.189725

#Negative Binomial

X_Train_cluster_3_c = sm.add_constant(X_Train_cluster_3)

X_Test_cluster_3_c = sm.add_constant(X_Test_cluster_3)

poisson_training_results_cluster_3 = sm.GLM(Y_Train_cluster_3, 
                                            X_Train_cluster_3_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_3.summary())

BB_LAMBDA_cluster_3 = poisson_training_results_cluster_3.mu

AUX_OLS_DEP_cluster_3 = ((Y_Train_cluster_3 - BB_LAMBDA_cluster_3)**2 - Y_Train_cluster_3) / BB_LAMBDA_cluster_3

aux_reg_cluster_3 = sm.OLS(AUX_OLS_DEP_cluster_3, BB_LAMBDA_cluster_3).fit()

print(aux_reg_cluster_3.summary())

#Indeed significant at 0.001

nb2_training_results_cluster_3 = sm.GLM(Y_Train_cluster_3, 
                                        X_Train_cluster_3_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_3.summary())

nb2_predictions_cluster_3 = nb2_training_results_cluster_3.get_prediction(X_Test_cluster_3_c)

predictions_cluster_3_summary_frame = nb2_predictions_cluster_3.summary_frame()

print(predictions_cluster_3_summary_frame)

r2_test_NB_reg_cluster_3 = r2_score(Y_Test_cluster_3, predictions_cluster_3_summary_frame['mean'])

#0.1352949887567373

######################
##CLUSTER 4 - POOLED##
######################

const_in_test_4 = []

for i in list(Test_cluster_4):
        
    if Test_cluster_4[i].nunique() == 1:
        
        const_in_test_0.append(i)
            
        Train_cluster_4.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4.drop(i, axis = 1, inplace = True)
        
len(const_in_test_4)

const_in_train_4 = []

for i in list(Train_cluster_4):
        
    if Train_cluster_4[i].nunique() == 1:
        
        const_in_train_0.append(i)
            
        Train_cluster_4.drop(i, axis = 1, inplace = True)
            
        Test_cluster_4.drop(i, axis = 1, inplace = True)
        
len(const_in_train_4)

Y_Test_cluster_4 = Test_cluster_4['dvisit']

X_Test_cluster_4 = Test_cluster_4.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

Y_Train_cluster_4 = Train_cluster_4['dvisit']

X_Train_cluster_4 = Train_cluster_4.drop(['syear', 'pid', 'hid', 'dvisit', 'SE'], axis = 1)

PR_reg_cluster_4 = PR_reg.fit(X_Train_cluster_4, Y_Train_cluster_4)

pred_test_PR_reg_cluster_4 = PR_reg_cluster_4.predict(X_Test_cluster_4)

r2_test_PR_reg_cluster_4 = r2_score(Y_Test_cluster_4, pred_test_PR_reg_cluster_4)

#0.13416308310199687

linreg_cluster_4 = linreg_train_test(X_train = X_Train_cluster_4, 
                                     y_train = Y_Train_cluster_4, 
                                     X_test = X_Test_cluster_4,
                                     y_test = Y_Test_cluster_4) 

MSE_Test_linreg_cluster_4 = linreg_cluster_4[0]

#5.414851068589415

MSE_Train_linreg_cluster_4 = linreg_cluster_4[1]

#6.606466269499825

R2_Test_linreg_cluster_4 = linreg_cluster_4[2]

#0.13328573277030964

R2_test_linreg_nonegs_cluster_4 = linreg_cluster_4[4]

#0.1333275695411985

R2_Train_linreg_cluster_4 = linreg_cluster_4[3]

#0.13432361191228148

test_pred_neg_pctg_cluster_4 = (np.sum(linreg_cluster_4[-2] < 0) / len(X_Test_cluster_4))*100

#0.4745470232959448

train_pred_neg_pctg_cluster_4 = (np.sum(linreg_cluster_4[-1] < 0) / len(X_Train_cluster_4))*100

#0.6040992448759439

test_pred_neg_pctg_cluster_4_df = pd.DataFrame(linreg_cluster_4[-2])

test_pred_neg_pctg_cluster_4_df_1 = test_pred_neg_pctg_cluster_4_df[(test_pred_neg_pctg_cluster_4_df < 0).all(1)]

test_pred_neg_pctg_cluster_4_df_1.mean()

#-0.09445

train_pred_neg_pctg_cluster_4_df = pd.DataFrame(linreg_cluster_4[-1])

train_pred_neg_pctg_cluster_4_df_1 = train_pred_neg_pctg_cluster_4_df[(train_pred_neg_pctg_cluster_4_df < 0).all(1)]

train_pred_neg_pctg_cluster_4_df_1.mean()

#-0.118343

#Negative Binomial

X_Train_cluster_4_c = sm.add_constant(X_Train_cluster_4)

X_Test_cluster_4_c = sm.add_constant(X_Test_cluster_4)

poisson_training_results_cluster_4 = sm.GLM(Y_Train_cluster_4, 
                                            X_Train_cluster_4_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_4.summary())

BB_LAMBDA_cluster_4 = poisson_training_results_cluster_4.mu

AUX_OLS_DEP_cluster_4 = ((Y_Train_cluster_4 - BB_LAMBDA_cluster_4)**2 - Y_Train_cluster_4) / BB_LAMBDA_cluster_4

aux_reg_cluster_4 = sm.OLS(AUX_OLS_DEP_cluster_4, BB_LAMBDA_cluster_4).fit()

print(aux_reg_cluster_4.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_4 = sm.GLM(Y_Train_cluster_4, 
                                        X_Train_cluster_4_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_4.summary())

nb2_predictions_cluster_4 = nb2_training_results_cluster_4.get_prediction(X_Test_cluster_4_c)

predictions_cluster_4_summary_frame = nb2_predictions_cluster_4.summary_frame()

print(predictions_cluster_4_summary_frame)

r2_test_NB_reg_cluster_4 = r2_score(Y_Test_cluster_4, predictions_cluster_4_summary_frame['mean'])

#0.13515078262725844

#############################################
##PR ON CLUSTERS ON TRANSFORMED POOLED DATA##
#############################################

##################################
##CLUSTER 0 - TRANSFORMED POOLED##
##################################


y_Train_cluster_MD_0 = Train_cluster_MD_0['Doctor Visits']

X_Train_cluster_MD_0 = Train_cluster_MD_0.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                                'cluster.1','cluster.2', 
                                                'Group Mean Doctor Visits', 
                                                'Group-Demeaned Doctor Visits'], axis = 1)

y_Test_cluster_MD_0 = Test_cluster_MD_0['Doctor Visits']

X_Test_cluster_MD_0 = Test_cluster_MD_0.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                              'cluster.1','cluster.2', 
                                              'Group Mean Doctor Visits', 
                                              'Group-Demeaned Doctor Visits'], axis = 1)

PR_reg_cluster_MD_0 = PR_reg.fit(X_Train_cluster_MD_0, y_Train_cluster_MD_0)

pred_test_PR_reg_cluster_MD_0 = PR_reg_cluster_MD_0.predict(X_Test_cluster_MD_0)

r2_test_PR_reg_cluster_MD_0 = r2_score(y_Test_cluster_MD_0, pred_test_PR_reg_cluster_MD_0)

#0.13088725289166925

linreg_cluster_MD_0 = linreg_train_test(X_train = X_Train_cluster_MD_0, 
                                        y_train = y_Train_cluster_MD_0, 
                                        X_test = X_Test_cluster_MD_0,
                                        y_test = y_Test_cluster_MD_0) 

MSE_Test_linreg_cluster_MD_0 = linreg_cluster_MD_0[0]

#25.50072773633491

MSE_Train_linreg_cluster_MD_0 = linreg_cluster_MD_0[1]

#23.006691250367815

R2_Test_linreg_cluster_MD_0 = linreg_cluster_MD_0[2]

#0.1278113307666746

R2_test_linreg_nonegs_cluster_MD_0 = linreg_cluster_MD_0[4]

#0.12807109950665718

R2_Train_linreg_cluster_MD_0 = linreg_cluster_MD_0[3]

#.1279742395218958

test_pred_neg_pctg_cluster_MD_0 = (np.sum(linreg_cluster_MD_0[-2] < 0) / len(X_Test_cluster_MD_0))*100

#0.4155007207665564

train_pred_neg_pctg_cluster_MD_0 = (np.sum(linreg_cluster_MD_0[-1] < 0) / len(X_Train_cluster_MD_0))*100

#0.30526583566522514

test_pred_neg_pctg_cluster_MD_0_df = pd.DataFrame(linreg_cluster_MD_0[-2])

test_pred_neg_pctg_cluster_MD_0_df_1 = test_pred_neg_pctg_cluster_MD_0_df[(test_pred_neg_pctg_cluster_MD_0_df < 0).all(1)]

test_pred_neg_pctg_cluster_MD_0_df_1.mean()

#-0.541888

train_pred_neg_pctg_cluster_MD_0_df = pd.DataFrame(linreg_cluster_MD_0[-1])

train_pred_neg_pctg_cluster_MD_0_df_1 = train_pred_neg_pctg_cluster_MD_0_df[(train_pred_neg_pctg_cluster_MD_0_df < 0).all(1)]

train_pred_neg_pctg_cluster_MD_0_df_1.mean()

#-0.452765

#Negative Binomial

X_Train_cluster_MD_0_c = sm.add_constant(X_Train_cluster_MD_0)

X_Test_cluster_MD_0_c = sm.add_constant(X_Test_cluster_MD_0)

poisson_training_results_cluster_MD_0 = sm.GLM(y_Train_cluster_MD_0, 
                                            X_Train_cluster_MD_0_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_MD_0.summary())

BB_LAMBDA_cluster_MD_0 = poisson_training_results_cluster_MD_0.mu

AUX_OLS_DEP_cluster_MD_0 = ((y_Train_cluster_MD_0 - BB_LAMBDA_cluster_MD_0)**2 - y_Train_cluster_MD_0) / BB_LAMBDA_cluster_MD_0

aux_reg_cluster_MD_0 = sm.OLS(AUX_OLS_DEP_cluster_MD_0, BB_LAMBDA_cluster_MD_0).fit()

print(aux_reg_cluster_MD_0.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_MD_0 = sm.GLM(y_Train_cluster_MD_0, 
                                        X_Train_cluster_MD_0_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_MD_0.summary())

nb2_predictions_cluster_MD_0 = nb2_training_results_cluster_MD_0.get_prediction(X_Test_cluster_MD_0_c)

predictions_cluster_MD_0_summary_frame = nb2_predictions_cluster_MD_0.summary_frame()

print(predictions_cluster_MD_0_summary_frame)

r2_test_NB_reg_cluster_MD_0 = r2_score(y_Test_cluster_MD_0, predictions_cluster_MD_0_summary_frame['mean'])

#0.1273788995662678

##################################
##CLUSTER 1 - TRANSFORMED POOLED##
##################################

y_Train_cluster_MD_1 = Train_cluster_MD_1['Doctor Visits']

X_Train_cluster_MD_1 = Train_cluster_MD_1.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                                'cluster.1','cluster.2', 
                                                'Group Mean Doctor Visits', 
                                                'Group-Demeaned Doctor Visits'], axis = 1)

y_Test_cluster_MD_1 = Test_cluster_MD_1['Doctor Visits']

X_Test_cluster_MD_1 = Test_cluster_MD_1.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                              'cluster.1','cluster.2', 
                                              'Group Mean Doctor Visits', 
                                              'Group-Demeaned Doctor Visits'], axis = 1)

PR_reg_cluster_MD_1 = PR_reg.fit(X_Train_cluster_MD_1, y_Train_cluster_MD_1)

pred_test_PR_reg_cluster_MD_1 = PR_reg_cluster_MD_1.predict(X_Test_cluster_MD_1)

r2_test_PR_reg_cluster_MD_1 = r2_score(y_Test_cluster_MD_1, pred_test_PR_reg_cluster_MD_1)

#0.14120083121858618

linreg_cluster_MD_1 = linreg_train_test(X_train = X_Train_cluster_MD_1, 
                                     y_train = y_Train_cluster_MD_1, 
                                     X_test = X_Test_cluster_MD_1,
                                     y_test = y_Test_cluster_MD_1) 

MSE_Test_linreg_cluster_MD_1 = linreg_cluster_MD_1[0]

#6.278332449477104

MSE_Train_linreg_cluster_MD_1 = linreg_cluster_MD_1[1]

#7.052784934793279

R2_Test_linreg_cluster_MD_1 = linreg_cluster_MD_1[2]

#0.13030754228385877

R2_test_linreg_nonegs_cluster_MD_1 = linreg_cluster_MD_1[4]

#0.13110036577189177

R2_Train_linreg_cluster_MD_1 = linreg_cluster_MD_1[3]

#0.1272258049290831

test_pred_neg_pctg_cluster_MD_1 = (np.sum(linreg_cluster_MD_1[-2] < 0) / len(X_Test_cluster_MD_1))*100

#2.2413263739637705

train_pred_neg_pctg_cluster_MD_1 = (np.sum(linreg_cluster_MD_1[-1] < 0) / len(X_Train_cluster_MD_1))*100

#2.2122942765905185

test_pred_neg_pctg_cluster_MD_1_df = pd.DataFrame(linreg_cluster_MD_1[-2])

test_pred_neg_pctg_cluster_MD_1_df_1 = test_pred_neg_pctg_cluster_MD_1_df[(test_pred_neg_pctg_cluster_MD_1_df < 0).all(1)]

test_pred_neg_pctg_cluster_MD_1_df_1.mean()

#-0.192502

train_pred_neg_pctg_cluster_MD_1_df = pd.DataFrame(linreg_cluster_MD_1[-1])

train_pred_neg_pctg_cluster_MD_1_df_1 = train_pred_neg_pctg_cluster_MD_1_df[(train_pred_neg_pctg_cluster_MD_1_df < 0).all(1)]

train_pred_neg_pctg_cluster_MD_1_df_1.mean()

#-0.192738

#Negative Binomial

X_Train_cluster_MD_1_c = sm.add_constant(X_Train_cluster_MD_1)

X_Test_cluster_MD_1_c = sm.add_constant(X_Test_cluster_MD_1)

poisson_training_results_cluster_MD_1 = sm.GLM(y_Train_cluster_MD_1, 
                                            X_Train_cluster_MD_1_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_MD_1.summary())

BB_LAMBDA_cluster_MD_1 = poisson_training_results_cluster_MD_1.mu

AUX_OLS_DEP_cluster_MD_1 = ((y_Train_cluster_MD_1 - BB_LAMBDA_cluster_MD_1)**2 - y_Train_cluster_MD_1) / BB_LAMBDA_cluster_MD_1

aux_reg_cluster_MD_1 = sm.OLS(AUX_OLS_DEP_cluster_MD_1, BB_LAMBDA_cluster_MD_1).fit()

print(aux_reg_cluster_MD_1.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_MD_1 = sm.GLM(y_Train_cluster_MD_1, 
                                        X_Train_cluster_MD_1_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_MD_1.summary())

nb2_predictions_cluster_MD_1 = nb2_training_results_cluster_MD_1.get_prediction(X_Test_cluster_MD_1_c)

predictions_cluster_MD_1_summary_frame = nb2_predictions_cluster_MD_1.summary_frame()

print(predictions_cluster_MD_1_summary_frame)

r2_test_NB_reg_cluster_MD_1 = r2_score(y_Test_cluster_MD_1, predictions_cluster_MD_1_summary_frame['mean'])

#0.1395373566278253

##################################
##CLUSTER 2 - TRANSFORMED POOLED##
##################################

y_Train_cluster_MD_2 = Train_cluster_MD_2['Doctor Visits']

X_Train_cluster_MD_2 = Train_cluster_MD_2.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                                'cluster.1','cluster.2', 
                                                'Group Mean Doctor Visits', 
                                                'Group-Demeaned Doctor Visits'], axis = 1)

y_Test_cluster_MD_2 = Test_cluster_MD_2['Doctor Visits']

X_Test_cluster_MD_2 = Test_cluster_MD_2.drop(['Doctor Visits', 'pid', 'hid', 'cluster', 'pid.1',
                                              'cluster.1','cluster.2', 
                                              'Group Mean Doctor Visits', 
                                              'Group-Demeaned Doctor Visits'], axis = 1)

PR_reg_cluster_MD_2 = PR_reg.fit(X_Train_cluster_MD_2, y_Train_cluster_MD_2)

pred_test_PR_reg_cluster_MD_2 = PR_reg_cluster_MD_2.predict(X_Test_cluster_MD_2)

r2_test_PR_reg_cluster_MD_2 = r2_score(y_Test_cluster_MD_2, pred_test_PR_reg_cluster_MD_2)

#0.11687762982323613

linreg_cluster_MD_2 = linreg_train_test(X_train = X_Train_cluster_MD_2, 
                                     y_train = y_Train_cluster_MD_2, 
                                     X_test = X_Test_cluster_MD_2,
                                     y_test = y_Test_cluster_MD_2) 

MSE_Test_linreg_cluster_MD_2 = linreg_cluster_MD_2[0]

#7.862936559425069

MSE_Train_linreg_cluster_MD_2 = linreg_cluster_MD_2[1]

#7.665988170639644

R2_Test_linreg_cluster_MD_2 = linreg_cluster_MD_2[2]

#0.10620819795470848

R2_test_linreg_nonegs_cluster_MD_2 = linreg_cluster_MD_2[4]

#0.10663711776593243

R2_Train_linreg_cluster_MD_2 = linreg_cluster_MD_2[3]

#0.11419055246686816

test_pred_neg_pctg_cluster_MD_2 = (np.sum(linreg_cluster_MD_2[-2] < 0) / len(X_Test_cluster_MD_2))*100

#1.2697022767075306

train_pred_neg_pctg_cluster_MD_2 = (np.sum(linreg_cluster_MD_2[-1] < 0) / len(X_Train_cluster_MD_2))*100

#1.2205133818619671

test_pred_neg_pctg_cluster_MD_2_df = pd.DataFrame(linreg_cluster_MD_2[-2])

test_pred_neg_pctg_cluster_MD_2_df_1 = test_pred_neg_pctg_cluster_MD_2_df[(test_pred_neg_pctg_cluster_MD_2_df < 0).all(1)]

test_pred_neg_pctg_cluster_MD_2_df_1.mean()

#-0.18843

train_pred_neg_pctg_cluster_MD_2_df = pd.DataFrame(linreg_cluster_MD_2[-1])

train_pred_neg_pctg_cluster_MD_2_df_1 = train_pred_neg_pctg_cluster_MD_2_df[(train_pred_neg_pctg_cluster_MD_2_df < 0).all(1)]

train_pred_neg_pctg_cluster_MD_2_df_1.mean()

#-0.190856

#Negative Binomial

X_Train_cluster_MD_2_c = sm.add_constant(X_Train_cluster_MD_2)

X_Test_cluster_MD_2_c = sm.add_constant(X_Test_cluster_MD_2)

poisson_training_results_cluster_MD_2 = sm.GLM(y_Train_cluster_MD_2, 
                                            X_Train_cluster_MD_2_c, 
                                            family = sm.families.Poisson()).fit()

print(poisson_training_results_cluster_MD_2.summary())

BB_LAMBDA_cluster_MD_2 = poisson_training_results_cluster_MD_2.mu

AUX_OLS_DEP_cluster_MD_2 = ((y_Train_cluster_MD_2 - BB_LAMBDA_cluster_MD_2)**2 - y_Train_cluster_MD_2) / BB_LAMBDA_cluster_MD_2

aux_reg_cluster_MD_2 = sm.OLS(AUX_OLS_DEP_cluster_MD_2, BB_LAMBDA_cluster_MD_2).fit()

print(aux_reg_cluster_MD_2.summary())

#Indeed signficant at 0.001

nb2_training_results_cluster_MD_2 = sm.GLM(y_Train_cluster_MD_2, 
                                        X_Train_cluster_MD_2_c,
                                        family = sm.families.NegativeBinomial(alpha = aux_reg_mundlak.params[0])).fit()

print(nb2_training_results_cluster_MD_2.summary())

nb2_predictions_cluster_MD_2 = nb2_training_results_cluster_MD_2.get_prediction(X_Test_cluster_MD_2_c)

predictions_cluster_MD_2_summary_frame = nb2_predictions_cluster_MD_2.summary_frame()

print(predictions_cluster_MD_2_summary_frame)

r2_test_NB_reg_cluster_MD_2 = r2_score(y_Test_cluster_MD_2, predictions_cluster_MD_2_summary_frame['mean'])

#0.11526239471291877
