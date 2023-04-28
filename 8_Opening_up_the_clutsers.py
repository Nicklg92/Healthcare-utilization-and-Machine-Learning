############################################
##EIGHTH SCRIPT - OPENING UP THE CLUSTERS##
############################################

import time
import pandas as pd
import numpy as np

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

'''
COMMENTS

This is the eighth script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

In here, I investigate in great details who do we have in each cluster,
building on the descriptive statistics produced in 7_Additional_EDA.py.

How to do that?

1) Obtain mean and sd of each unstandardized variable (say mu and sigma).
2) Unstandardize the individuals in the clusters using
    
    Value unstandardized in cluster = sigma*value in cluster + mu
    
3) Compute the descriptive statistics on these unstandardized clusters.
4) Compute differences between mean and sd of variables in the entire datasets
vs. those in these unstandardized datasets.
    
At this point, eventual differences can be referred to uniquely the
clustering.
'''

####################################################
##IMPORTING ALL DATASETS, BOTH ENTIRE AND CLUSTERS##
####################################################

#To get the unstardized mean and sd of the variables in entire pooled.
#These had already been computed in 7_Additional_EDA.py

entire_pooled_descr = pd.read_csv('C:\\Some\\Local\\Path\\description_all.csv') 

#To get the unstardized mean and sd of the variables in entire transformed pooled.

X_datacomplete_MD = pd.read_csv('C:\\Some\\Local\\Path\\X_datacomplete_MD.csv')

X_datacomplete_MD.drop(["Unnamed: 0"], axis = 1, inplace = True)

X_datacomplete_MD_descr = X_datacomplete_MD.describe().T 

#Importing clusters from pooled and binding them

for i in [0,1,2,3,4]:
    
    Train_name = "train_cluster_" + str(i) 
    
    Test_name = "test_cluster_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_" + str(i)] = pd.read_csv(path_test) 
    
for i in [0,1,2,3,4]:
    
    globals()['Cluster_' + str(i) + '_pooled'] = pd.concat([globals()["Train_cluster_" + str(i)], globals()["Test_cluster_" + str(i)]], axis = 0)
    
    del globals()["Train_cluster_" + str(i)], globals()["Test_cluster_" + str(i)]

#Importing clusters from transformed pooled and binding them
    
for i in [0,1,2]:
    
    Train_name = "train_cluster_mundlaked_" + str(i) 
    
    Test_name = "test_cluster_mundlaked_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_MD_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_MD_" + str(i)] = pd.read_csv(path_test)
    
for i in [0,1,2]:
    
    globals()['Cluster_' + str(i) + '_MD'] = pd.concat([globals()["Train_cluster_MD_" + str(i)], globals()["Test_cluster_MD_" + str(i)]], axis = 0)
    
    del globals()["Train_cluster_MD_" + str(i)], globals()["Test_cluster_MD_" + str(i)]

del Test_name, Train_name, i, path_test, path_train    

#Deleting useless variables in clusters under both specifications

for i in [0,1,2,3,4]:
    
    globals()['Cluster_' + str(i) + '_pooled'].drop(['syear', 'pid', 'hid', 'dvisit', 'Pension', 'SE', 'cluster', 'cluster.1'], axis = 1, inplace = True)
    
for i in [0,1,2]: 
    
    globals()['Cluster_' + str(i) + '_MD'].drop(['pid', 'hid', 'cluster', 'pid.1',
                                                 'cluster.1','cluster.2', 
                                                 'Group Mean Doctor Visits', 
                                                 'Group-Demeaned Doctor Visits'], axis = 1, inplace = True)
        
    

#Proceeding as follows:

#1) Obtain mean and sd of each unstandardized variable (say mu and sigma).
#2) Unstandardize the individuals in the clusters using
    
#Value unstandardized in cluster = sigma*value in cluster + mu
    
#3) Compute the descriptive statistics on these unstandardized clusters.
#4) Compute differences between mean and sd of variables in the entire datasets
#vs. those in these unstandardized datasets.

###################################
##UNSTANDARDIZING POOLED CLUSTERS##
###################################

#1) Mean and std of each variable
    
#Leaving the name "means" in the dataframes' names, even though I am
#actually also taking the standard deviation now.

entire_pooled_descr.rename(columns = {'Unnamed: 0': 'Variable'}, inplace = True)

entire_pooled_descr_means = entire_pooled_descr[['Variable', 'mean', 'std']]

#2) Obtain unstandardized clusters

#2.1) Transpose entire_pooled_descr_means

entire_pooled_descr_means_T = entire_pooled_descr_means.T

#2.2) Transform first row ('Variable') in column names

entire_pooled_descr_means_T.columns = list(entire_pooled_descr_means_T.iloc[0])

entire_pooled_descr_means_T.drop(['Variable'], axis = 0, inplace = True)

#2.3) Put the mean and std of each variable as last two rows under the 
#associated variable/column in each of the clusters.

entire_pooled_descr_means_T.drop(['Pension', 'SE', 'dvisit', 'hid', 'pid', 'syear'], axis = 1, inplace = True) 

#list(entire_pooled_descr_means_sds_T) == list(Cluster_0_pooled)
#True, good.

Cluster_0_pooled_unst = pd.concat([entire_pooled_descr_means_T, Cluster_0_pooled], axis = 0)
Cluster_1_pooled_unst = pd.concat([entire_pooled_descr_means_T, Cluster_1_pooled], axis = 0)
Cluster_2_pooled_unst = pd.concat([entire_pooled_descr_means_T, Cluster_2_pooled], axis = 0)
Cluster_3_pooled_unst = pd.concat([entire_pooled_descr_means_T, Cluster_3_pooled], axis = 0)
Cluster_4_pooled_unst = pd.concat([entire_pooled_descr_means_T, Cluster_4_pooled], axis = 0)

#2.4) Modify each variable in Cluster_n_pooled_unst (n in {0,1,2,3,4}) so 
#that we are adding to it  

for i in [Cluster_0_pooled_unst, Cluster_1_pooled_unst, Cluster_2_pooled_unst, Cluster_3_pooled_unst, Cluster_4_pooled_unst]:
    
    for j in list(i):
    
        #i[j] = i[j].apply(lambda x: x + i[j].loc['std']*i[j].loc['mean']) 
        #is a bit slow. Let's rather create a new column.
        
        newname = j + '_unstand'
        
        i[newname] = i[j].loc['std']*i[j] + i[j].loc['mean'] 
        
        #This is instead immediate.
        
Cluster_0_pooled_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_1_pooled_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_2_pooled_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_3_pooled_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_4_pooled_unst.drop(to_del, axis = 1, inplace = True) 

Cluster_0_pooled_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_1_pooled_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_2_pooled_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_3_pooled_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_4_pooled_unst.drop(['mean','std'], axis = 0, inplace = True)

#2.6) Compute the summary statistics in the unstandardized clusters.

Cluster_0_pooled_unst_descr = Cluster_0_pooled_unst.astype(float).describe().T
Cluster_1_pooled_unst_descr = Cluster_1_pooled_unst.astype(float).describe().T
Cluster_2_pooled_unst_descr = Cluster_2_pooled_unst.astype(float).describe().T
Cluster_3_pooled_unst_descr = Cluster_3_pooled_unst.astype(float).describe().T
Cluster_4_pooled_unst_descr = Cluster_4_pooled_unst.astype(float).describe().T

Cluster_0_pooled_unst_descr = Cluster_0_pooled_unst_descr[['mean']]
Cluster_1_pooled_unst_descr = Cluster_1_pooled_unst_descr[['mean']]
Cluster_2_pooled_unst_descr = Cluster_2_pooled_unst_descr[['mean']]
Cluster_3_pooled_unst_descr = Cluster_3_pooled_unst_descr[['mean']]
Cluster_4_pooled_unst_descr = Cluster_4_pooled_unst_descr[['mean']]

#2.7) Make comparisons with values in the entire pooled by inner joining
#the two descriptive datasets. To do so, need to delete "_unstand"
#from the row indexes of Cluster_n_pooled_unst_descr and make it
#as a new joining column

Cluster_0_pooled_unst_descr['Variable'] = Cluster_0_pooled_unst_descr.index
Cluster_0_pooled_unst_descr.reset_index(inplace = True, drop = True)
Cluster_0_pooled_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_1_pooled_unst_descr['Variable'] = Cluster_1_pooled_unst_descr.index
Cluster_1_pooled_unst_descr.reset_index(inplace = True, drop = True)
Cluster_1_pooled_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_2_pooled_unst_descr['Variable'] = Cluster_2_pooled_unst_descr.index
Cluster_2_pooled_unst_descr.reset_index(inplace = True, drop = True)
Cluster_2_pooled_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_3_pooled_unst_descr['Variable'] = Cluster_3_pooled_unst_descr.index
Cluster_3_pooled_unst_descr.reset_index(inplace = True, drop = True)
Cluster_3_pooled_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_4_pooled_unst_descr['Variable'] = Cluster_4_pooled_unst_descr.index
Cluster_4_pooled_unst_descr.reset_index(inplace = True, drop = True)
Cluster_4_pooled_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_0_pooled_unst_descr['Variable'] = Cluster_0_pooled_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_1_pooled_unst_descr['Variable'] = Cluster_1_pooled_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_2_pooled_unst_descr['Variable'] = Cluster_2_pooled_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_3_pooled_unst_descr['Variable'] = Cluster_3_pooled_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_4_pooled_unst_descr['Variable'] = Cluster_4_pooled_unst_descr['Variable'].apply(lambda x: x[:-8])

Cluster_0_vs_entire = entire_pooled_descr_means.merge(Cluster_0_pooled_unst_descr, on = 'Variable', how = 'inner')
Cluster_1_vs_entire = entire_pooled_descr_means.merge(Cluster_1_pooled_unst_descr, on = 'Variable', how = 'inner')
Cluster_2_vs_entire = entire_pooled_descr_means.merge(Cluster_2_pooled_unst_descr, on = 'Variable', how = 'inner')
Cluster_3_vs_entire = entire_pooled_descr_means.merge(Cluster_3_pooled_unst_descr, on = 'Variable', how = 'inner')
Cluster_4_vs_entire = entire_pooled_descr_means.merge(Cluster_4_pooled_unst_descr, on = 'Variable', how = 'inner')

Cluster_0_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_1_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_2_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_3_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_4_vs_entire.drop(['std'], axis = 1, inplace = True)

#2.8) Final statistics of interest: absolute difference of means
#between mean in cluster and whole sample, and what percentage
#it represents w.r.to mean in whole sample. Investigate more
#for security odd values, as for instance cluster_mean = 1
#for eversmoke in cluster 0 (indeed, cluster mean 4 times larger).
#It's actually interesting!

Cluster_0_vs_entire['|mean - cluster mean|'] = (Cluster_0_vs_entire['mean'] - Cluster_0_vs_entire['cluster_mean']).abs()
Cluster_1_vs_entire['|mean - cluster mean|'] = (Cluster_1_vs_entire['mean'] - Cluster_1_vs_entire['cluster_mean']).abs()
Cluster_2_vs_entire['|mean - cluster mean|'] = (Cluster_2_vs_entire['mean'] - Cluster_2_vs_entire['cluster_mean']).abs()
Cluster_3_vs_entire['|mean - cluster mean|'] = (Cluster_3_vs_entire['mean'] - Cluster_3_vs_entire['cluster_mean']).abs()
Cluster_4_vs_entire['|mean - cluster mean|'] = (Cluster_4_vs_entire['mean'] - Cluster_4_vs_entire['cluster_mean']).abs()

Cluster_0_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_0_vs_entire['|mean - cluster mean|'] / Cluster_0_vs_entire['mean']) - 1)*100).abs()
Cluster_1_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_1_vs_entire['|mean - cluster mean|'] / Cluster_1_vs_entire['mean']) - 1)*100).abs()
Cluster_2_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_2_vs_entire['|mean - cluster mean|'] / Cluster_2_vs_entire['mean']) - 1)*100).abs()
Cluster_3_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_3_vs_entire['|mean - cluster mean|'] / Cluster_3_vs_entire['mean']) - 1)*100).abs()
Cluster_4_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_4_vs_entire['|mean - cluster mean|'] / Cluster_4_vs_entire['mean']) - 1)*100).abs()

#Cluster_0_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_0_vs_entire.csv')
#Cluster_1_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_1_vs_entire.csv')
#Cluster_2_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_2_vs_entire.csv')
#Cluster_3_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_3_vs_entire.csv')
#Cluster_4_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_4_vs_entire.csv')

#All the above steps now need to be done also for the clusters from the Mundlaked.
#Overall, they look interesting and worth exploring. Also the procedure to unstandardize
#the values look interesting and worth describing.
#I will write less comments anyway.

###############################################
##UNSTANDARDIZING TRANSFORMED POOLED CLUSTERS##
###############################################

#There are some oddities on the transformed pooled.
#Some values are infinities since some averages are already 0.

X_datacomplete_MD_1_descr.rename(columns = {'Unnamed: 0': 'Variable'}, inplace = True)

X_datacomplete_MD_1_descr_means = X_datacomplete_MD_1_descr[['Variable', 'mean', 'std']]

X_datacomplete_MD_1_descr_means = X_datacomplete_MD_1_descr_means[~X_datacomplete_MD_1_descr_means['Variable'].isin(['P[D]_SE', 'P[D]_Pension', 'M[D]_SE', 'M[D]_Pension'])]

X_datacomplete_MD_1_descr_means_T = X_datacomplete_MD_1_descr_means.T

X_datacomplete_MD_1_descr_means_T.columns = list(X_datacomplete_MD_1_descr_means_T.iloc[0])

X_datacomplete_MD_1_descr_means_T.drop(['Variable'], axis = 0, inplace = True)

Cluster_0_MD.drop(['Doctor Visits'], axis = 1, inplace = True)
Cluster_1_MD.drop(['Doctor Visits'], axis = 1, inplace = True)
Cluster_2_MD.drop(['Doctor Visits'], axis = 1, inplace = True)


Cluster_0_MD_unst = pd.concat([X_datacomplete_MD_1_descr_means_T, Cluster_0_MD], axis = 0)
Cluster_1_MD_unst = pd.concat([X_datacomplete_MD_1_descr_means_T, Cluster_1_MD], axis = 0)
Cluster_2_MD_unst = pd.concat([X_datacomplete_MD_1_descr_means_T, Cluster_2_MD], axis = 0)

for i in [Cluster_0_MD_unst, Cluster_1_MD_unst, Cluster_2_MD_unst]:
    
    for j in list(i):
    
        #i[j] = i[j].apply(lambda x: x + i[j].loc['std']*i[j].loc['mean']) a bit slow. Let's rather create a new column.
        
        newname = j + '_unstand'
        
        i[newname] = i[j].loc['std']*i[j] + i[j].loc['mean'] 

#Here, instead of using to_del define manually, I can use:

to_del = list(Cluster_0_MD) #of course identical to 1 and 2.

Cluster_0_MD_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_1_MD_unst.drop(to_del, axis = 1, inplace = True) 
Cluster_2_MD_unst.drop(to_del, axis = 1, inplace = True) 

Cluster_0_MD_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_1_MD_unst.drop(['mean','std'], axis = 0, inplace = True)
Cluster_2_MD_unst.drop(['mean','std'], axis = 0, inplace = True)

Cluster_0_MD_unst_descr = Cluster_0_MD_unst.astype(float).describe().T
Cluster_1_MD_unst_descr = Cluster_1_MD_unst.astype(float).describe().T
Cluster_2_MD_unst_descr = Cluster_2_MD_unst.astype(float).describe().T

Cluster_0_MD_unst_descr = Cluster_0_MD_unst_descr[['mean']]
Cluster_1_MD_unst_descr = Cluster_1_MD_unst_descr[['mean']]
Cluster_2_MD_unst_descr = Cluster_2_MD_unst_descr[['mean']]

Cluster_0_MD_unst_descr['Variable'] = Cluster_0_MD_unst_descr.index
Cluster_0_MD_unst_descr.reset_index(inplace = True, drop = True)
Cluster_0_MD_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_1_MD_unst_descr['Variable'] = Cluster_1_MD_unst_descr.index
Cluster_1_MD_unst_descr.reset_index(inplace = True, drop = True)
Cluster_1_MD_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_2_MD_unst_descr['Variable'] = Cluster_2_MD_unst_descr.index
Cluster_2_MD_unst_descr.reset_index(inplace = True, drop = True)
Cluster_2_MD_unst_descr.rename(columns = {'mean':'cluster_mean'}, inplace = True)

Cluster_0_MD_unst_descr['Variable'] = Cluster_0_MD_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_1_MD_unst_descr['Variable'] = Cluster_1_MD_unst_descr['Variable'].apply(lambda x: x[:-8])
Cluster_2_MD_unst_descr['Variable'] = Cluster_2_MD_unst_descr['Variable'].apply(lambda x: x[:-8])

Cluster_0_MD_vs_entire = X_datacomplete_MD_1_descr_means.merge(Cluster_0_MD_unst_descr, on = 'Variable', how = 'inner')
Cluster_1_MD_vs_entire = X_datacomplete_MD_1_descr_means.merge(Cluster_1_MD_unst_descr, on = 'Variable', how = 'inner')
Cluster_2_MD_vs_entire = X_datacomplete_MD_1_descr_means.merge(Cluster_2_MD_unst_descr, on = 'Variable', how = 'inner')

Cluster_0_MD_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_1_MD_vs_entire.drop(['std'], axis = 1, inplace = True)
Cluster_2_MD_vs_entire.drop(['std'], axis = 1, inplace = True)

Cluster_0_MD_vs_entire['|mean - cluster mean|'] = (Cluster_0_MD_vs_entire['mean'] - Cluster_0_MD_vs_entire['cluster_mean']).abs()
Cluster_1_MD_vs_entire['|mean - cluster mean|'] = (Cluster_1_MD_vs_entire['mean'] - Cluster_1_MD_vs_entire['cluster_mean']).abs()
Cluster_2_MD_vs_entire['|mean - cluster mean|'] = (Cluster_2_MD_vs_entire['mean'] - Cluster_2_MD_vs_entire['cluster_mean']).abs()

Cluster_0_MD_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_0_MD_vs_entire['|mean - cluster mean|'] / Cluster_0_MD_vs_entire['mean']) - 1)*100).abs()
Cluster_1_MD_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_1_MD_vs_entire['|mean - cluster mean|'] / Cluster_1_MD_vs_entire['mean']) - 1)*100).abs()
Cluster_2_MD_vs_entire['|mean - cluster mean| / mean as abs %'] = (((Cluster_2_MD_vs_entire['|mean - cluster mean|'] / Cluster_2_MD_vs_entire['mean']) - 1)*100).abs()

#Cluster_0_MD_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_0_MD_vs_entire.csv')
#Cluster_1_MD_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_1_MD_vs_entire.csv')
#Cluster_2_MD_vs_entire.to_csv('C:\\Some\\Local\\Path\\Cluster_2_MD_vs_entire.csv')
