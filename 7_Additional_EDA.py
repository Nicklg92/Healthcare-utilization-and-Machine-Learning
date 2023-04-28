###########################################################
###SEVENTH SCRIPT - ADDITIONAL EXPLORATORY DATA ANALYSIS###
###########################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 40)

np.random.seed(1123581321)

'''
COMMENTS

This is the seventh script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

Aim of this script is to do all the exploratory data analysis calculations,
both on the Mundlak transformed dataset as well as on the pooled one, both
in clusters and on the whole data.

Here I do:
    
-- Plots of dvisit and each variables.
-- Print of summary statistics.
-- Joint disitributions/scatter plots.
-- Correlation maps/heatmaps.
-- Histogram of pairwise Euclidean distances.

I can also use this script to check consistency in the data and 
alike diagnostics.

Moreover, the observed qualitative characteristics of the data are 
commented and put in comparison with the quantitative results observed 
in the previous scripts.

'''

###########################################
###IMPORTING ALL POOLED - UNSTANDARDIZED###
###########################################

datacomplete_unstand = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_1_unstand.csv')

#############################
###IMPORTING ENTIRE POOLED###
#############################

train_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_train.csv')

test_datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed_test.csv')

datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed.csv')

train_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

test_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

#########################################
###IMPORTING ENTIRE TRANSFORMED POOLED###
#########################################

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

################################################
###IMPORTING CLUSTERS FROM TRANSFORMED POOLED###
################################################
    
for i in [0,1,2]:
    
    Train_name = "train_cluster_mundlaked_" + str(i) 
    
    Test_name = "test_cluster_mundlaked_" + str(i) 
        
    path_train = 'C:\\Some\\Local\\Path\\' + Train_name + '.csv'
    
    path_test = 'C:\\Some\\Local\\Path\\' + Test_name + '.csv'

    globals()["Train_cluster_MD_" + str(i)] = pd.read_csv(path_train)
        
    globals()["Test_cluster_MD_" + str(i)] = pd.read_csv(path_test)
    
del Test_name, Train_name, i, path_test, path_train

#The pairwise Euclidean distances' computations take quite sometime.
#Hence, not redoing them.

##############################################################
###CHECKING EQUALITY DATACOMPLETE AND DATACOMPLETE_UNSTAND###
##############################################################

#I further need to assess that datacomplete and datacomplete_unstand
#are the same animal, so that I will be able to compute all the summary
#statistics on the latter.

#It is in general a good consistency check.

#Filling the remaining NaNs
#(as in 1_Summary_statistics_and_cluster_creation_pooled_specification.py
#for the standardized datacomplete_1).

datacomplete_unstand['eversmoke'].fillna(datacomplete_unstand['eversmoke'].mode()[0], inplace = True)

datacomplete_unstand['insured'].fillna(datacomplete_unstand['insured'].mode()[0], inplace = True)

datacomplete_unstand['alone'].fillna(datacomplete_unstand['alone'].mode()[0], inplace = True)

datacomplete_unstand['isced'].fillna(datacomplete_unstand['isced'].mode()[0], inplace = True)

datacomplete_unstand['disabled'].fillna(datacomplete_unstand['disabled'].mean(), inplace = True)

datacomplete_unstand['masmoke'].fillna(datacomplete_unstand['masmoke'].mean(), inplace = True)

datacomplete_unstand.isna().sum().sum()

#And getting the dummies (as in the afoermentioned) 

ohed_workstat = pd.get_dummies(datacomplete_unstand['workstat'])

ohed_isced = pd.get_dummies(datacomplete_unstand['isced'])

datacomplete_unstand_ohed = pd.concat([datacomplete_unstand, ohed_isced, ohed_workstat], axis = 1)

datacomplete_unstand_ohed.drop(['workstat', 'isced'], axis = 1, inplace = True)

#At this point, datacomplete_unstand_ohed should be simply the unstndardized,
#identical version of datacomplete.

datacomplete_unstand_ohed.drop(['everchronicill'], axis = 1, inplace = True)

datacomplete_unstand_ohed.describe().T

datacomplete.describe().T

#Now, also eversmoke, masmoke and alone should be equal.

np.sum(datacomplete_unstand_ohed['eversmoke'] != datacomplete['eversmoke'])

#0

np.sum(datacomplete_unstand_ohed['masmoke'] != datacomplete['masmoke'])

#They are identical.

np.sum(datacomplete_unstand_ohed['alone'] != datacomplete['alone'])

#0

#To further check it:

#datacomplete_unstand_ohed['masmoke'].unique()

#datacomplete['masmoke'].unique()

#They are indeed equal.

#np.sum(datacomplete_unstand_ohed['dvisit'] != datacomplete['dvisit'])

#0

'''
##############################################################
##HISTOGRAM OF PAIRWISE EUCLIDEAN DISTANCES - POOLED DATASET##
##############################################################

#In order to avoid memory errors, we consider a random subsample
#of individuals. 

datacomplete_for_euclid = datacomplete.drop(['pid', 'syear', 'hid', 'dvisit'], axis = 1)

datacomplete_sampled = datacomplete_for_euclid.sample(110000)

start_time = time.time()

eucl_dists_datacomplete = pdist(datacomplete_sampled.values, 'euclid')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Runtime was 107.42464900016785 seconds with 100.000 observations.

to_bin = np.arange(np.round(np.min(eucl_dists_datacomplete), 2), np.round(np.max(eucl_dists_datacomplete), 2), 0.5)

#np.min(eucl_dists_datacomplete)
#0.035443321577562026

#np.max(eucl_dists_datacomplete)
#102.80968523806658

#Computing the max and min takes around 3 minutes each.
#Hence, the time complexity is neither in computing max and min, nor in the computations
#of Euclidean distances themselves.

#To create the bins, thakes indeed just a few seconds. The problem is therefore
#in the plotting itself, which makes sense, since we have 100.000(100.000 - 1)/2
#observations to plot. Let's stick to 30.000.

#plt.hist(eucl_dists_datacomplete, density = True, bins = to_bin)

#plt.xlabel("Pairwise Euclidean distances - pooled data")


#Plotting with 30.000 takes around 100 seconds, and super cool graph with
#plenty of picks! This indeed suggests why we are observing increase
#in performances in the clusters here. Moreover, considering
#that clusters are built only based on indepvars, this seems to suggest
#that it is not the nature of the depvar to actually have an influence. 

del datacomplete_for_euclid, datacomplete_sampled, eucl_dists_datacomplete, to_bin

#################################################################
##HISTOGRAM OF PAIRWISE EUCLIDEAN DISTANCES - MUNDLAKED DATASET##
#################################################################

X_datacomplete_MD_stand_1_for_euclid = X_datacomplete_MD_stand_1.drop(['pid', 'hid', 'Unnamed: 0'], axis = 1)

X_datacomplete_MD_stand_1_sampled = X_datacomplete_MD_stand_1_for_euclid.sample(110000)

#Differently from Datacomplete, here we have many more
#variables: this will surely affect the time complexity.

start_time = time.time()

eucl_dists_datacomplete_MD_stand_1 = pdist(X_datacomplete_MD_stand_1_sampled.values, 'euclid')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Runtime was 1.6817409992218018 seconds, 10.000 observations.
#Runtime was 15.062766790390015 seconds, 30.000 observations.
#Runtime was 65.24083638191223 seconds, 60.000 observations.
#Runtime was 116.81240630149841 seconds, 80.000 observations. 

to_bin = np.arange(np.round(np.min(eucl_dists_datacomplete_MD_stand_1), 2), np.round(np.max(eucl_dists_datacomplete_MD_stand_1), 2), 0.5)

#np.min(eucl_dists_datacomplete_MD_stand_1)
#0.2328572399926791

#np.max(eucl_dists_datacomplete_MD_stand_1)
#41.36339179664727

plt.hist(eucl_dists_datacomplete_MD_stand_1, density = True, bins = to_bin)

plt.xlabel("Pairwise Euclidean distances - transformed pooled")
'''

################################
###DOCTOR VISITS DISTRIBUTION###
################################

datacomplete_unstand_ohed['dvisit'].describe().T


#count    208903.000000
#mean          2.465321
#std           3.817630
#min           0.000000
#25%           0.000000
#50%           1.000000
#75%           3.000000
#max          99.000000


#How many values greater than 10?

np.sum(datacomplete_unstand_ohed['dvisit'] > 10)

#(5917/208903)*100 = 2.83%

##How many values greater than 20?

np.sum(datacomplete_unstand_ohed['dvisit'] > 20)

#(1043/208903)*100 = 0.50%

#Maybe we should windsorize here?
#Who are these people?

more_than_ten = datacomplete_unstand_ohed[datacomplete_unstand_ohed['dvisit'] > 10]

more_than_ten.describe().T

arr = np.sort(datacomplete_unstand_ohed['dvisit'].unique())

sns.displot(datacomplete_unstand_ohed['dvisit'], bins = arr, kind = "hist")

sns.ecdfplot(datacomplete_unstand_ohed['dvisit'])

#Let's see if we can observe something better with dvisits <= 20

less_than_20 = datacomplete_unstand_ohed['dvisit'][datacomplete_unstand_ohed['dvisit'] <= 20]

arr_less_than_20 = np.sort(less_than_20.unique())

sns.displot(less_than_20, bins = arr_less_than_20, kind = "hist")

plt.xlabel('Number of Doctor Visits')

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

plt.show()

less_than_20_df = pd.DataFrame(less_than_20, columns = ['dvisit'])

sns.ecdfplot(less_than_20_df, stat = 'proportion', x = 'dvisit')

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

plt.show()

#From these graphs, possible questions/conclusions.
#We now know that ML algorithms are particularly vulnerable
#to outliers and extreme values. However, on dvisit, we already
#observe that they are outperforming Linear methods.

#Hence, for how reasonable it would be to windsorize at, say, 99.5% (20 visits),
#this would likely give a further advantage to ML methods.

############################################################
###DESCRIPTIONS OF INDEPENDENT VARIABLES IN ENTIRE POOLED###
############################################################

#Also in this case, we differentiate the analysis for those who
#have less than or more/equal to 20 dvisits.

#What characteristics define these people the most?

#Let's redefine less_than_20_df as the entire df (not only dvisits)
#of who has gone to the doctor more than 20 times in 3 months.

less_than_20_df = datacomplete_unstand_ohed[datacomplete_unstand_ohed['dvisit'] <= 20]

more_than_20_df = datacomplete_unstand_ohed[datacomplete_unstand_ohed['dvisit'] > 20]

less_than_20_df.describe().T

more_than_20_df.describe().T

#Saving them as .csvs

description_less_than_20_df = less_than_20_df.describe().T

#description_less_than_20_df.to_csv('C:\\Some\\Local\\Path\\description_less_than_20_df.csv')

description_more_than_20_df = more_than_20_df.describe().T

#description_more_than_20_df.to_csv('C:\\Some\\Local\\Path\\description_more_than_20_df.csv')

description_all = datacomplete_unstand_ohed.describe().T

#description_all.to_csv('C:\\Some\\Local\\Path\\description_all.csv')

#The categoricals isced and workstat have already been ohed.

#The only striking difference across the two groups is the degree of disability,
#which indeed in the Machine Learning estimations was appearing to be among the
#most important variables, although self-assessed health and physiological scale
#were dominating.

#Hence, the algorithms have been capable of extracting information not immediately
#visible via descriptive.

##########################
###CORRELATION MATRICES###
##########################

less_than_20_df_corrs = less_than_20_df.corr(method = 'pearson')

more_than_20_df_corrs = more_than_20_df.corr(method = 'pearson')

all_df_corrs = datacomplete_unstand_ohed.corr(method = 'pearson')

#less_than_20_df_corrs.to_csv('C:\\Some\\Local\\Path\\less_than_20_df_corrs.csv')

#more_than_20_df_corrs.to_csv('C:\\Some\\Local\\Path\\more_than_20_df_corrs.csv')

#all_df_corrs.to_csv('C:\\Some\\Local\\Path\\all_df_corrs.csv')

#In the .csvs, I will only be highlighting those more than |rho| > 0.50.

#Overall, considering |rho| > 0.50, we don't see major correlation issues.

#This indees sugests that lower-variance linear methods like Ridge and LASSO 
#are useless here, and a non-penalized Linear Regression already suffices as 
#benchmark for the Random Forests.

#Let's now see if any interesting scatter plot emerges.

###################
##ALL INDIVIDUALS##
###################

#"The main-diagonal subplots are the univariate histograms
#(distributions) for each attribute".

list(datacomplete_unstand_ohed)

to_plot = [x for x in list(datacomplete_unstand_ohed) if x not in ['pid', 'hid', 'syear']]

len(to_plot) #22

#sns.pairplot(datacomplete_unstand_ohed[to_plot].sample(10000), diag_kind = 'kde', corner = True)

#Let's try again with subgroups.

group_1 = ['dvisit','selfahealthimp','disabled','eversmoke',
           'masmoke','bmiimp2','insured','alone']

group_2 = ['hhincap', 'gender', 'age', 'PsyScaleimp2',
           'PhyScaleimp2','BA','MA'] 

group_3 = ['SE', 'lower', 'FT', 'NO', 'PT', 'Pension', 'VTraining']

#Plots to make: 1 vs. 2, 1 vs. 3, 2 vs. 3, 1 vs. 1, 2 vs. 2, 3 vs. 3

datacomplete_unstand_ohed_subset = datacomplete_unstand_ohed[to_plot].sample(10000)

g_1 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_2)

g_1.map(sns.scatterplot)

g_2 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_3)

g_2.map(sns.scatterplot)


g_3 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_2,
                   y_vars = group_3)

g_3.map(sns.scatterplot)

sns.pairplot(datacomplete_unstand_ohed_subset[group_1], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_2], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_3], diag_kind = 'kde')

#Take home: at the global level, these graphs tell us nothing.
#Mayvbe for the case dvisits < 20?

################
##DVISITS < 20##
################

datacomplete_unstand_ohed_subset = less_than_20_df[to_plot].sample(10000)

g_1 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_2)

g_1.map(sns.scatterplot)

g_2 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_3)

g_2.map(sns.scatterplot)


g_3 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_2,
                   y_vars = group_3)

g_3.map(sns.scatterplot)

sns.pairplot(datacomplete_unstand_ohed_subset[group_1], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_2], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_3], diag_kind = 'kde')

################
##DVISITS > 20##
################

#Only 1043 observations in this case, no need to subsample.

datacomplete_unstand_ohed_subset = more_than_20_df[to_plot]

g_1 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_2)

g_1.map(sns.scatterplot)

g_2 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_1,
                   y_vars = group_3)

g_2.map(sns.scatterplot)


g_3 = sns.PairGrid(datacomplete_unstand_ohed_subset,
                   x_vars = group_2,
                   y_vars = group_3)

g_3.map(sns.scatterplot)

sns.pairplot(datacomplete_unstand_ohed_subset[group_1], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_2], diag_kind = 'kde')

sns.pairplot(datacomplete_unstand_ohed_subset[group_3], diag_kind = 'kde')

#All the above steps now need to be done in the clusters.

#For improved readability, in re-running the abpve, let's not run
#the parts in which I create the graphs.


########################################
###IN-CLUSTERS ANALYSIS - POOLED CASE###
########################################

Cluster_0 = pd.concat([Train_cluster_0, Test_cluster_0], ignore_index = True)

Cluster_1 = pd.concat([Train_cluster_1, Test_cluster_1], ignore_index = True)

Cluster_2 = pd.concat([Train_cluster_2, Test_cluster_2], ignore_index = True)

Cluster_3 = pd.concat([Train_cluster_3, Test_cluster_3], ignore_index = True)

Cluster_4 = pd.concat([Train_cluster_4, Test_cluster_4], ignore_index = True)

#################################################
##DISTRIBUTION OF YEARS IN CLUSTERS FROM POOLED##
#################################################

Cluster_0['syear'].nunique()

#11

Cluster_1['syear'].nunique()

#11

Cluster_2['syear'].nunique()

#11

Cluster_3['syear'].nunique()

#11

Cluster_4['syear'].nunique()

#11

Cluster_0_pooled_descr = Cluster_0.describe().T

Cluster_1_pooled_descr = Cluster_1.describe().T

Cluster_2_pooled_descr = Cluster_2.describe().T

Cluster_3_pooled_descr = Cluster_3.describe().T

Cluster_4_pooled_descr = Cluster_4.describe().T

#Let's compare mean. std, median and max of the values at the global and each-cluster
#level.

Cluster_0_pooled_descr.drop(['syear', 'pid', 'hid', 'cluster', 'cluster.1'], inplace = True)

Cluster_1_pooled_descr.drop(['syear', 'pid', 'hid', 'cluster', 'cluster.1'], inplace = True)

Cluster_2_pooled_descr.drop(['syear', 'pid', 'hid', 'cluster', 'cluster.1'], inplace = True)

Cluster_0_mean_median_sd = pd.DataFrame(Cluster_0_pooled_descr[['mean', 'std', '50%', 'max']])

Cluster_0_mean_median_sd.columns = ['mean_0', 'std_0', '50%_0', 'max_0']

Cluster_1_mean_median_sd = pd.DataFrame(Cluster_1_pooled_descr[['mean', 'std', '50%', 'max']])

Cluster_1_mean_median_sd.columns = ['mean_1', 'std_1', '50%_1', 'max_1']

Cluster_2_mean_median_sd = pd.DataFrame(Cluster_2_pooled_descr[['mean', 'std', '50%', 'max']])

Cluster_2_mean_median_sd.columns = ['mean_2', 'std_2', '50%_2', 'max_2']

Cluster_3_mean_median_sd = pd.DataFrame(Cluster_3_pooled_descr[['mean', 'std', '50%', 'max']])

Cluster_3_mean_median_sd.columns = ['mean_3', 'std_3', '50%_3', 'max_3']

Cluster_4_mean_median_sd = pd.DataFrame(Cluster_4_pooled_descr[['mean', 'std', '50%', 'max']])

Cluster_4_mean_median_sd.columns = ['mean_4', 'std_4', '50%_4', 'max_4']

#How to interpret?

#For instance, in cluster 0 only people whose mean is
#eversmoke = 1.59. 

#In cluster 2, only people with the VTraining dummy = -0.24

#Given that eversmoke ("Whether Ever Smoked”), is a binary variable for 
#whether the individual has ever smoked, it could look weird that 
#its mean, in Cluster 0 is 1.59. However, this is fine:
    
#On datacomplete_unstand we can see that:
    
#datacomplete_unstand["eversmoke"].describe().T

#count   208903.00
#mean         0.28
#std          0.45
#min          0.00
#25%          0.00
#50%          0.00
#75%          1.00
#max          1.00
#Name: eversmoke, dtype: float64

#Meaning that, when standardizing:
    
#People who never smoked, hence 0 in eversmoked, become:
#(0 - 0.28) / 0.45 = -0.62

#People who have indeed smoked, hence 1 in eversmoked, become:
#(1 - 0.28) / 0.45 = 1.59

#and since the clusters are obtained from the standardized version
#of datacomplete, mean = 1.59 means in fact all smokers. Hence
#why in Table 2 in Appendix I directly wrote 1.

##############################################
##DVISITS DISTRIBUTION IN CLUSTERS BY POOLED##
##############################################

#At the global level, we had 2.83% and 0.50% for greater 10 and 20.

#The average was 2.89 at the global level.
#I should see in the clusters vs. at the global level how many more there are. 

#How many values greater than 10 in Cluster_0?

np.sum(Cluster_0['dvisit'] > 10)

#(922/37057)*100 = 2.49%

##How many values greater than 20 in Cluster_0?

np.sum(Cluster_0['dvisit'] > 20)

#(182/37057)*100 = 0.49%

#How many values greater than 10 in Cluster_1?

np.sum(Cluster_1['dvisit'] > 10)

#(2854/51799)*100 = 5.51%

##How many values greater than in Cluster_1?

np.sum(Cluster_1['dvisit'] > 20)

#(569/51799)*100 = 1.10%

#How many values greater than 10 in Cluster_2?

np.sum(Cluster_2['dvisit'] > 10)

#(839/57886)*100 = 1.45%

##How many values greater than in Cluster_2?

np.sum(Cluster_2['dvisit'] > 20)

#(113/57886)*100 = 0.20%

np.sum(Cluster_3['dvisit'] > 10)

#(1180/50573)*100 = 2.33%

np.sum(Cluster_3['dvisit'] > 20)

#(164/50573)*100 = 0.324%

np.sum(Cluster_4['dvisit'] > 10)

#(122/11588)*100 = 1.05%

np.sum(Cluster_4['dvisit'] > 20)

#(15/11588)*100 = 0.12%

#In other words, K-means decided to create three clusters, 
#of which one with individuals healthy as average, one with particularly
#healthcare-utilizing people, and one with people particluarly underusing it.

#Since K-means is unsupervised (meaning that clusters are based on the features
#ignoring the depvar dvisits) is interesting what is different between the
#three (correlations and scatter plots).

##########################################
###SCATTER PLOTS IN CLUSTERS BY POOLING###
##########################################

#Also in the clusters we do the distinction in the plots between
#all individuals and less than/more than 20 dvisits.

###################################
##DVISITS IN CLUSTER 0 BY POOLING##
###################################

arr_less_than_20 = np.arange(0,20)

save_path_clusters = 'C:\\Some\\Local\\Path\\'

#Less than 20

Cluster_0_less_than_20 = pd.DataFrame(Cluster_0[Cluster_0['dvisit'] <= 20])

Cluster_0_less_than_20_df = pd.DataFrame(Cluster_0_less_than_20, columns = ['dvisit'])

sns.displot(Cluster_0_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'dvisits_less20_cluster0.png')

plt.show()


###################################
##DVISITS IN CLUSTER 1 BY POOLING##
###################################

#Less than 20


Cluster_1_less_than_20 = pd.DataFrame(Cluster_1[Cluster_1['dvisit'] <= 20])

Cluster_1_less_than_20_df = pd.DataFrame(Cluster_1_less_than_20, columns = ['dvisit'])

sns.displot(Cluster_1_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'dvisits_less20_cluster1.png')

plt.show()

###################################
##DVISITS IN CLUSTER 2 BY POOLING##
###################################

#Less than 20


Cluster_2_less_than_20 = pd.DataFrame(Cluster_2[Cluster_2['dvisit'] <= 20])

Cluster_2_less_than_20_df = pd.DataFrame(Cluster_2_less_than_20, columns = ['dvisit'])

sns.displot(Cluster_2_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'dvisits_less20_cluster2.png')

plt.show()

###################################
##DVISITS IN CLUSTER 3 BY POOLING##
###################################

#Less than 20


Cluster_3_less_than_20 = pd.DataFrame(Cluster_3[Cluster_3['dvisit'] <= 20])

Cluster_3_less_than_20_df = pd.DataFrame(Cluster_3_less_than_20, columns = ['dvisit'])

sns.displot(Cluster_3_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'dvisits_less20_cluster3.png')

plt.show()


###################################
##DVISITS IN CLUSTER 4 BY POOLING##
###################################

#Less than 20


Cluster_4_less_than_20 = pd.DataFrame(Cluster_4[Cluster_4['dvisit'] <= 20])

Cluster_4_less_than_20_df = pd.DataFrame(Cluster_4_less_than_20, columns = ['dvisit'])

sns.displot(Cluster_4_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'dvisits_less20_cluster4.png')

plt.show()

#################################################
###CORRELATION MATRICES IN CLUSTERS BY POOLING###
#################################################

Cluster_0_corrs = Cluster_0.corr(method = 'pearson')

Cluster_1_corrs = Cluster_1.corr(method = 'pearson')

Cluster_2_corrs = Cluster_2.corr(method = 'pearson')

Cluster_3_corrs = Cluster_3.corr(method = 'pearson')

Cluster_4_corrs = Cluster_4.corr(method = 'pearson')

#Cluster_0_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_0_pooled_corrs.csv')

#Cluster_1_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_1_pooled_corrs.csv')

#Cluster_2_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_2_pooled_corrs.csv')

#Cluster_3_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_3_pooled_corrs.csv')

#Cluster_4_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_4_pooled_corrs.csv')

#Overall, nothing particularly cool comes out of these correlations.
#In the clusters, they seem to be less strong.

####################################################
###IN-CLUSTERS ANALYSIS - TRANSFORMED POOLED CASE###
####################################################

arr_less_than_20 = np.arange(0,20)

Cluster_MD_0 = pd.concat([Train_cluster_MD_0, Test_cluster_MD_0], ignore_index = True)

Cluster_MD_1 = pd.concat([Train_cluster_MD_1, Test_cluster_MD_1], ignore_index = True)

Cluster_MD_2 = pd.concat([Train_cluster_MD_2, Test_cluster_MD_2], ignore_index = True)

################################################################
###CORRELATION MATRICES IN CLUSTERS FROM TRANSFORMED - POOLED###
################################################################

Cluster_0_MD_corrs = Cluster_MD_0.corr(method = 'pearson')

Cluster_1_MD_corrs = Cluster_MD_1.corr(method = 'pearson')

Cluster_2_MD_corrs = Cluster_MD_2.corr(method = 'pearson')

#Cluster_0_MD_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_0_pooled_MD_corrs.csv')

#Cluster_1_MD_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_1_pooled_MD_corrs.csv')

#Cluster_2_MD_corrs.to_csv('C:\\Some\\Local\\Path\\Cluster_2_pooled_MD_corrs.csv')

#########################################################
##DVISITS IN THE THREE CLUSTERS FROM TRANSFORMED POOLED##
#########################################################

##############################################
##DVISITS IN CLUSTER 0 BY TRANSFORMED POOLED##
##############################################

#Less than 20

Cluster_MD_0_less_than_20 = pd.DataFrame(Cluster_MD_0[Cluster_MD_0['Doctor Visits'] <= 20])

Cluster_MD_0_less_than_20_df = pd.DataFrame(Cluster_MD_0_less_than_20, columns = ['Doctor Visits'])

sns.displot(Cluster_MD_0_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'Doctor Visitss_less20_cluster0.png')

plt.show()


##############################################
##DVISITS IN CLUSTER 1 BY TRANSFORMED POOLED##
##############################################

#Less than 20


Cluster_MD_1_less_than_20 = pd.DataFrame(Cluster_MD_1[Cluster_MD_1['Doctor Visits'] <= 20])

Cluster_MD_1_less_than_20_df = pd.DataFrame(Cluster_MD_1_less_than_20, columns = ['Doctor Visits'])

sns.displot(Cluster_MD_1_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'Doctor Visitss_less20_cluster1.png')

plt.show()

##############################################
##DVISITS IN CLUSTER 2 BY TRANSFORMED POOLED##
##############################################

#Less than 20


Cluster_MD_2_less_than_20 = pd.DataFrame(Cluster_MD_2[Cluster_MD_2['Doctor Visits'] <= 20])

Cluster_MD_2_less_than_20_df = pd.DataFrame(Cluster_MD_2_less_than_20, columns = ['Doctor Visits'])

sns.displot(Cluster_MD_2_less_than_20_df, bins = arr_less_than_20, kind = "hist", legend = False)

plt.xticks(arr_less_than_20)

plt.figure(figsize = (30,8))

#plt.savefig(save_path_clusters + 'Doctor Visitss_less20_cluster2.png')

plt.show()