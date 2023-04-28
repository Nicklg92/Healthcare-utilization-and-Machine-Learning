################################################################################
###THIRD SCRIPT - CREATING THE TRANSFORMED POOLED DATASET, ENTIRE AND CLUSTERS##
################################################################################

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import randint

scaler = StandardScaler()

np.random.seed(1123581321)

'''
COMMENTS:
    
This is the third script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

Defining "means of independent variables" as:
    
average over T periods of the variable j for individual i 
(also called "group-mean" in the Econometric literature)

and "demeaned" as:
    
value of variable j for individual i at time t - average over the T 
periods of variable j for individual i
(also called "group-mean deviations" in the Econometric literature) 

the aim of this script is to fit and predict:
    
Healthcare utilization = f(means of independent variables, demeaned independent variables)
    
that is, formally:    

y_(i,t) = f((\overline(X_(i)), X_(i,t) - \overline(X_(i)) + ε_(i,t)
        
On top of these, I also create the three clusters of individuals using (the
Unsupervised Learning algorithm) K-Means, as previously done in the Pooled 
specification.

And as in the previous case, we extract the Absolute Coefficients and
Mean Absolute Shapley Values.
'''

#Let's import the full version:

datacomplete = pd.read_csv('C:\\Some\\Local\\Path\\datacomplete_ohed_imputed.csv')

train_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

test_datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

datacomplete.drop(['Unnamed: 0'], axis = 1, inplace = True)

#Need to save the pids, hids, and healthcare utilization aside.
#Will concatenate them to the Transformed pooled dataset after.

pid_hid_all = X_datacomplete[['pid', 'hid']]

pid_hid_train = X_train_datacomplete[['pid', 'hid']]

pid_hid_test = X_test_datacomplete[['pid', 'hid']]

y_datacomplete = datacomplete[['pid', 'dvisit']]

y_train_datacomplete = train_datacomplete[['pid', 'dvisit']]

y_test_datacomplete = test_datacomplete[['pid', 'dvisit']]

X_datacomplete = datacomplete.drop(['dvisit'], axis = 1)

X_train_datacomplete = train_datacomplete.drop(['dvisit'], axis = 1)

X_test_datacomplete = test_datacomplete.drop(['dvisit'], axis = 1)

#####################################################
###ENTIRE DATASET - GROUP-MEAND AND GROUP-DEMEANED###
#####################################################
    
#Computing P[D]y_datacomplete and M[D]y_datacomplete

y_datacomplete_PD = pd.DataFrame(y_datacomplete.groupby(['pid']).mean())

y_datacomplete_PD['pid'] = y_datacomplete_PD.index

y_datacomplete_PD.reset_index(inplace = True, drop = True)

y_datacomplete_PD_y = y_datacomplete.merge(y_datacomplete_PD, how = 'inner', on = 'pid')

y_datacomplete_PD_y.columns = ['pid', 'Doctor Visits', 'Group Mean Doctor Visits']

y_datacomplete_MD = pd.DataFrame(y_datacomplete_PD_y['Doctor Visits'] - y_datacomplete_PD_y['Group Mean Doctor Visits'], columns = ['Group-Demeaned Doctor Visits'])

#Computing P[D]X_datacomplete and M[D]X_datacomplete

X_datacomplete_MD = X_datacomplete.copy()

for i in list(X_datacomplete):
    
    if i in ['pid', 'syear', 'hid']:
        
        print('No within transformation for pid, year and hid')
    
    else:
    
        var_1_PD = pd.DataFrame(X_datacomplete_MD[['pid', i]].groupby(['pid']).mean())
    
        var_1_PD['pid'] = var_1_PD.index
    
        var_1_PD.reset_index(inplace = True, drop = True)
        
        X_datacomplete_MD = X_datacomplete_MD.merge(var_1_PD, how = 'inner', on = 'pid')
        
        cols_to_change = [i + '_x', i + '_y']
        
        PD_x = 'P[D]_' + i
        
        X_datacomplete_MD.rename(columns = {cols_to_change[0]: i,
                                     cols_to_change[1]: PD_x}, inplace = True)
        
        MD_x = 'M[D]_' + i
    
        X_datacomplete_MD[MD_x] =  X_datacomplete_MD[i] - X_datacomplete_MD[PD_x]

#Let's drop constant columns                 
        
for j in list(X_datacomplete_MD):
    
    if np.sum(X_datacomplete_MD[j] == 0) == len(X_datacomplete_MD):
        
        X_datacomplete_MD.drop([j], axis = 1, inplace = True)
        
#np.sum(y_datacomplete_PD_y['pid'] != X_datacomplete_MD['pid'])
#0
        
#np.sum(y_test_PD_y['pid'] != X_test_MD['pid'])
#0        

vars_no_PD_MD = [x for x in list(X_datacomplete_MD) if x[0:4] != 'P[D]' and x[0:4] != 'M[D]']

X_datacomplete_MD.drop(vars_no_PD_MD, axis = 1, inplace = True)

#######################################################
###TRAINING DATASET - GROUP-MEAND AND GROUP-DEMEANED###
#######################################################
    
#Computing P[D]y_train_datacomplete and M[D]y_train_datacomplete

y_train_datacomplete_PD = pd.DataFrame(y_train_datacomplete.groupby(['pid']).mean())

y_train_datacomplete_PD['pid'] = y_train_datacomplete_PD.index

y_train_datacomplete_PD.reset_index(inplace = True, drop = True)

y_train_datacomplete_PD_y = y_train_datacomplete.merge(y_train_datacomplete_PD, how = 'inner', on = 'pid')

y_train_datacomplete_PD_y.columns = ['pid', 'Doctor Visits', 'Group Mean Doctor Visits']

y_train_datacomplete_MD = pd.DataFrame(y_train_datacomplete_PD_y['Doctor Visits'] - y_train_datacomplete_PD_y['Group Mean Doctor Visits'], columns = ['Group-Demeaned Doctor Visits'])

#Computing P[D]X_train_datacomplete and M[D]X_train_datacomplete

X_train_datacomplete_MD = X_train_datacomplete.copy()

for i in list(X_train_datacomplete):
    
    if i in ['pid', 'syear', 'hid']:
        
        print('No within transformation for pid, year and hid')
    
    else:
    
        var_1_PD = pd.DataFrame(X_train_datacomplete_MD[['pid', i]].groupby(['pid']).mean())
    
        var_1_PD['pid'] = var_1_PD.index
    
        var_1_PD.reset_index(inplace = True, drop = True)
        
        X_train_datacomplete_MD = X_train_datacomplete_MD.merge(var_1_PD, how = 'inner', on = 'pid')
        
        cols_to_change = [i + '_x', i + '_y']
        
        PD_x = 'P[D]_' + i
        
        X_train_datacomplete_MD.rename(columns = {cols_to_change[0]: i,
                                     cols_to_change[1]: PD_x}, inplace = True)
        
        MD_x = 'M[D]_' + i
    
        X_train_datacomplete_MD[MD_x] =  X_train_datacomplete_MD[i] - X_train_datacomplete_MD[PD_x]

#Let's drop constant columns                 
        
for j in list(X_train_datacomplete_MD):
    
    if np.sum(X_train_datacomplete_MD[j] == 0) == len(X_train_datacomplete_MD):
        
        X_train_datacomplete_MD.drop([j], axis = 1, inplace = True)
        
#np.sum(y_train_datacomplete_PD_y['pid'] != X_train_datacomplete_MD['pid'])
#0
        
#np.sum(y_test_PD_y['pid'] != X_test_MD['pid'])
#0        

vars_no_PD_MD = [x for x in list(X_train_datacomplete_MD) if x[0:4] != 'P[D]' and x[0:4] != 'M[D]']

X_train_datacomplete_MD.drop(vars_no_PD_MD, axis = 1, inplace = True)


######################################################
###TESTING DATASET - GROUP-MEAND AND GROUP-DEMEANED###
######################################################
    
#Computing P[D]y_test_datacomplete and M[D]y_test_datacomplete

y_test_datacomplete_PD = pd.DataFrame(y_test_datacomplete.groupby(['pid']).mean())

y_test_datacomplete_PD['pid'] = y_test_datacomplete_PD.index

y_test_datacomplete_PD.reset_index(inplace = True, drop = True)

y_test_datacomplete_PD_y = y_test_datacomplete.merge(y_test_datacomplete_PD, how = 'inner', on = 'pid')

y_test_datacomplete_PD_y.columns = ['pid', 'Doctor Visits', 'Group Mean Doctor Visits']

y_test_datacomplete_MD = pd.DataFrame(y_test_datacomplete_PD_y['Doctor Visits'] - y_test_datacomplete_PD_y['Group Mean Doctor Visits'], columns = ['Group-Demeaned Doctor Visits'])

#Computing P[D]X_test_datacomplete and M[D]X_test_datacomplete

X_test_datacomplete_MD = X_test_datacomplete.copy()

for i in list(X_test_datacomplete):
    
    if i in ['pid', 'syear', 'hid']:
        
        print('No within transformation for pid, year and hid')
    
    else:
    
        var_1_PD = pd.DataFrame(X_test_datacomplete_MD[['pid', i]].groupby(['pid']).mean())
    
        var_1_PD['pid'] = var_1_PD.index
    
        var_1_PD.reset_index(inplace = True, drop = True)
        
        X_test_datacomplete_MD = X_test_datacomplete_MD.merge(var_1_PD, how = 'inner', on = 'pid')
        
        cols_to_change = [i + '_x', i + '_y']
        
        PD_x = 'P[D]_' + i
        
        X_test_datacomplete_MD.rename(columns = {cols_to_change[0]: i,
                                     cols_to_change[1]: PD_x}, inplace = True)
        
        MD_x = 'M[D]_' + i
    
        X_test_datacomplete_MD[MD_x] =  X_test_datacomplete_MD[i] - X_test_datacomplete_MD[PD_x]

#Let's drop constant columns                 
        
for j in list(X_test_datacomplete_MD):
    
    if np.sum(X_test_datacomplete_MD[j] == 0) == len(X_test_datacomplete_MD):
        
        X_test_datacomplete_MD.drop([j], axis = 1, inplace = True)
        
#np.sum(y_test_datacomplete_PD_y['pid'] != X_test_datacomplete_MD['pid'])
#0
        
#np.sum(y_test_PD_y['pid'] != X_test_MD['pid'])
#0        

vars_no_PD_MD = [x for x in list(X_test_datacomplete_MD) if x[0:4] != 'P[D]' and x[0:4] != 'M[D]']

X_test_datacomplete_MD.drop(vars_no_PD_MD, axis = 1, inplace = True)

#list(X_train_datacomplete_MD) == list(X_test_datacomplete_MD)
#True 

#list(X_datacomplete_MD) == list(X_test_datacomplete_MD)
#True

#Standardization can now be done: this is necessary since K-Means clustering
#is based on Euclidean distances.

#########################################
##STANDARDIZE ENTIRE TRANSFORMED POOLED##
#########################################

X_datacomplete_MD_stand = pd.DataFrame(scaler.fit_transform(X_datacomplete_MD), index = y_datacomplete.index)

X_datacomplete_MD_stand_1 = pd.concat([X_datacomplete_MD_stand, pid_hid_all], axis = 1, ignore_index = True)

X_datacomplete_MD_stand_1.columns = list(X_datacomplete_MD) + ['pid', 'hid']

X_datacomplete_MD_stand_1.describe().T

pid_hid_all.describe().T

X_datacomplete_MD_stand_1.to_csv('C:\\Some\\Local\\Path\\X_datacomplete_MD_stand_1.csv')

y_datacomplete_PD_y.to_csv('C:\\Some\\Local\\Path\\y_datacomplete_PD_y.csv')

del pid_hid_all, X_datacomplete_ohed_stand, pid_hid

#In this case, differently from the entire Pooled specification, we also standardize 
#the training and test set from of entire Transformed Pooled datasets.

#While this has no effects on the predictive accuracy and interpretability
#of Linear Regressions and Tree-based algorthms, it may be fundamental if 
#I was to explore other algorithms sensitive to variables' magnitude.

#Moreover, it helps solving potential numerical issues, possibly arising from
#the large number of variables (loss of degree of freedom w.r.to the cluster
#sizes) and how they are defined.

##################################
##STANDARDIZE DATACOMPLETE TRAIN##
##################################

X_train_datacomplete_MD_stand = pd.DataFrame(scaler.fit_transform(X_train_datacomplete_MD), index = y_train_datacomplete.index)

X_train_datacomplete_MD_stand_1 = pd.concat([X_train_datacomplete_MD_stand, pid_hid_train], axis = 1, ignore_index = True)

X_train_datacomplete_MD_stand_1.columns = list(X_train_datacomplete_MD) + ['pid', 'hid']

X_train_datacomplete_MD_stand_1.describe().T

pid_hid_train.describe().T

X_train_datacomplete_MD_stand_1.to_csv('C:\\Some\\Local\\Path\\X_train_datacomplete_MD_stand_1.csv')

y_train_datacomplete_PD_y.to_csv('C:\\Some\\Local\\Path\\y_train_datacomplete_PD_y.csv') 

#################################
##STANDARDIZE DATACOMPLETE TEST##
#################################

X_test_datacomplete_MD_stand = pd.DataFrame(scaler.transform(X_test_datacomplete_MD), index = y_test_datacomplete.index)

X_test_datacomplete_MD_stand_1 = pd.concat([X_test_datacomplete_MD_stand, pid_hid_test], axis = 1, ignore_index = True)

X_test_datacomplete_MD_stand_1.columns = list(X_test_datacomplete_MD) + ['pid', 'hid']

X_test_datacomplete_MD_stand_1.describe().T

pid_hid_test.describe().T

X_test_datacomplete_MD_stand_1.to_csv('C:\\Some\\Local\\Path\\X_test_datacomplete_MD_stand_1.csv')

y_test_datacomplete_PD_y.to_csv('C:\\Some\\Local\\Path\\y_test_datacomplete_PD_y.csv')

############################
###K - MEANS - CLUSTERING###
############################

#Important note: at the time this code has been written, the paper
#Schubert, E., "Stop using the elbow criterion for k-means and 
#how to choose the number of clusters instead" had yet to be published
#and popularize. 

#Here, I still use the Elbow method nonetheless. In further checks robustness
#checks, I will consider the sensitivity of the overall findings against
#using different clustering criteria.

def k_means_clustering(X, clusters, threshold, if_graph = False):
    
    #Idea is to do one threshold at a time, and save the results as .csv.
    #I then import the .csvs separately and create the graph as how I was doing
    #for the average effects.
    
    distortions = {}
    
    labels = {} 
    
    kmeanModel = KMeans(n_clusters = 1)
    
    kmeanModel.fit(X)
        
    distortions[0] = kmeanModel.inertia_
    
    K = range(2, clusters)
    
    for k in K:
            
        kmeanModel = KMeans(n_clusters = k)
    
        kmeanModel.fit(X)
    
        distortions[k - 1] = kmeanModel.inertia_
        
        labels[k - 1] = kmeanModel.labels_
        
        if if_graph == True:
        
        #To make the graph more appealing, given an ex-ante fixed number of
        #clusters into the "clusters" parameter, simply letting go the function
        #to plot the graph.
        
        #That is, suppose that running this function in a previous moment (with 
        #if_graph == False, see else below) you have found the optimal 
        #number of clusters being 7. To show the elbow in the graph,
        #rerun the function setting the "clusters" parameter at, say, 20 and 
        #if_graph == True. The below will plot the behaviour of the distortions
        #against the number of clusters, showing an elbow at 7 and a plateau
        #in reduced distortions after that (in this example, 20).
        
           plt.figure()   
           plt.plot(list(distortions.keys()), list(distortions.values()), 'bx-', label = 'Inertia')
           plt.xlabel('k')
           plt.ylabel('Distortion')
           plt.title('The Elbow Method showing the optimal k')
           plt.show()
           
        else:
            
           #In this case, you're still looking for the optimal amount of clusters,
           #and not yet interested in plotting the behaviour. The elbow is found
           #uniquely numerically.
            
           if list(distortions.values())[k - 2] < (1 + threshold) * list(distortions.values())[k - 1]:
            
               break
    
           list_of_results = [list(distortions.values()), list(labels.values())]
    
    #The function returns:
    #1) The list of the (reduced) distortions in increasing the number of clusters (list_of_results[0])
    #2) The list of lists of labels of the clusters to which each individual belongs to 
    #in each of the iterations (list_of_results[1]).
    
    #That is, list_of_results[1][-1] includes the clusters' indexes at which
    #each individual belongs to in the last iteration, and list_of_results[0][-1]
    #is a list with length the number of clusters created in the last iteration,
    #and values the distortions - sum of squared disctances from the centroid -
    #in each of the clusters.
    
    #Hence, what I need for the graph, are only the values 
    #in list_of_results[0].
    
    return list_of_results

##############
##CLUSTERING##
##############

X_datacomplete_MD_stand_for_kmeans = X_datacomplete_MD_stand_1.drop(['pid', 'hid'], axis = 1)

clusters_MD = k_means_clustering(X = X_datacomplete_MD_stand_for_kmeans, 
                                     clusters = 30, 
                                     threshold = 0.05)

#pd.DataFrame(clusters_MD[0], columns = ['Inertia']).to_csv('C:\\Some\\Local\\Path\\inertias_mundlake_data_thresh_005_nostop.csv')

#################################################################
###CREATING THE CLUSTERS ON TRANSFORMED POOLED AND SAVING THEM###
#################################################################

X_datacomplete_MD_stand_1['cluster'] = clusters_MD[1][-1]

y_datacomplete_PD_y['cluster'] = clusters_MD[1][-1]

y_datacomplete_MD['cluster'] = clusters_MD[1][-1]

dataframes_per_cluster = {}

for i in list(X_datacomplete_MD_stand_1['cluster'].unique()):
    
    globals()["Cluster" + str(i)] = X_datacomplete_MD_stand_1[X_datacomplete_MD_stand_1['cluster'] == i]
    
for i in list(y_datacomplete_PD_y['cluster'].unique()):
    
    globals()["Cluster" + str(i) + 'dvisit_PD'] = y_datacomplete_PD_y[y_datacomplete_PD_y['cluster'] == i]    
    
for i in list(y_datacomplete_MD['cluster'].unique()):
    
    globals()["Cluster" + str(i) + 'dvisit_MD'] = y_datacomplete_MD[y_datacomplete_MD['cluster'] == i]    
    
for i in list(X_datacomplete_MD_stand_1['cluster'].unique()):
    
    Dataset_name = "Cluster" + str(i)
    
    save_path = 'C:\\Some\\Local\\Path\\' + Dataset_name + '.csv'
    
    globals()["Cluster" + str(i)].to_csv(save_path)

for i in list(y_datacomplete_PD_y['cluster'].unique()):
    
    Dataset_name = "Cluster" + str(i) + 'dvisit_PD'
    
    save_path = 'C:\\Some\\Local\\Path\\' + Dataset_name + '.csv'
    
    globals()["Cluster" + str(i) + 'dvisit_PD'].to_csv(save_path)
    
for i in list(y_datacomplete_MD['cluster'].unique()):
    
    Dataset_name = "Cluster" + str(i) + 'dvisit_MD'
    
    save_path = 'C:\\Some\\Local\\Path\\' + Dataset_name + '.csv'
    
    globals()["Cluster" + str(i) + 'dvisit_MD'].to_csv(save_path)
    
#######################################
####TRAIN - TEST SPLITS ON CLUSTERS####
#######################################
    
clusters_path = 'C:\\Some\\Local\\Path\\'

for i in [0,1,2]:
    
    X_i = globals()["Cluster_" + str(i)].drop(['Doctor Visits', 
                                               'Group Mean Doctor Visits',
                                               'Group-Demeaned Doctor Visits'], axis = 1)
    
    y_i = globals()["Cluster_" + str(i)][['Doctor Visits', 
                                         'Group Mean Doctor Visits',
                                         'Group-Demeaned Doctor Visits']] 
      
    seed = randint(0, 1000)    
    
    X_train, X_test, y_train, y_test = train_test_split(X_i, 
                                                        y_i,
                                                        test_size = 0.20,
                                                        random_state = seed) 
    
    train = pd.concat([y_train, X_train], axis = 1)
    
    test = pd.concat([y_test, X_test], axis = 1)
        
    save_train = clusters_path + '\\train_cluster_mundlaked_' + str(i) + '.csv'
    
    save_test = clusters_path + '\\test_cluster_mundlaked_' + str(i) + '.csv'
    
    train.to_csv(save_train, index = False)
    
    test.to_csv(save_test, index = False) 