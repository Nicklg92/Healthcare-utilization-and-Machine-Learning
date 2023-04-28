##################################################################################
###FIRST SCRIPT - DATA ENGINEERING AND K-MEANS CLUSTERING, POOLED SPECIFICATION###
##################################################################################

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
    
A part from a first R script  in which the variables were extracted and renamed, 
this is the first script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

The goal of the paper is to predict and understand the determinants of
healthcare utilization, measured as Number of doctor visits in the last
three months.

In this script, I produce summary statistics and perform necessary data
cleaning and engineering operations on what in the paper is identified
as the "Pooled specifications".

On top of these, I also create the five clusters of individual using (the
Unsupervised Learning algorithm) K-Means.
'''

path = 'C:\\Some\\Local\\Path\\'

datacomplete = pd.read_csv(path + 'datacomplete_1.csv', sep = ';')

datacomplete.shape

#208903 x 19

list(datacomplete)

#How many different years are there?

datacomplete['syear'].unique()

#11 years. How many observations in each year?

for i in datacomplete['syear'].unique().tolist():
    
    print([i, len(datacomplete[datacomplete['syear'] == i])])
    

#[2004, 16801],[2005, 17179],[2006, 18881],[2007, 19177],[2008, 18321],
#[2009, 17231],[2010, 19721],[2011, 22400],[2012, 20640],[2013, 19650],
#[2014, 18902]

#In the initial specification, I thought also including as predictor
#the variable "Whether the individual ever had a chronic disease".

#However, it would intuitively become an overwhelming predictor for
#healthcare utilization, hence I have decided to drop it.

datacomplete.drop(['everchronicill'], axis = 1, inplace = True)

#############################
###MISSING VALUES ANALYSIS###
#############################

#How many missing values are there in each year, and which fraction 
#of the year do they represent?

for i in datacomplete['syear'].unique().tolist():
    
    print([i, 
           datacomplete[datacomplete['syear'] == i].isna().sum().sum(), 
           round(datacomplete[datacomplete['syear'] == i].isna().sum().sum() / (len(datacomplete[datacomplete['syear'] == i]) * len(list(datacomplete[datacomplete['syear'] == i]))), 4) * 100])
    

#[2004, 1093, 0.36]
#[2005, 2262, 0.73]
#[2006, 876, 0.26]
#[2007, 1716, 0.5]
#[2008, 634, 0.19],
#[2009, 1140, 0.37]
#[2010, 8200, 2.31]
#[2011, 15910, 3.95]
#[2012, 14287, 3.85]
#[2013, 14207, 4.02]
#[2014, 696, 0.2]

#Overall, manageable numbers. In which variables are there still missing?

print(datacomplete.isna().sum())


#disabled            954
#eversmoke         27971
#masmoke           27971
#insured             732
#alone               886
#isced              2507

#Are these variable numerical continuous or categorical/discrete?

for i in ['disabled', 'eversmoke', 'masmoke', 'insured', 'alone', 'isced']:
    
    print([i, datacomplete[i].nunique(dropna = False)])
    

#['disabled', 70]
#['eversmoke', 3]
#['masmoke', 371]
#['insured', 4]
#['alone', 3]
#['isced', 5]

#And which fraction of individuals are missing in each of these?

for i in ['disabled', 'eversmoke', 'masmoke', 'insured', 'alone', 'isced']:

    print([i, datacomplete[i].isna().sum(), round((datacomplete[i].isna().sum() / len(datacomplete)) * 100, 2)])
    

#['disabled', 954, 0.46]
#['eversmoke', 27971, 13.39]
#['masmoke', 27971, 13.39]
#['insured', 732, 0.35]
#['alone', 886, 0.42]
#['isced', 2507, 1.20]

#Given the relatively low proportions of missingness (with smoking habits only
#reaching the 13% mark), the specific imputation technique won't strongly 
#affect the results. For this reason, to simplify the data-engineering process,
#I simply consider the mean or mode (preserving the nature of the variable). 

print('Eversmokes mode is: ' + str(datacomplete['eversmoke'].mode()[0]))
#Eversmokes mode is: 0.0

datacomplete['eversmoke'].fillna(datacomplete['eversmoke'].mode()[0], inplace = True)

print('insured mode is: ' + str(datacomplete['insured'].mode()[0]))
#insured mode is: 1.0

datacomplete['insured'].fillna(datacomplete['insured'].mode()[0], inplace = True)

print('alone mode is: ' + str(datacomplete['alone'].mode()[0]))
#alone mode is: 0.0

datacomplete['alone'].fillna(datacomplete['alone'].mode()[0], inplace = True)

print('isced mode is: ' + str(datacomplete['isced'].mode()[0]))
#isced mode is: SE

datacomplete['isced'].fillna(datacomplete['isced'].mode()[0], inplace = True)

print('disabled mean is: ' + str(datacomplete['disabled'].mean()))
#disabled mean is: 7.652761013517738

datacomplete['disabled'].fillna(datacomplete['disabled'].mean(), inplace = True)

print('masmoke mean is: ' + str(datacomplete['masmoke'].mean()))
#masmoke mean is: 3.232013750496112

datacomplete['masmoke'].fillna(datacomplete['masmoke'].mean(), inplace = True)

datacomplete.isna().sum().sum()

#0

#For how the variables are engineered, there are no missing values that
#are negative values.

##########################
##DESCRIPTIVE STATISTICS##
##########################

#First, need to assess how my variables are organized: 
#How many are multiclass categorical?

for i in list(datacomplete):
    
    print([i, datacomplete[i].nunique()])

#['pid', 28046]
#['syear', 11]
#['hid', 17789]
#['dvisit', 66]
#['selfahealthimp', 5],
#['disabled', 70]
#['eversmoke', 2]
#['masmoke', 371]
#['bmiimp2', 640]
#['insured', 3]
#['alone', 2]
#['workstat', 5]
#['isced', 4]
#['hhincap', 70276]
#['gender', 2]
#['age', 89]
#['PsyScaleimp2', 117001]
#['PhyScaleimp2', 117022]

#The variables to be trated as categorical are insured, isced and workstat.

####################
##ONE HOT ENCODING##
####################

ohed_workstat = pd.get_dummies(datacomplete['workstat'])

ohed_isced = pd.get_dummies(datacomplete['isced'])

datacomplete_ohed = pd.concat([datacomplete, ohed_isced, ohed_workstat], axis = 1)

datacomplete_ohed.drop(['workstat', 'isced'], axis = 1, inplace = True)

#I now have have both datasets.

datacomplete_ohed.shape

#(208903, 25)

del ohed_isced, ohed_workstat

#The variable insured is not one-hot-encoded given its ordinal nature:
#indeed, each of its values represents an increasing amount of coverage
#by the insurance. To preserve this information, is better to leave it
#as it is.

########################
##CORRELATION ANALYSIS##
########################

corr_matrix = np.corrcoef(datacomplete_ohed, rowvar = False)

corr_matrix_pd = pd.DataFrame(corr_matrix, columns = list(datacomplete_ohed), index = list(datacomplete_ohed))

corr_matrix_pd.to_csv('C:\\Some\\Local\\Path\\Correlation_matrix.csv')

#The reference category for workstat should be 'Pension', becuase of its 
#73% correlation with age, -46% with PhyScaleimp2 and -46% with FT (Full-Time).

datacomplete_ohed.drop(['Pension'], axis = 1, inplace = True)

#For Education, let's drop SE (Secondary Education), which looks to 
#be the most collinear.

datacomplete_ohed.drop(['SE'], axis = 1, inplace = True)

datacomplete_ohed.shape

##############################################
###TRAIN TEST SPLITS AND K-MEANS CLUSTERING###
##############################################

datacomplete_ohed.to_csv('C:\\Some\\Local\\Path\\Datacomplete_ohed_imputed.csv')

X_datacomplete = datacomplete_ohed.drop(['dvisit'], axis = 1)

y_datacomplete = datacomplete_ohed['dvisit']

X_train, X_test, y_train, y_test = train_test_split(X_datacomplete, 
                                                    y_datacomplete,
                                                    test_size = 0.20) 
    
train = pd.concat([y_train, X_train], axis = 1)
    
test = pd.concat([y_test, X_test], axis = 1)

train.to_csv('C:\\Some\\Local\\Path\\Datacomplete_ohed_imputed_train.csv')

test.to_csv('C:\\Some\\Local\\Path\\Datacomplete_ohed_imputed_test.csv')

############################
###K - MEANS - CLUSTERING###
############################

#Important note: at the time this code had been written, the paper
#Schubert, E., "Stop using the elbow criterion for k-means and 
#how to choose the number of clusters instead" had yet to be published
#and popularize. 

#Here, hence, I still use the Elbow method. In further robustness
#checks, I will consider the sensitivity of the overall findings against
#using different clustering criteria.

#As will be noticed down the road, even using the Elbow method leads
#to clusters producing interesting insights.

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

######################################
###STANDARDIZATION OF ENTIRE DATASET##
######################################

#K-Means clustering is an algorithm based on Euclidean distances,
#hence variables' standardization is necessary ex-ante (otherwise the
#whole clustering procedure would just be driven by greater-magnitude
#variables, independently from their real contribution).

#Since standardization preserves the order of the observations, it won't be
#difficult then to get back to the oroginal years/ values.

pid_hid = X_datacomplete_ohed[['pid', 'hid']]

X_datacomplete_ohed_to_stand = X_datacomplete_ohed.drop(['pid', 'hid'], axis = 1)

X_datacomplete_ohed_stand = pd.DataFrame(scaler.fit_transform(X_datacomplete_ohed_to_stand), index = y_datacomplete_ohed.index)

X_datacomplete_ohed_stand_1 = pd.concat([X_datacomplete_ohed_stand, pid_hid], axis = 1, ignore_index = True)

X_datacomplete_ohed_stand_1.columns = list(X_datacomplete_ohed_to_stand) + ['pid', 'hid']

X_datacomplete_ohed_stand_1.describe().T

pid_hid.describe().T

del X_datacomplete_ohed_to_stand, X_datacomplete_ohed_stand, pid_hid

#The clustering procedure is performed on the entire dataset.

X_datacomplete_ohed_stand_1_for_kmeans = X_datacomplete_ohed_stand_1.drop(['pid', 'hid'], axis = 1)

#These values have been found via comparison of resulting number of clusters
#and considered threshold. Using a threshold 0.075, leading to the five
#clusters mentioned in the paper, represented the best compromise.

#Here, clusters = 30 since the code is used to produce the graph (see above).
#The clutsers had been produced in a previous phase.

##############
##CLUSTERING##
##############

clusters_soep = k_means_clustering(X = X_datacomplete_ohed_stand_1_for_kmeans, 
                                       clusters = 30, 
                                       threshold = 0.075,
                                       if_graph = True)

len(clusters_soep[0])


X_datacomplete_ohed_stand_1['cluster'] = clusters_soep[1][-1]

y_datacomplete_ohed['cluster'] = clusters_soep[1][-1]

#How many individuals in each of the clusters?

for i in list(X_datacomplete_ohed_stand_1['cluster'].unique()):
    
    print([i, len(X_datacomplete_ohed_stand_1[X_datacomplete_ohed_stand_1['cluster'] == i])])

#[0, 37057]
#[1, 51799]
#[2, 57886]
#[3, 50573] 
#[4, 11588] 

#with 0.075 threshold.

#Now, the key question: in each of the clusters, how many different years?

for i in [0,1,2,3,4]:
    
    print([i, globals()["Cluster_" + str(i)]['syear'].nunique(), len(globals()["Cluster_" + str(i)])])
    
#[0, 11, 37057] 
#[1, 11, 51799]
#[2, 11, 57886] 
#[3, 11, 50573]
#[4, 11, 11588]

#In all clusters there are all the years! This means that the variable 'year' 
#has no much impact in defining the clusters: in turn, this means that people
#tend to be repeated in each cluster. This seems a confirmation of the
#presence of individual - time constant effects.

#########################
###SAVING THE CLUSTERS###
#########################

dataframes_per_cluster = {}

for i in list(X_datacomplete_ohed_stand_1['cluster'].unique()):
    
    globals()["Cluster" + str(i)] = X_datacomplete_ohed_stand_1[X_datacomplete_ohed_stand_1['cluster'] == i]
    
for i in list(y_datacomplete_ohed['cluster'].unique()):
    
    globals()["Cluster" + str(i) + 'dvisit'] = y_datacomplete_ohed[y_datacomplete_ohed['cluster'] == i]    
    
for i in list(X_datacomplete_ohed_stand_1['cluster'].unique()):
    
    Dataset_name = "Cluster" + str(i)
    
    save_path = 'C:\\Some\\Local\\Path\\' + Dataset_name + '.csv'
    
    globals()["Cluster" + str(i)].to_csv(save_path)

for i in list(y_datacomplete_ohed['cluster'].unique()):
    
    Dataset_name = "Cluster" + str(i) + 'dvisit'
    
    save_path =  'C:\\Some\\Local\\Path\\' + Dataset_name + '.csv'
    
    globals()["Cluster" + str(i) + 'dvisit'].to_csv(save_path)

#######################################
####TRAIN - TEST SPLITS ON CLUSTERS####
#######################################
    
clusters_path = 'C:\\Some\\Local\\Path\\'

for i in [0,1,2,3,4]:
    
    X_i = globals()["Cluster_" + str(i)].drop(['dvisit'], axis = 1)
    
    y_i = globals()["Cluster_" + str(i)]['dvisit'] 
      
    seed = randint(0, 1000)    
    
    X_train, X_test, y_train, y_test = train_test_split(X_i, 
                                                        y_i,
                                                        test_size = 0.20,
                                                        random_state = seed) 
    
    train = pd.concat([y_train, X_train], axis = 1)
    
    test = pd.concat([y_test, X_test], axis = 1)
        
    save_train = clusters_path + '\\train_cluster_' + str(i) + '.csv'
    
    save_test = clusters_path + '\\test_cluster_' + str(i) + '.csv'
    
    #train.to_csv(save_train, index = False)
    
    #test.to_csv(save_test, index = False) 




