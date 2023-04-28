#####################################
##NINTH SCRIPT - RESULTS AS GRAPHIC##
#####################################

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

'''
COMMENTS

This is the ninth and final script in producing the results in:
Gentile, N., "Healthcare utilization and its evolution during the years: 
building a predictive and interpretable model", 2022. Third chapter of my PhD
thesis and working paper.

Aim of this script is to present the results on both pooled and transformed pooled,
as well as in the clusters associated with both specifications, into nice bar charts.

Presented values are Test R2.

1) Pooled vs. Transformed Pooled (Mundlaked in the below):

- A four bars bar-chart: one couple for pooled (LR vs. RF) and the other for transformed pooled (LR vs. RF).

2) Clusters in Pooled:

- A ten bars bar-chrt: a couple (LR vs. RF) for each of the five clusters.

3) Clusters in Transformed Pooled (Mundlaked in the below):
    
 - A six bars bar-chrt: a couple (LR vs. RF) for each of the three clusters.

'''

path = 'C:\\Some\\Local\\Path\\'

pooled_results = pd.read_excel(path + 'All_results_100622_to_import_python_version.xlsx', sheet_name = 'pooled')

pooled_clusters_results = pd.read_excel(path + 'All_results_100622_to_import_python_version.xlsx', sheet_name = 'pooled_clusters')

mundlak_results = pd.read_excel(path + 'All_results_100622_to_import_python_version.xlsx', sheet_name = 'mundlak')

mundlak_clusters_results = pd.read_excel(path + 'All_results_100622_to_import_python_version.xlsx', sheet_name = 'mundlak_clusters')

################################
##POOLED AND MUNDLAKED RESULTS##
################################

N = 2
linregs = [pooled_results.iloc[0,3], mundlak_results.iloc[0,3]]
rfs = [pooled_results.iloc[1,3], mundlak_results.iloc[1,3]]

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, linregs, width, label = 'Linear Regression')
plt.bar(ind + width, rfs, width, label = 'Random Forest', color = 'r')

plt.ylabel('Test R2')

plt.xticks(ind + width/2, ('Pooled', 'Transformed Pooled'))
plt.legend()
plt.show()


###########################
##POOLED CLUSTERS RESULTS##
###########################


N = 5

linregs = [pooled_clusters_results.iloc[0,3], pooled_clusters_results.iloc[2,3],
           pooled_clusters_results.iloc[4,3], pooled_clusters_results.iloc[6,3],
           pooled_clusters_results.iloc[8,3]]

rfs = [pooled_clusters_results.iloc[1,3], pooled_clusters_results.iloc[3,3],
       pooled_clusters_results.iloc[5,3], pooled_clusters_results.iloc[7,3],
       pooled_clusters_results.iloc[9,3]]

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, linregs, width, label = 'Linear Regression')
plt.bar(ind + width, rfs, width, label = 'Random Forest', color = 'r')

plt.ylabel('Test R2')

plt.xticks(ind + width/2, ('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'))
plt.legend()
plt.show()


############################
##MUNDLAK CLUSTERS RESULTS##
############################

N = 3

linregs = [mundlak_clusters_results.iloc[0,3], mundlak_clusters_results.iloc[2,3],
           mundlak_clusters_results.iloc[4,3]]

rfs = [mundlak_clusters_results.iloc[1,3], mundlak_clusters_results.iloc[3,3],
       mundlak_clusters_results.iloc[5,3]]

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, linregs, width, label = 'Linear Regression')
plt.bar(ind + width, rfs, width, label = 'Random Forest', color = 'r')

plt.ylabel('Test R2')

plt.xticks(ind + width/2, ('Cluster 0', 'Cluster 1', 'Cluster 2'))
plt.legend()
plt.show()




##############################################
###GRAPHS ON CLUSTERS FROM ABLATION STUDIES###
##############################################

pooled_clusters_results_1 = pd.read_excel(path + 'Results_ablation_studies_200722.xlsx', sheet_name = 'Pooled_clusters')

mundlak_clusters_results_1 = pd.read_excel(path + 'Results_ablation_studies_200722.xlsx', sheet_name = 'Mundlak_clusters')

###########################################################################
##GRAPH OF THE WEIGHTED AVERAGE TEST R2 ACROSS CLUSTERS IN THE FOUR CASES##
###########################################################################

#The aim is to create TWO two-ticks bar charts (clusters from pooled and clusters
#from transformed pooled) with two groups of four bars respectively, representing:

#1) Weighted Average Test R2 of LinReg over the 5 (3) clusters 
#2) Weighted Average Test R2 of LinReg over the 5 (3) clusters without SRH
#3) Weighted Average Test R2 of RF over the 5 (3) clusters 
#4) Weighted Average Test R2 of RF over the 5 (3) clusters without SRH

#5) Weighted Average Test R2 of LinReg over the 5 (3) clusters 
#6) Weighted Average Test R2 of LinReg over the 5 (3) clusters without Dis. and Physc.
#7) Weighted Average Test R2 of RF over the 5 (3) clusters 
#8) Weighted Average Test R2 of RF over the 5 (3) clusters without Dis. and Physc.

#The weights of the five pooled clusters are:
#0.1774, 0.2479, 0.2771, 0.2421, 0.0555.

#The weights of the three mundlak clusters are:
#0.2823, 0.3897, 0.3280

wa_test_r2_linreg_pooled = 0.1774 * pooled_clusters_results.iloc[0,3] + 0.2479 * pooled_clusters_results.iloc[2,3] + 0.2771 * pooled_clusters_results.iloc[4,3] + 0.2421 * pooled_clusters_results.iloc[6,3] + 0.0555 * pooled_clusters_results.iloc[8,3]

#0.13922218522757399

wa_test_r2_linreg_pooled_no_srh = 0.1774 * pooled_clusters_results_1.iloc[0,1] + 0.2479 * pooled_clusters_results_1.iloc[2,1] + 0.2771 * pooled_clusters_results_1.iloc[4,1] + 0.2421 * pooled_clusters_results_1.iloc[6,1] + 0.0555 * pooled_clusters_results_1.iloc[8,1] 

#0.11595825530158738

wa_test_r2_rf_pooled = 0.1774 * pooled_clusters_results.iloc[1,3] + 0.2479 * pooled_clusters_results.iloc[3,3] + 0.2771 * pooled_clusters_results.iloc[5,3] + 0.2421 * pooled_clusters_results.iloc[7,3] + 0.0555 * pooled_clusters_results.iloc[9,3]

#0.19905375969079322

wa_test_r2_rf_pooled_no_srh = 0.1774 * pooled_clusters_results_1.iloc[1,1] + 0.2479 * pooled_clusters_results_1.iloc[3,1] + 0.2771 * pooled_clusters_results_1.iloc[5,1] + 0.2421 * pooled_clusters_results_1.iloc[7,1] + 0.0555 * pooled_clusters_results_1.iloc[9,1] 

#0.16035673045762527

wa_test_r2_linreg_pooled_no_dis_no_physc = 0.1774 * pooled_clusters_results_1.iloc[11,1] + 0.2479 * pooled_clusters_results_1.iloc[13,1] + 0.2771 * pooled_clusters_results_1.iloc[15,1] + 0.2421 * pooled_clusters_results_1.iloc[17,1] + 0.0555 * pooled_clusters_results_1.iloc[19,1] 

#0.11987280234696363

wa_test_r2_rf_pooled_no_dis_no_physc = 0.1774 * pooled_clusters_results_1.iloc[12,1] + 0.2479 * pooled_clusters_results_1.iloc[14,1] + 0.2771 * pooled_clusters_results_1.iloc[16,1] + 0.2421 * pooled_clusters_results_1.iloc[18,1] + 0.0555 * pooled_clusters_results_1.iloc[20,1] 

#0.16745857179828716




wa_test_r2_linreg_mundlak = 0.2823 * mundlak_clusters_results.iloc[0,3] + 0.3897 * mundlak_clusters_results.iloc[2,3] + 0.3280 * mundlak_clusters_results.iloc[4,3] 

#0.12169827683259576

wa_test_r2_linreg_mundlak_no_srh = 0.2823 * mundlak_clusters_results_1.iloc[0,1] + 0.3897 * mundlak_clusters_results_1.iloc[2,1] + 0.3280 * mundlak_clusters_results_1.iloc[4,1]

#0.09793954700213105

wa_test_r2_rf_mundlak = 0.2823 * mundlak_clusters_results.iloc[1,3] + 0.3897 * mundlak_clusters_results.iloc[3,3] + 0.3280 * mundlak_clusters_results.iloc[5,3] 

#0.20686462890636417

wa_test_r2_rf_mundlak_no_srh = 0.2823 * mundlak_clusters_results_1.iloc[1,1] + 0.3897 * mundlak_clusters_results_1.iloc[3,1] + 0.3280 * mundlak_clusters_results_1.iloc[5,1] 

#0.1803012653517642

wa_test_r2_linreg_mundlak_no_dis_no_physc = 0.2823 * mundlak_clusters_results_1.iloc[7,1] + 0.3897 * mundlak_clusters_results_1.iloc[9,1] + 0.3280 * mundlak_clusters_results_1.iloc[11,1] 

#0.11038319437700775

wa_test_r2_rf_mundlak_no_dis_no_physc = 0.2823 * mundlak_clusters_results_1.iloc[8,1] + 0.3897 * mundlak_clusters_results_1.iloc[10,1] + 0.3280 * mundlak_clusters_results_1.iloc[12,1] 

#0.20312338450691358


###################################
##ABLATION STUDY: POOLED CLUSTERS##
###################################

N = 2

lirneg_pure = [wa_test_r2_linreg_pooled, wa_test_r2_linreg_pooled]

linregs_ablated = [wa_test_r2_linreg_pooled_no_srh, wa_test_r2_linreg_pooled_no_dis_no_physc]

rf_pure = [wa_test_r2_rf_pooled, wa_test_r2_rf_pooled]

rf_ablated = [wa_test_r2_rf_pooled_no_srh, wa_test_r2_rf_pooled_no_dis_no_physc]

ind = np.arange(N) 

width = 0.175
plt.figure(figsize = (10.2, 4.8))       
plt.bar(ind, lirneg_pure, width, label = 'Lin. Reg.', color = 'g')
plt.bar(ind + width, linregs_ablated, width, label = 'Lin. Reg. ablated', color = 'y')
plt.bar(ind + 2*width, rf_pure, width, label = 'RF.', color = 'r')
plt.bar(ind + 3*width, rf_ablated, width, label = 'RF. ablated', color = 'black')


plt.xticks(ind + 1.5*width, ('Full models vs. No Self-Rated Health', 'Full models vs. No Disability, No Phys. Scale'))
plt.legend()
plt.show()


####################################
##ABLATION STUDY: MUNDLAK CLUSTERS##
####################################

N = 2

lirneg_pure = [wa_test_r2_linreg_mundlak, wa_test_r2_linreg_mundlak]

linregs_ablated = [wa_test_r2_linreg_mundlak_no_srh, wa_test_r2_linreg_mundlak_no_dis_no_physc]

rf_pure = [wa_test_r2_rf_mundlak, wa_test_r2_rf_mundlak]

rf_ablated = [wa_test_r2_rf_mundlak_no_srh, wa_test_r2_rf_mundlak_no_dis_no_physc]

ind = np.arange(N) 

width = 0.175
plt.figure(figsize = (10.2, 4.8))       
plt.bar(ind, lirneg_pure, width, label = 'Lin. Reg.', color = 'g')
plt.bar(ind + width, linregs_ablated, width, label = 'Lin. Reg. ablated', color = 'y')
plt.bar(ind + 2*width, rf_pure, width, label = 'RF.', color = 'r')
plt.bar(ind + 3*width, rf_ablated, width, label = 'RF. ablated', color = 'black')


plt.xticks(ind + 1.5*width, ('Full models vs. No Self-Rated Health', 'Full models vs. No Disability, No Phys. Scale'))
plt.legend()
plt.show()
