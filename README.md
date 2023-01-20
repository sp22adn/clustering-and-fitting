# clustering-and-fitting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import warnings
from matplotlib import style

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
# %matplotlib inline

df_data=pd.read_csv('API_4_DS2_en_csv_v2_4775247.csv',skiprows=4) # reading the data

df_data.head(5)

df_data.describe() # a statistica description of the data

#drop all null rows
df_data.dropna(how='all')

# importing necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

"""# 1"""

# selecting the data with the indicator School enrollment, tertiary, female (% gross)
df_data_1=df_data[df_data['Indicator Code']=='SE.TER.ENRR.FE']#'PA.NUS.ATLS']#'SP.POP.GROW']  #CO2 emissions (kg per PPP $ of GDP)

df_data_1.fillna(0,inplace=True)

df_data_1a=pd.DataFrame()

# creating clusters for the years 1995 and 2012.
df_data_1a['1993']=df_data_1['1993'].copy()
df_data_1a['2016']=df_data_1['2016'].copy()

# resetting the column index
df_data_1a.reset_index(drop=True)

from sklearn.preprocessing import StandardScaler
df_data_1a_colnam=df_data_1a.columns.values.tolist()

#create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_df_data_1 = StandardScaler().fit_transform(df_data_1a.to_numpy())

#creating the dataframe
scaled_df_data_1=pd.DataFrame(scaled_df_data_1, columns=[df_data_1a_colnam])

# changing the datatype
scaled_df_data_1 = scaled_df_data_1.astype(float)

n_clust_1=2
kmeans=KMeans(n_clusters=n_clust_1,random_state=42)

# running k means clustering
kmeans=kmeans.fit(scaled_df_data_1)

#creating a duplicate dataframe
scaled_df_data_1a=scaled_df_data_1

# creating cluster ids
scaled_df_data_1a['clust_id']=kmeans.predict(scaled_df_data_1)

# deterining the labels 
labels = kmeans.labels_

#finding the centers of the clusters
cen = kmeans.cluster_centers_
print(cen)
import sklearn.metrics as skmet

# calculate the silhoutte score
print(skmet.silhouette_score(scaled_df_data_1a, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

for l in range(n_clust_1): # loop over the different labels
  plt.scatter(scaled_df_data_1a[labels==l][df_data_1a_colnam[0]], scaled_df_data_1a[labels==l][df_data_1a_colnam[1]])

# # show cluster centres
for ix in range(n_clust_1):
  xc, yc = cen[ix,:]
  plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel(df_data_1a_colnam[0])
plt.ylabel(df_data_1a_colnam[1])
plt.title('Clusters for the years 1995 and 2012 for the indicator School enrollment, tertiary, female (% gross)')
plt.show()

"""From the clusters we can observe that there is an increase in female enrollment for schooling in almost all countries.
# 2
"""

# selecting the data with the indicator Educational attainment, at least completed short-cycle tertiary, population 25+, total (%) (cumulative)
df_data_2=df_data[df_data['Indicator Code']=='SE.TER.CUAT.ST.ZS']#'NY.ADJ.NNTY.PC.KD.ZG']#'PA.NUS.ATLS']#'SP.POP.GROW']  #CO2 emissions (kg per PPP $ of GDP)

df_data_2.fillna(0,inplace=True)

df_data_2a=pd.DataFrame()
# creating clusters for the years 1995 and 2012.
df_data_2a['1995']=df_data_2['1995'].copy()
df_data_2a['2019']=df_data_2['2019'].copy()
df_data_2a.reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
df_data_2a_colnam=df_data_2a.columns.values.tolist()

#create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_df_data_2 = StandardScaler().fit_transform(df_data_2a.to_numpy())

#creating the dataframe
scaled_df_data_2=pd.DataFrame(scaled_df_data_2, columns=[df_data_2a_colnam])
# changing the datatype
scaled_df_data_2 = scaled_df_data_2.astype(float)

n_clust_2=3
kmeans=KMeans(n_clusters=n_clust_2,random_state=42)
# running k means clustering
kmeans=kmeans.fit(scaled_df_data_2)
#creating a duplicate dataframe
scaled_df_data_2a=scaled_df_data_2
# creating cluster ids
scaled_df_data_2a['clust_id']=kmeans.predict(scaled_df_data_2)

# deterining the labels 
labels = kmeans.labels_
#finding the centers of the clusters
cen = kmeans.cluster_centers_

import sklearn.metrics as skmet
# calculate the silhoutte score
print(skmet.silhouette_score(scaled_df_data_1a, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

for l in range(n_clust_2): # loop over the different labels
  plt.scatter(scaled_df_data_2a[labels==l][df_data_2a_colnam[0]], scaled_df_data_2a[labels==l][df_data_2a_colnam[1]])

# # show cluster centres
for ix in range(n_clust_2):
  xc, yc = cen[ix,:]
  plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel(df_data_2a_colnam[0])
plt.ylabel(df_data_2a_colnam[1])
plt.title('Clusters for the years 1995 and 2012 for the indicator-Educational attainment, at least completed short-cycle tertiary, population 25+, total (%) (cumulative)')
plt.show()
