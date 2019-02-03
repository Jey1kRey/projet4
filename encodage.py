# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:20:40 2018

@author: Home
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
#from sklearn.linear_model import LinearRegression



#df=pd.read_csv("base_complete.csv",sep=',', low_memory=False)
df=pd.read_csv("base_janvier.csv",sep=',',low_memory=False)

df=df.drop(['Unnamed: 0'], axis=1)

df_ville_origin=df['ORIGIN_AIRPORT_ID']
df_ville_dest=df['DEST_CITY_NAME']
df_cie=df['CARRIER']
df_categorique=df[['MONTH','DAY_OF_MONTH','DAY_OF_WEEK','CRS_DEP_TIME','CRS_ARR_TIME']]


y_final=df['ARR_DELAY'].values

''' compilation des données encodées '''


df_numerique=df.iloc[:,9:]


std_scale=preprocessing.StandardScaler().fit(df_numerique)
donnees=std_scale.transform(df_numerique)



acp=PCA()
acp.fit(donnees)

somme_variance=np.cumsum(acp.explained_variance_ratio_)
plt.plot(somme_variance)

composants=acp.components_

''' graphes des dimensions 1 et 2 : cela ne sera pas utilisé pour la suite du projet. On utilisera
R en partant du fichier de données remplis'''

for i,(x,y) in enumerate(zip(composants[0,:],composants[1,:])):
    plt.plot([0,x],[0,y],color='k')
    plt.text(x,y,df_numerique.columns[i],fontsize='14')
    
plt.plot([-0.5,0.5],[0,0],color='grey',ls='--')

plt.plot([0,0],[-0.5,0.5],color='grey',ls='--')

plt.xlim([-0.5,0.5])
plt.ylim([-0.5,0.5])
#plt.savefig("graphe_dim12.png",dpi=400)



