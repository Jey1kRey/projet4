# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:20:40 2018

@author: Home
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('base_finale.csv',sep=',',low_memory=False)
#df=pd.read_csv("base_complete.csv",sep=',', low_memory=False)
#df=pd.read_csv("base_janvier_deux.csv",sep=',',low_memory=False)

df=df.drop(['Unnamed: 0'], axis=1)

liste_reg_lineaire=[]
liste_emq_reg_lineaire=[]


    
df_numerique=df.iloc[:,[0,1,2,6,7,9,10,11,12,13,14,16,17,18,19]]
    
matrice=np.array(df_numerique)
std_scale=preprocessing.StandardScaler().fit(df_numerique)
donnees=std_scale.transform(df_numerique)
 

 
''' mise en place de l'ACP : on garde trois composantes : tests effectués sur le graphe de la variance cumulée '''
  
acp=PCA(n_components=3)
acp.fit(donnees)


''' calcul de la variance cumulée des nouvelles composantes '''
#somme_variance=np.cumsum(acp.explained_variance_ratio_)

''' graphe du pourcentage de variance pour la détermination du nombre de comp principales'''
#plt.plot(somme_variance)


composants=acp.components_
#transpose_composant=composants.T
#df_acp=np.dot(matrice,transpose_composant)
df_acp=acp.fit_transform(donnees)

''' graphes des variables sur le plan factoriel( ici, dimension 1 et 2)

for i,(x,y) in enumerate(zip(composants[0,:],composants[1,:])):
    plt.plot([0,x],[0,y],color='k')
    plt.text(x,y,df_numerique.columns[i],fontsize='14')
    
plt.plot([-0.5,0.5],[0,0],color='grey',ls='--')

plt.plot([0,0],[-0.5,0.5],color='grey',ls='--')

plt.xlim([-0.5,0.5])
plt.ylim([-0.5,0.5])
#plt.savefig("graphe_dim12.png",dpi=400)
'''



y=df['ARR_DELAY'].values

scale=preprocessing.StandardScaler().fit(df_acp)
x=scale.transform(df_acp)


''' boucle qui permet d'effectuer des calculs sur plusieurs bases tests et apprentissages pour limiter
l'impact de valeurs aberrantes '''

for k in range(1,11):
        

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


    reg_lineaire=LinearRegression()

    reg_lineaire.fit(x_train,y_train)

    emq_reglineaire=np.mean((reg_lineaire.predict(x_train)-y_train)**2)
    
    liste_emq_reg_lineaire.append(emq_reglineaire)
    liste_reg_lineaire.append(reg_lineaire.score(x_train,y_train))
    

''' on affiche la moyenne des résultats obtenus par les boucles précédentes '''
     
print("pour compagnie,",x,", moyenne emq regression lineaire :",np.mean(liste_emq_reg_lineaire))
print("pour compagnie,",x,", moyenne r² regression lineaire :",np.mean(liste_reg_lineaire))

