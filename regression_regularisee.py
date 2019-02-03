# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:03:14 2018

@author: Jérôme
"""


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df=pd.read_csv('base_finale_ajout.csv',sep=',',low_memory=False)
#df=pd.read_csv('base_finale.csv',sep=',',low_memory=False)
#df=pd.read_csv('base_test_janvier.csv',sep=',',low_memory=False)
#df=pd.read_csv("base_janvier_deux.csv",sep=',',low_memory=False)

df=df.drop(['Unnamed: 0'], axis=1)





#¶df_numerique=df.iloc[:,[6,7,9,10,11,12,13,14,16,17,18,19]]
df_numerique=df.iloc[:,[0,1,2,6,7,9,10,11,12,13,14,15,17,18,19,20,21]]
#df_numerique=df.iloc[:,[0,1,2,10,11,13,14,15,16,17,20,21,22]]

 


matrice=np.array(df_numerique)
 
''' standardisation des variables pour éviter les différences d'échelle entre chacune '''
   
scale=preprocessing.StandardScaler().fit(matrice)
x=scale.transform(matrice)


y=df['ARR_DELAY'].values
    
    
    
liste_emq_ridge=[]
liste_emq_lasso=[]
liste_emq_elasticnet=[]
    
liste_r_ridge=[]
liste_r_lasso=[]
liste_r_elasticnet=[]
    

''' calcul des régressions régularisées sous forme d'une boucle pour effectuer plusieurs scissions
aléatoires de la base de données en test et apprentissage '''
    
for k in range(1,11):
        
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


    reg_ridge=Ridge(alpha=4, max_iter=100000)
    reg_ridge.fit(x_train,y_train)
    
    reg_lasso=Lasso(alpha=0.1, max_iter=100000)
    reg_lasso.fit(x_train,y_train)
    
    reg_elasticnet=ElasticNet(alpha=0.1,l1_ratio=0.9, max_iter=100000)
    reg_elasticnet.fit(x_train,y_train)
    
    
    
    emq_ridge=np.mean((reg_ridge.predict(x_train)-y_train)**2)
    emq_lasso=np.mean((reg_lasso.predict(x_train)-y_train)**2)
    emq_elasticnet=np.mean((reg_elasticnet.predict(x_train)-y_train)**2)
        

        
    liste_emq_ridge.append(emq_ridge)
    liste_emq_lasso.append(emq_lasso)
    liste_emq_elasticnet.append(emq_elasticnet)
        
    liste_r_ridge.append(reg_ridge.score(x_train,y_train))
    liste_r_lasso.append(reg_lasso.score(x_train,y_train))
    liste_r_elasticnet.append(reg_elasticnet.score(x_train,y_train))



''' affichage des moyennes obtenues pour les EMQ et R² '''

print("moyenne des emq en regression Ridge : ", np.mean(liste_emq_ridge))
print("moyenne des emq en regression Lasso : ", np.mean(liste_emq_lasso))
print("moyenne des emq en regression ElasticNet: ", np.mean(liste_emq_elasticnet))
    
print("moyenne des r² en regression Ridge :", np.mean(liste_r_ridge))
print("moyenne des r² en regression Lasso :", np.mean(liste_r_lasso))
print("moyenne des r² en regression ElasticNet :", np.mean(liste_r_elasticnet))

