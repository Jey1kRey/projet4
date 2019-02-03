# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:42:05 2018

@author: Jérôme
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
import numpy as np
import pickle


df=pd.read_csv('base_finale.csv',sep=',',low_memory=False)
#df=pd.read_csv('base_test_janvier.csv',sep=',',low_memory=False)
df=df.drop(['Unnamed: 0'], axis=1)




''' encodage des variables potentiellement intéressantes : n'est conservé ici que les compagnies aériennes :
les autres variables qualitatives apportaient très peu d'informations supplémentaires ( modif de R² à 4 chiffres
après la virgule), pour des temps de calcul très long voire un débordement mémoire. '''

compagnies=LabelBinarizer()

df_cie=compagnies.fit_transform(df['CARRIER'])
df_onehot_cie=pd.DataFrame(df_cie,columns=["compagnie_"+str(int(i)) for i in range(df_cie.shape[1])])
df_numerique=df.iloc[:,[0,1,2,6,7,9,10,11,12,13,14,16,17,18,19]]
df_total=pd.concat([df_numerique,df_onehot_cie], axis=1)





''' création d'un fichier à exporter pour une utilisation dans l'API '''
df_origin=df['ORIGIN_CITY_NAME']
df_dest=df['DEST_CITY_NAME']
df_carrier=df['CARRIER']
df_supp=pd.concat([df_total, df_origin], axis=1)
df_supp=pd.concat([df_supp, df_dest], axis=1)
df_supp=pd.concat([df_supp, df_carrier], axis=1)
#df_supp.to_csv("base_flask.csv",sep=',')


''' définition de la variable y : le retard à l'arrivée '''
y=df['ARR_DELAY'].values




liste_r_train=[]
liste_r_test=[]

liste_emq=[]

''' On effectue une boucle sur plusieurs valeurs d'un indice k afin d'obtenir plusieurs
tests pour l'arbre de décision : on renvoie la moyenne des R² et des EMQ comme valeurs finales. '''

for k in range(1,11):
    
    x_train,x_test,y_train,y_test = train_test_split(df_total,y,test_size=0.3)
    
    arbre=DecisionTreeRegressor(max_depth=17)   


    arbre.fit(x_train,y_train)
    arbre.fit(x_test,y_test)
    
    
    ''' tests sur différents paramètres : recherche des critères optimums'''
    #arbre_minleaf=DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
    #arbre_dix=DecisionTreeRegressor(max_depth=10)
    #arbre_dix_minleaf=DecisionTreeRegressor(max_depth=10, min_samples_leaf=10)
    
    #arbre_minleaf.fit(x_train,y_train)
    #arbre_dix.fit(x_train,y_train)
    #arbre_dix_minleaf.fit(x_train,y_train)

    
    ''' export de l'arbre de décision sous forme de graphique pour visualisation '''
    #export_graphviz(arbre, out_file="arbre_quinze.dot",feature_names=df_total.columns)
    
    
    liste_r_train.append(arbre.score(x_train,y_train))
    liste_r_test.append(arbre.score(x_test,y_test))

    #print(" arbre avec min leaf de 10 :", arbre_minleaf.score(x_train,y_train))
    #print(" arbre avec profndeur de 10 :", arbre_dix.score(x_train,y_train))
    #print(" arbre avec profondeur de 10 et min leaf de 10 :", arbre_dix_minleaf.score(x_train,y_train))

    emq=np.mean((arbre.predict(x_train)-y_train)**2)
    liste_emq.append(emq)

print("moyenne R² pour la base apprentissage : ", np.mean(liste_r_train))
print("moyenne R² pour la base test :", np.mean(liste_r_test))
print("EMQ : ", np.mean(liste_emq))




''' sauvegarde du modèle de l'arbre de décision '''

#nom_fichier='arbre_decision_total.sav'
#pickle.dump(arbre, open(nom_fichier, 'wb'))
