# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:07:51 2018

@author: Home
"""


from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
import pickle


df=pd.read_csv("base_flask.csv",sep=',',low_memory=False)
df=df.drop(['Unnamed: 0'], axis=1)



''' chargement de l'arbre de décision qui permet d'effectuer la prédiction '''
arbre=pickle.load(open('arbre_decision_total.sav', 'rb'))






''' définition de la fonction qui va rechercher le vol le plus proche, via la distance Canberra
suivant les informations entrées par l'utilisateur '''



def donnees_vol(depart, arrivee, jour_mois, compagnie, heure,mois):
    
    
    df_donnees=df[(df.ORIGIN_CITY_NAME == depart) & (df.DEST_CITY_NAME == arrivee) & (df.CARRIER == compagnie) ]
    
        
    
    df_numerique=df_donnees.iloc[:,:27]
    
    df_numerique=df_numerique.reset_index()
    df_numerique=df_numerique.drop(['index'], axis=1)
    

    ligne=[mois,jour_mois,0,heure,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    df_ligne=pd.DataFrame([ligne], index=[100], columns=df_numerique.columns)
    df_total=df_numerique.append(df_ligne)
    df_total=df_total.fillna(0)
    
    matrice=np.array(df_total)
    
    
    dist_canb=DistanceMetric.get_metric("canberra")

    distance_canberra=dist_canb.pairwise(matrice,matrice)
    

    liste_canb=list(enumerate(distance_canberra[len(distance_canberra)-1]))
    liste_dist_canb=sorted(liste_canb, key=lambda x: x[1], reverse=False)
    
    liste_dist_canb=liste_dist_canb[1]
    ligne_vol_similaire=df_numerique.iloc[liste_dist_canb[0]]
    

    
    return ligne_vol_similaire
    




app = Flask(__name__)

''' options '''

#app.config.from_object('config')

''' page d'accueil '''

@app.route('/')
def home():
    return render_template("prediction.html")




@app.route('/', methods=['POST'])
def formulaire():
    text=request.form['text']
    #modif_text=text.title()
    
    arrivee = request.form['arrivee']
    #modif_arrivee = arrivee.title()
    
    compagnie=request.form['compagnie']
    #modif_compagnie = compagnie.upper()
    
    jour = request.form['jour']
    modif_jour = np.int(jour)
    
    mois = request.form['mois']
    #modif_mois = np.int(mois)
    modif_mois=1
 
    heure=request.form['horaire']
    modif_heure=np.int(heure)
    
    
    ''' appel à la fonction qui va chercher le Kppv du vol choisi par l'utilisateur '''
    
    ligne_vol=donnees_vol(text, arrivee, modif_jour, compagnie, modif_heure, modif_mois)
    
    heure_depart=np.int(ligne_vol.CRS_DEP_TIME)/100
    heure_depart=str(heure_depart)
    heure_depart=heure_depart.replace('.','h')
    
    ''' calcul du temps de retard via l'arbre de décision ''' 
    
    temps_retard=np.int(arbre.predict(ligne_vol.values.reshape(1,-1)))



    return render_template('prediction.html', jour_mois=np.int(ligne_vol.DAY_OF_MONTH), heure_depart=heure_depart, retard=temps_retard)




if __name__ == '__main__':
    app.run()
    
    
