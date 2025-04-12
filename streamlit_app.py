# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:56:25 2023

@author: PERFECTO
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib
import shap
import lightgbm

from xplotter.insights import plot_aggregation
from lightgbm import LGBMClassifier
from streamlit_shap import st_shap


# Pour  initialiser les bibliothèques JavaScript de SHAP 
shap.initjs()


#st.set_option('deprecation.showPyplotGlobalUse', False)
#PATH = 'C:/Users/PERFECTO/PROJET_7_Streamlit/'

# Chargeons le dataset
X_test = pd.read_csv("X_test_feat_new.csv")

# Chargeons le modèle
mon_best_model = joblib.load("LE_PROJET_7_MODELE")

## La prédiction
y_pred_model = mon_best_model.predict(X_test)
y_pred_model_df = pd.DataFrame(y_pred_model, columns=['y_pred_test'])

# y_pred_model_proba est la probabilité que l'instance de données appartienne à 
#chaque classe.
y_pred_model_proba = mon_best_model.predict_proba(X_test)

## y_pred_model_proba en DataFrame
y_pred_model_proba_df = pd.DataFrame(y_pred_model_proba, columns=['proba_classe_0', 'proba_classe_1'])


#Les valeurs SHAP montrent comment chaque fonctionnalité affecte chaque 
#prédiction finale, l'importance de chaque fonctionnalité par rapport aux 
#autres et la dépendance du modèle à l'égard de l'interaction entre les 
#fonctionnalités.

#Les valeurs SHAP quantifient la contribution de chaque caractéristique (feature)
#à la prédiction d'un modèle de machine learning, en se basant sur la théorie 
#des jeux de Shapley.

explainer = shap.TreeExplainer(mon_best_model)
shap_values = explainer.shap_values(X_test)

# shap_values_df correspond au DataFrame issu de shap_values
#shap_values_df = pd.DataFrame(data=shap_values[1], columns=X_test.columns)
shap_values_df = pd.DataFrame(data=shap_values, columns=X_test.columns)

data_groupes = pd.concat([y_pred_model_proba_df['proba_classe_1'], shap_values_df], axis=1)
#data_groupes.shape

# On décide de former 3 groupes
bins = [0, 0.20, 0.40, 1.00]
labelos = ['moins_de_20%', '20%_à_40%', 'plus_de_40%']

# On labellise les 3 groupes
data_groupes['classe_clients'] = pd.cut(data_groupes.proba_classe_1, bins=bins, labels=labelos, include_lowest=True)

# On pratique la moyenne de chaque classe
data_groupes_mean = data_groupes.groupby(['classe_clients'], observed=False).mean()
data_groupes_mean = data_groupes_mean.rename_axis('classe_clients').reset_index()                                                  
le_max = X_test.shape[0]-1                                                 
                                                  




st.sidebar.title("Sommaire")
pages = ["Demande de crédit","Déroulement","Identification du client","Analyse globale","Analyse finale", "Décision finale"]

page = st.sidebar.radio("Aller vers la page : ", pages)
if page == pages[0]:
    st.write("### :red[Demande de crédit]")
    st.write("##### La société financière 'Prêt à dépenser' propose des crédits à la consommation ")
    st.write("##### Vous avez sollicité un prêt auprès de notre société.")
    st.write("##### Nous avons recueilli une série d'informations vous concernant et avons étudié votre dossier.")  
    st.write("##### Notre décision vous sera expliquée avec la plus grande clarté.")
    st.image("images/Credit_score_1.png")
    
elif page == pages[1]:
    st.write("### :red[Déroulement]")
    st.write("##### :blue[Nous solliciterons votre identifiant.]")
    st.write("##### :blue[Nous présenterons une analyse globale :]")
    st.write(" - ###### en scindant les clients en 3 groupes")
    st.write(" - ###### en affichant votre groupe d'appartenance")
    st.write(" - ###### en affichant les analyses concernant votre groupe")
    st.write("##### :blue[Nous présenterons une analyse finale:]")
    st.write(" - ###### en affichant les caractéristiques graphiques vous concernant")
    st.write(" - ###### en affichant des comparatifs entre les groupes de clients et vous")
    st.write("##### :blue[Nous conclurons par la décision finale.]")   
    st.image("images/Credit_score.png")

elif page == pages[2]:
    st.write("### :red[Identification du client]") 
    
    idx = st.number_input(" ", min_value=-1, max_value=le_max , placeholder="Tapez ici votre identifiant...")
        
    if idx == -1:
        st.write('##### Merci de nous indiquer votre identifiant')
        st.image("images/Identification_client.png")             
    elif idx >= 0:
         st.write('#### Identifiant retenu :', idx)
         st.write("#### Nous procédons à l'analyse de ce dossier ")
         st.image("images/Identification_client_1.png")
    
elif page == pages[3]:    
    st.write("### :red[Analyses globales]")   
    st.write("##### Notre analyse algorithmique scinde nos :blue[clients en 3 groupes]")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_aggregation(df=data_groupes,
                 group_col='classe_clients',
                 value_col='proba_classe_1',
                 aggreg='mean',
                 ax=ax, orient='v', size_labels=12)
    sns.color_palette("husl", 9)
    plt.title("Probabilité Moyenne de Défaut de Paiement par Groupe de Clients\n",loc="center", fontsize=18, fontstyle='italic')
    st.pyplot(fig)
    st.write()
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    
    if st.checkbox("##### Afficher le groupe du client"):        
        shap_du_client = data_groupes[data_groupes.index == idx]
        groupe_du_client = shap_du_client['classe_clients'].values
        st.write("##### Nos analyses révèlent que vous faites partie du groupe : ",groupe_du_client[0] )
        st.write()
        st.write()
        if groupe_du_client[0] == "moins_de_20%":
            def_pay = round(data_groupes_mean.iloc[0]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "20%_à_40%":           
            def_pay = round(data_groupes_mean.iloc[1]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "plus_de_40%":
            def_pay = round(data_groupes_mean.iloc[2]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
    
    if st.checkbox("##### Afficher les analyses concernant le groupe du client"): 
        st.write("##### Ce graphique s'intitule 'beeswarm'.")
        st.write("##### Les variables sont classées en fonction de leur effet sur la prédiction")
        fig, ax = plt.subplots(figsize=(6, 4))
        the_group = data_groupes[data_groupes['classe_clients']==groupe_du_client[0]].drop(labels=['classe_clients', "proba_classe_1"], axis=1)
        shap_values_group = explainer.shap_values(the_group)
        #shap.summary_plot(shap_values_group[1], the_group)
        shap.summary_plot(shap_values_group, the_group)
        st.pyplot(fig)
    
elif page == pages[4]:
    st.write("### :red[Analyse finale]")
    st.write("##### Cette partie présente les éléments qui ont influé sur la décision finale.")
    st.write("##### La première analyse révèle les éléments constitutifs des caractéristiques du client")
    st.write("##### La seconde analyse compare les résultats du client à la moyenne de chacun des autres.")
    st.write()
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    if st.checkbox("#### Analyse sur le client"):
        st.write("##### Les deux figures montrent comment les caractéristiques ont influencé la prédiction ")
        st.write()
        st.write("##### La première visualisation explique comment le modèle est arrivé à la prédiction faite ")
        #fig, ax = plt.subplots(figsize=(6, 4))
        st_shap(shap.force_plot(explainer.expected_value, 
                shap_values[idx,:], 
                X_test[X_test.index == idx], 
                link='logit',
                figsize=(20, 8),
                ordering_keys=True,
                text_rotation=0,
                contribution_threshold=0.05))
        
        #st.pyplot(fig)
        st.write("##### La deuxième visualisation aide à identifier les principales caractéristiques ayant un pouvoir de décision élevé dans le modèle au niveau individuel")
        st.write("##### Ce graphique est une meilleure visualisation de l'importance des caractéristiques de tous les prédicteurs chez chaque individu.")        
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.decision_plot(explainer.expected_value, 
                   shap_values[idx,:], 
                   X_test[X_test.index == idx], 
                   feature_names=X_test.columns.to_list(),
                   feature_order='importance',
                   #feature_display_range=slice(None, -15, -1),
                   link='logit')
        st.pyplot(fig)
    if st.checkbox("#### Analyse comparative des groupes avec le client"):  
         st.write("##### Chaque groupe est représenté par sa moyenne")
         st.write()
         st.write("##### Meilleure représentation graphique pour les groupes et le client")
         comparaison_client_groupe = pd.concat([data_groupes[data_groupes.index == idx], 
                                      data_groupes_mean],axis = 0)
                                      
         comparaison_client_groupe['classe_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                          X_test.iloc[idx, 0],comparaison_client_groupe['classe_clients'])
         comparaison_client_groupe_light = comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1)
         comparaison_client_groupe_numpy = comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1).to_numpy()
         les_groupes_plus = ['Client','Groupe 1 avec moins_de_20%', 'Groupe 2 avec 20%_à_40%', 'Groupe 3 avec plus_de_40%']

         fig, ax = plt.subplots(figsize=(6, 4))
         shap.decision_plot(explainer.expected_value, 
                   comparaison_client_groupe_numpy, 
                   feature_names=comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1).columns.to_list(),
                   feature_order='importance',
                   highlight=0,
                   legend_labels=les_groupes_plus,
                   legend_location='lower left',
                   feature_display_range=slice(None, -57, -1),
                   link='logit')
         st.pyplot(fig)
    
                                                              
elif page == pages[5]:
    st.write("## :red[Décision finale]")
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    st.write('### Décision finale concernant le client : ', idx)
    result = round(y_pred_model_proba[idx][1]*100,2)
    st.write("##### L'analyse algorithmique révèle un défaut de paiement de {}%".format(result))
    st.write('### Le score retenu est: ', result)
    if result >=40.0:
        st.write("### Votre demande de crédit est :red[REFUSÉE]")
    else:
        st.write("### Votre demande de crédit est :green[ACCEPTÉE]")
    
   
    
   
    
   
    
   
   
   