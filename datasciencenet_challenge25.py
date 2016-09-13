#!/opt/local/bin/python
"""
CODE ANALYSANT LES DONNEES DU CHALLENGE 25 DE DATASCIENCE.NET

Version: 1.0
Created: 13/09/2016
Compiler: python

Author: Dr. Samia Drappeau (SD), drappeau.samia@gmail.com
Notes: 
"""
# coding: utf-8

# # Challenge 'Le meilleur data scientist de France #1'
# [url](https://www.datascience.net/fr/challenge/25/details)

# ## Importation des librairies

# In[1]:

#from matplotlib import rc_file
#rc_file('/Users/samiadrappeau/.matplotlib/journalplotrc/mnras.rc')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#from matplotlib.ticker import MultipleLocator
get_ipython().magic('matplotlib inline')
import seaborn as sns


# ## Importation des données et visualisation des 10 premiers éléments

# In[2]:

data_train = pd.read_csv('../data_challenge/boites_medicaments_train.csv', encoding='utf-8', sep=';')
data_test = pd.read_csv('../data_challenge/boites_medicaments_test.csv', encoding='utf-8', sep=';')

# permet de voir toutes les colonnes
pd.set_option("display.max_columns", 99)    
# Visualisons les 10 premières entrées
data_train.head(10)


# ## Analyse descriptive
# ### Structures des datasets

# In[3]:

print('Les données train ont ' + str(data_train.shape[0]) + ' observables, de ' + str(data_train.shape[1]) + ' variables chacune.')
print('Les données test ont ' + str(data_test.shape[0]) + ' observables, de ' + str(data_test.shape[1]) + ' variables chacune.')


# Il est **très** important de lire la [description des variables](https://www.datascience.net/fr/challenge/25/details#tab_brief71), afin de correctement appréhender le problème.

# ### Que cherchons-nous à prédire?
# Nous cherchons à développer un modèle prédictif permettant d'estimer le coût d'une boîte de médicament.
# 
# Commençons donc par étudier la distribution des prix dans les données train.

# ### Description des données

# In[4]:

data_train['prix'].value_counts()


# In[5]:

ax = data_train['prix'].hist()
ax.set_xlabel('prix [euros]')
ax.set_ylabel('nombres de medicaments')


# Ce n'est pas très joli. Essayons d'améliorer les choses en prenant le logarithme des prix.

# In[6]:

data_train['logprix'] = data_train['prix'].apply(np.log)
ax = data_train['logprix'].hist()
ax.set_xlabel('log(prix [euros])')
ax.set_ylabel('nombres de medicaments')


# C'est un peu plus joli. Nous avons une belle distribution, proche d'une gaussienne.
# 
# Observons maintenant l'influence de deux variables sur le prix final des médicaments : le taux de remboursement de la Sécurité Sociale et le statut administratif de la boîte.
# 
# Pour cela, nous allons utiliser la fonction *violinplot()* de Seaborn. La [documentation](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.violinplot.html) donne une description de la fonction :
# >Draw a combination of boxplot and kernel density estimate.
# 
# >A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across >several levels of one (or more) categorical variables such that those distributions can be compared. Unlike a box >plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density >estimation of the underlying distribution.

# Un *kernel density estimate* (ou *Estimation du noyau* en français) est une fonction de probabilité.
# 
# Quelques définitions :  
# **Kernel** : c'est un type particulier de fonction de probabilité qui a la propriété d'être paire.  
# **KDE** : c'est une méthode non-paramétrique d'estimation de la fonction de probabilité d'une variable aléatoire continue.  
# **Non-paramétrique** : n'assume aucune distribution sous-jacente de la variable.

# In[7]:

# Influence du taux de remboursement sur le prix
ax = sns.violinplot(y="logprix", x="tx rembours", data=data_train)
ax.set_xlabel('Taux de remboursement Secu')
ax.set_ylabel('log(prix [euros])')


# In[8]:

# Influence du statut du medicament sur son prix
ax = sns.violinplot(y="logprix", x="statut", data=data_train)
ax.set_xlabel('Statut du medicament')
ax.set_ylabel('log(prix [euros])')


# # XXX INSERT HERE DESCRIPTION OF DISTRIBUTION

# ## Préparation des données
# ### Séparation des différents types de variables
# Nous constatons, en analysant les variables disponibles, qu'elles sont de quatre types différentes :
# * numérique  
# * catégorielle  
# * date  
# * texte
# 
# Séparons les donc :

# In[9]:

# variables numériques
var_num = ['libelle_plaquette', 'libelle_ampoule', 'libelle_flacon', 
            'libelle_tube', 'libelle_stylo', 'libelle_seringue',
            'libelle_pilulier', 'libelle_sachet', 'libelle_comprime', 
            'libelle_gelule', 'libelle_film', 'libelle_poche',
            'libelle_capsule'] + ['nb_plaquette', 'nb_ampoule', 
            'nb_flacon', 'nb_tube', 'nb_stylo', 'nb_seringue',
            'nb_pilulier', 'nb_sachet', 'nb_comprime', 'nb_gelule', 
            'nb_film', 'nb_poche', 'nb_capsule', 'nb_ml']
# variables catégorielles
var_cat = ['statut', 'etat commerc', 'agrement col', 'tx rembours',
          'voies admin', 'statut admin', 'type proc']
# variables dates
var_dates = ['date declar annee', 'date amm annee']

# variable texte
var_txt = ['libelle', 'titulaires', 'substances', 'forme pharma']


# ### Encodage des variables catégorielles
# Nous avons besoin de transformer les variables catégorielles en nombre, afin de pouvoir les utiliser dans les algorithmes de machine learning (qui nécessitent des nombres comme paramètres d'entrée).
# 
# Pour cela, nous allons utiliser la fonction [*LabelEncoder()*](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

# In[10]:

data_train[var_cat].head(10)


# In[11]:

for c in var_cat:
    le = LabelEncoder()
    le.fit(data_train[c].append(data_test[c]))
    data_train[c] = le.transform(data_train[c])
    data_test[c] = le.transform(data_test[c])


# In[12]:

data_train[var_cat].head(10)


# ## Création du modèle prédictif
# Nous sommes maintenant prêt à définir un modèle prédictif des prix des médicaments.
# 
# Nous allons tester plusieurs modèles :
# * Random forest  
# * XXX ADD EACH MODEL TESTED
# 
# Nous allons également utiliser toutes les variables à notre disposition, excepté les variables textuelles.
# 
# Enfin, afin d'éviter le [surapprentissage](https://fr.wikipedia.org/wiki/Surapprentissage) et correctement estimer les performances de nos modèles, nous allons utiliser la [validation croisée](https://fr.wikipedia.org/wiki/Validation_crois%C3%A9e) avec le critère *k-fold*.

# ### Critère de performance
# Le challenge utilise la métrique [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error). Celle-ci n'étant pas disponible dans sklearn, nous la codons manuellement :

# In[13]:

# Mean Absolute Percentage Error
def mape_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ### Validation croisée
# [Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html) de la fonction KFold() de validation croisée de sklearn.
# 
# > K-Folds cross validation iterator.  
# > Provides train/test indices to split data in train test sets. Split dataset into k consecutive folds (without shuffling by default).  
# > Each fold is then used a validation set once while the k - 1 remaining fold form the training set.
# 
# Prend comme paramètres d'entrée :
# * le nombre total d'éléments, ici data_train.shape[0] = 8564  
# * le nombre de fold à effectuer, ici NBROUND = 5

# In[21]:

err = 0
NBROUND = 5
VARIABLES = var_num+var_cat+var_dates
for train_index, test_index in KFold(data_train.shape[0], n_folds=NBROUND):
    y = data_train['logprix']
    X = data_train[VARIABLES]
    X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    # Random Forest model
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # ne pas oublier de reprendre l'exponentielle de la prédiction
    err += mape_error(np.exp(y_test), np.exp(pred))
    print (mape_error(np.exp(y_test), np.exp(pred)))
print ("*** MAPE Error : ", err / NBROUND)


# Notre modèle *Random Forest* prédit les prix des médicaments avec une erreur d'environ 61%. Pour un prix de médicament réel de 10€, notre modèle prédira soit 3.9€, soit 16.1€.

# ## Prédictions et soumission
# ### Calcul des prédictions
# Nous allons maintenant entrainer notre modèle sur l'intégralité des données train, avant d'effectuer une prédiction pour les données test.

# In[22]:

clf = RandomForestRegressor()
clf.fit(data_train[VARIABLES], data_train['logprix'])
# ATTENTION !! Bien penser à transformer les prédictions obtenues dans l'espace linéaire.
predictions = np.exp(clf.predict(data_test[VARIABLES]))


# ### Création du fichier de soumission
# Sauvegardons maintenant nos prédictions dans le *soumission.csv* pour le soumettre à datascience.net

# In[24]:

pd.DataFrame(predictions, index=data_test['id']).to_csv('soumission.csv',  
                          header=['prix'],
                          sep = ';')


# # Félicitations
# Nous venons d'analyser notre premier set de donnée. Soumettons notre fichier afin de comparer notre modèle à celui des autres Data Scientist du challenge. Puis amusons-nous à améliorer ce modèle !
