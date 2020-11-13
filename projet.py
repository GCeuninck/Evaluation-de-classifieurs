# -*- coding: utf-8 -*-
"""
Binome : HONORIN Alexandre, CEUNINCK Guillaume 

source :
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import *
from scipy import stats


a="fixed acidity"
b="volatile acidity"
c="citric acid"
d="residual sugar"
e="chlorides"
f="free sulfur dioxide"
g="total sulfur dioxide"
h="density"
i="pH"
j="sulphates"
k="alcohol"
l="quality"
named=[a,b,c,d,e,f,g,h,i,j,k,l]


#Preparation des donnees

data = pd.read_csv('red_wines.csv',sep=",")

def figcorr(named, data):
    
    """
    
    Entrées :
        named : liste des attributs
        data : DataFrame des données
    
    Sortie :
        graphiques des corrélations des attributs 2 à 2 (Quality et attribut avec lui même exclus)
        avec distinction de la Quality.
    
    """
    
    number=0
    for number in range(len(named)-1): # -1 car qualite n'est pas a analyser
        for e in range(len(named)-1): # -1 car qualite n'est pas a analyser
            if named[e]is not named[number]: # ne pas analyser un attribut avec lui-même
                fig, ax = plt.subplots()
                ax.set_xlabel(named[number])
                ax.set_ylabel(named[e])
                
                ax.plot(data[(data['quality'] == 1)][named[number]], data[(data['quality'] == 1)][named[e]], "+r", label = 'Quality = 1') 
                ax.plot(data[(data['quality'] == -1)][named[number]], data[(data['quality'] == -1)][named[e]], "og", label = 'Quality = -1') 
                ax.legend()

def fighist(datareduce):  
    
    """
    
    Entrée :
        datareduce : DataFrame des données
    
    Sortie :
        histogrammes représentant la distribution des attributs
    
    """
    
    datareduce.hist('fixed acidity')
    datareduce.hist('volatile acidity')
    datareduce.hist('citric acid')
    datareduce.hist('residual sugar')
    datareduce.hist('chlorides')
    datareduce.hist('free sulfur dioxide')
    datareduce.hist('total sulfur dioxide')
    datareduce.hist('density')
    datareduce.hist('pH')
    datareduce.hist('sulphates')
    datareduce.hist('alcohol')
    datareduce.hist('quality')

def normalize(datareduce):
    
    """
    
    Entrée :
        datareduce : DataFrame des données
    
    Sortie :
        DataFrame des données normalisées
    
    """
    
    temp = datareduce.drop('quality', axis=1)
    scaler = sk.preprocessing.StandardScaler()
    data_scale = scaler.fit_transform(temp)
    res = pd.DataFrame(data_scale, index=temp.index, columns=temp.columns)
    return res.merge(datareduce["quality"], left_index=True, right_index = True)

def reduce_ect(datareduce):
    
    """
    
    Entrée :
        datareduce : DataFrame des données
    
    Sortie :
        DataFrame des données dont la valeur est inférieure à 3 écarts types de la moyenne
    
    """
    
    return datareduce[(np.abs(stats.zscore(datareduce)) < 3).all(axis=1)]

def rechcorrsup(datareducenorm):
    
    """
    
    Entrée :
        datareducenorm : DataFrame des données
    
    Sortie :
        nouveau DataFrame des corrélations supérieures à 5 entre les attributs
    
    """
    
    dat=datareducenorm.corr()
    dat2=dat.abs()[dat.abs()<1]
    return dat2[dat2>0.5]

def affichecorr(named,datareduce):

    for number in named:
        print("\n"+str(number))
        print("\n"+str(rechcorrsup(datareduce)[number]))

def select_critere_val(data, critere, valeur):
    return data[data[critere] == valeur]

def proportion(datareduce):
    return select_critere_val(datareduce, "quality", 1).shape[0], select_critere_val(datareduce, "quality", -1).shape[0]

def proportion_pourcent_Quality(datareduce):
    
    return proportion(datareduce)[0]/(proportion(datareduce)[0] + proportion(datareduce)[1]), proportion(datareduce)[1]/(proportion(datareduce)[0] + proportion(datareduce)[1]) 

def choose_situation(n):
    
    """
    
    Entrée :
        n : entier allant de 1 à 5, correspondant au choix de situation
    
    Sortie :
        DataFrame correspondant à la situation choisie
    
    """
    
    assert n>0 and n<=5, "Veuillez choisir un nombre entre 1 et 5"
    
    datareduce=data[data["pH"]<=14].dropna()
    
    datareducenorm=normalize(datareduce)
    
    datareducenormect=normalize(reduce_ect(datareduce))  
    
    if n == 1: #Résultats obtenus avec les données aberrantes retirées et sans normalisation
        return datareduce
    
    elif n == 2: #Résultats obtenus avec les données aberrantes retirées et normalisation
        return datareducenorm
    
    elif n == 3: #Résultats obtenus avec les données aberrantes et les valeurs supérieures à 3 écarts-types de la moyenne retirées et normalisation
        return datareducenormect

    elif n == 4: #Résultats obtenus avec les données aberrantes, “fixed acidity” retirés et normalisation
        return datareducenorm.drop('fixed acidity', axis=1)
    
    elif n == 5: #Résultats obtenus avec les données aberrantes, “fixed acidity” et “free sulfur dioxide” retirés et normalisation
        return datareducenorm.drop('fixed acidity', axis=1).drop('free sulfur dioxide', axis=1)



#Definition du protocol experimental

def split_data(datareduce, n):
    
    """
    
    Entrée :
        datareduce : DataFrame des données
        n : entier allant de 1 à 5, correspondant au choix de situation
    
    Sortie :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    """
    
    skf = sk.model_selection.StratifiedKFold(n_splits=n)
    X = datareduce.drop('quality', axis=1)
    y = datareduce["quality"]
    X_train, X_test, y_train, y_test = [], [], [], []
    
    for train_index, test_index in skf.split(X, y):
        
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y.iloc[train_index])
        y_test.append(y.iloc[test_index])
        
    return X_train, X_test, y_train, y_test

def test_precision(y_true, y_pred):
    return sk.metrics.precision_score(y_true, y_pred)

def test_recall(y_true, y_pred):
    return sk.metrics.recall_score(y_true, y_pred)

def test_score(y_true, y_pred):
    return sk.metrics.f1_score(y_true, y_pred)

def test_accuracy(y_true, y_pred):
    return sk.metrics.accuracy_score(y_true, y_pred)


#Choix, entrainement et evaluation du classificateur

def test_metrique(y_pred, y_test):
    """
    
    Entrée :
        y_pred : la liste des valeurs de qualité prédites par un classifieur, correspondant aux plis de tests.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Moyenne des métriques suivantes calculées :
        - Précision
        - Recall
        - F1 score
        - Accuracy
    
    """
    
    precision, recall, score, accuracy = np.zeros(shape=(1, len(y_pred))), np.zeros(shape=(1, len(y_pred))), np.zeros(shape=(1, len(y_pred))), np.zeros(shape=(1, len(y_pred)))
    
    for i in range(len(y_pred)):
        precision[0,i] = test_precision(y_test[i], y_pred[i])
        recall[0,i] = test_recall(y_test[i], y_pred[i])
        score[0,i] = test_score(y_test[i], y_pred[i])
        accuracy[0,i] = test_accuracy(y_test[i], y_pred[i])
    
    print("Moyenne Test precision : " + str(precision.mean()))
    print("Moyenne Test recall : " + str(recall.mean()))
    print("Moyenne Test score : " + str(score.mean()))
    print("Moyenne Test accuracy : " + str(accuracy.mean()) + "\n")

def Logistic_Regression(X_train, X_test, y_train):
    
    
    LR = sk.linear_model.LogisticRegression() #max_iter = 1000 pour non normalise
    y_pred = []
    
    for i in range(len(X_train)):
        LR.fit(X_train[i],y_train[i])
        y_pred.append(LR.predict(X_test[i]))
    
    return y_pred

def test_LR(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur Logistic Regression
    
    """
    
    y_pred = Logistic_Regression(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)

def Linear_SVC(X_train, X_test, y_train):
    
    SVC = sk.svm.LinearSVC(max_iter = 10000) 
    y_pred = []
    
    for i in range(len(X_train)):
        SVC.fit(X_train[i],y_train[i])
        y_pred.append(SVC.predict(X_test[i]))
    
    return y_pred

def test_SVC(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur SVC
    
    """

    y_pred = Linear_SVC(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)

def Linear_Discriminant_Analysis(X_train, X_test, y_train):
    
    LDA = sk.discriminant_analysis.LinearDiscriminantAnalysis()
    y_pred = []
    
    for i in range(len(X_train)):
        LDA.fit(X_train[i],y_train[i])
        y_pred.append(LDA.predict(X_test[i]))
    
    return y_pred

def test_LDA(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur Linear Discriminant Analysis
    
    """

    y_pred = Linear_Discriminant_Analysis(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)

def Quadratic_Discriminant_Analysis(X_train, X_test, y_train):
    
    QDA = sk.discriminant_analysis.QuadraticDiscriminantAnalysis()
    y_pred = []
    
    for i in range(len(X_train)):
        QDA.fit(X_train[i],y_train[i])
        y_pred.append(QDA.predict(X_test[i]))
    
    return y_pred

def test_QDA(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur Quadratic Discriminant Analysis
    
    """
    
    y_pred = Quadratic_Discriminant_Analysis(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)
    
def KNeighbors_Classifier(X_train, X_test, y_train):
    
    KNC = sk.neighbors.KNeighborsClassifier()
    y_pred = []
    
    for i in range(len(X_train)):
        KNC.fit(X_train[i],y_train[i])
        y_pred.append(KNC.predict(X_test[i]))
    
    return y_pred

def test_KNC(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur KNeighbors Classifier
    
    """

    y_pred = KNeighbors_Classifier(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)

def Decision_Tree_Classifier(X_train, X_test, y_train):
    
    DTC = sk.tree.DecisionTreeClassifier(random_state = 1)
    y_pred = []
    
    for i in range(len(X_train)):
        DTC.fit(X_train[i],y_train[i])
        y_pred.append(DTC.predict(X_test[i]))
    
    return y_pred

def test_DTC(X_train, X_test, y_train, y_test): #Change de valeurs d'un test a l'autre
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur Decision Tree
    
    """
    
    y_pred = Decision_Tree_Classifier(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)
    
#Perceptron

class Perceptron(object):
    
    def __init__(self, nb_of_attributes, epochs=500, learning_rate=0.01):
        
        """
        
        Entrée :
            nb_of_attributes : nombres d'attributs dans les données d'entrainement
            epochs : nombres maximum d'itérations de l'algorithme, par défaut à 500
            learning_rate : taux d'apprentissage, par défaut à 0.01
        
        Sortie :
            Objet Perceptron
        
        """
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(nb_of_attributes + 1)
           
    def predict(self, value):
        
        """
        
        Entrée :
            value : une donnée dont la quality doit être prédite
        
        Sortie :
            Quality prédite par le Perceptron {1,-1}
        
        """
        
        signe = np.dot(value, self.weights[1:]) + self.weights[0]
        if signe > 0:
          prediction = 1
        else:
          prediction = -1            
        return prediction

    def fit(self, X_train, y_train):
        
        """
        
        Entrée :
            X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
            y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        
        Sortie :
            Construction du modèle Perceptron à partir des données d'entrainement
        
        """
        
        for _ in range(self.epochs):
            for inputs, label in zip(X_train, y_train):
                prediction = self.predict(inputs)
                if prediction != label:
                    self.weights[1:] += self.learning_rate * label * inputs
                    self.weights[0] += self.learning_rate * label
    

def pred_Perceptron(X_train, X_test, y_train):
    
    """
        
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
    
    Sortie :
        y_pred : la liste des valeurs de qualité prédites par le Perceptron, correspondant aux plis de tests.
            
    """
    
    y_pred = []
    
    for i in range(len(X_train)):
        
        temp = []
        
        x = X_train[i].to_numpy()
        y = y_train[i].to_numpy()
        x_t = X_test[i].to_numpy()
        
        perceptron = Perceptron(x.shape[1])
        perceptron.fit(x,y)
        for inputs in x_t:
            temp.append(perceptron.predict(inputs))
        y_pred.append(temp)

    return y_pred

def test_Perceptron(X_train, X_test, y_train, y_test):
    
    """
    
    Entrée :
        X_train : la liste de chaque plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        X_test : la liste de chaque plis de tests, utilisée par le classifieur pour prédire une qualité à partir du modèle créé.
        y_train : la liste des valeurs de qualité réelle correspondant respectivement à chacun des plis d'entraînements, utilisé par le classifieur pour créer son modèle.
        y_test : la liste des valeurs de qualité réelle correspondant aux plis de tests, utilisée pour tester et évaluer le classifieur par comparaison avec les qualités prédites.
    
    Sortie :
        Affiche le résultat des tests de métriques pour le classifieur Perceptron
    
    """
    
    y_pred = pred_Perceptron(X_train, X_test, y_train)
    test_metrique(y_pred, y_test)

def test_all(n):
    
    """
    
    Entrée : 
         n : entier allant de 1 à 5, correspondant au choix de situation
         
    Sortie :
        Affiche le résultat des tests de métriques pour tous les classifieurs selon la situation choisie
    
    """
    
    X_train, X_test, y_train, y_test = split_data(choose_situation(n), 10)
    
    print("Logistic Regression :") 
    test_LR(X_train, X_test, y_train, y_test)
    print("Linear SVC :")
    test_SVC(X_train, X_test, y_train, y_test)
    print("Linear Discriminant Analysis :")
    test_LDA(X_train, X_test, y_train, y_test)
    print("Quadratic Discriminant Analysis :")
    test_QDA(X_train, X_test, y_train, y_test)
    print("KNeighbors Classifier :")
    test_KNC(X_train, X_test, y_train, y_test)
    print("Decision Tree Classifier :")
    test_DTC(X_train, X_test, y_train, y_test)
    print("Perceptron :") 
    test_Perceptron(X_train, X_test, y_train, y_test)

 
    

