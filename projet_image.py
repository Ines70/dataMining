# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:35:27 2022

@author: Inès
"""

import random
from numpy import *
from matplotlib.pyplot import*
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.decomposition import NMF , PCA

#================================(QUESTION 1)==================================
digits = datasets.load_digits()
digits_labels = digits.target

#print echantillons
digits_data = digits.images.reshape((len(digits.images), -1))
print(digits_data.shape)

#================================(QUESTION 2)==================================
#@Param : debImage<INT> data<Array> n<INT> m<INT> title <String>  
#@Return : Un plot de n*m s de chiffres manuscrits 

def show_chiffre(debImage,data,n,m,title):
    plt.figure(figsize=(10,8))
    plt.suptitle(title,fontsize=14)
    for i in range(n*m):
        axi=plt.subplot(n, m, i + 1)
        axi.imshow(data[debImage+i].reshape((8, 8)), cmap=plt.cm.gray)
        axi.set_title(digits.target[i+debImage],color='red')
        #On peut enlever les axes car ici d'aucune utilitée
        axi.axis('off')
    plt.show()


#================================(QUESTION 3)==================================

#@Param : data<Array>
#@Return : <double>    
#Rq :On veut la moyenne le long des colonnes donc axis=1            
def MOYENNE(data):
    return np.mean(data,axis=1)
     
#@Param 
#@Return
def matriceA(data):
    A=zeros(digits.data.shape)
    for i in range(digits_data.shape[0]):
        for j in range(digits_data.shape[1]):
           A[i,j]=data[i,j]-(MOYENNE(data)[i])
    return A

A=matriceA(digits.data)

#Calculer les vecteurs propres
pca = PCA(n_components=10, whiten=True).fit(A)
pcaPRO = pca.components_

#=================================(QUESTION 4)=================================


#La factorisation par matrice non négative  NMF 
factorisation = NMF(n_components=13)
factorisation.fit(digits_data)
NMF_FACT=factorisation.components_

show_chiffre(0,digits_data,3,4,"Digits Data")
show_chiffre(0,NMF_FACT,3,4,"Factorisatation matricielle NON NEGATIVE (NMF)")


#=================================(QUESTION 5)=================================
#NMF_FACT c'est l'estimation d'une matrice non negative 
#donc pour la projective NMF on doit avoir X = NMF_FACT x NMF_FACT ^T x X 

#@Param : data <Array>,nmfData <Array>
#@Return : <Array>
def projectiveNMF(data,nmfData):
    nmfData_T= transpose (nmfData)
    A= nmfData_T.dot(nmfData)
    B= data.dot(A)
    return B

PROJNMF=projectiveNMF(digits_data,NMF_FACT)

show_chiffre(0,PROJNMF,3,4,"Projective NMF")
show_chiffre(5,PROJNMF,3,4,"Projective NMF")
show_chiffre(5,digits_data,3,4,"Digits Data")
show_chiffre(0,pcaPRO,3,4,"Analyse à Composantes Principales")

#=================================(QUESTION 6)=================================

#Fonction sparsity qui calcule la proportion de 0 dans une matrice 
#@param: data<Array> 
#@return : <Array>
def sparsity(data):
    T=count_nonzero(data)
    return 100-(T/size(data))*100