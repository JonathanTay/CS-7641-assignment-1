# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



adult = pd.read_hdf('datasets.hdf','adult')        
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

madelon = pd.read_hdf('datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values



adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     


d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]
d = madelonX.shape[1]
hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]


pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])  

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  



params_madelon= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'KNN','madelon')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')        


#madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
#adult_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
madelon_final_params=madelon_clf.best_params_
adult_final_params=adult_clf.best_params_



pipeM.set_params(**madelon_final_params)
makeTimingCurve(madelonX,madelonY,pipeM,'KNN','madelon')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')