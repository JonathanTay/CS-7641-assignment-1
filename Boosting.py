# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""


import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



adult = pd.read_hdf('datasets.hdf','adult')        
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

madelon = pd.read_hdf('datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     



madelon_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
           'Boost__base_estimator__alpha':alphas}
                                   
         
madelon_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=madelon_base,random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('Boost',madelon_booster)])

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',adult_booster)])

#
madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,paramsM,'Boost','madelon')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,paramsA,'Boost','adult')        

#
#
#madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

madelon_final_params = madelon_clf.best_params_
adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##
pipeM.set_params(**madelon_final_params)
pipeA.set_params(**adult_final_params)
makeTimingCurve(madelonX,madelonY,pipeM,'Boost','madelon')
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#
pipeM.set_params(**madelon_final_params)
iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','madelon')        
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')                
pipeM.set_params(**OF_params)
iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','madelon')                
pipeA.set_params(**OF_params)
iterationLC(pipeA,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','adult')                

             
