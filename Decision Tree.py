# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:55:32 2017

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return



    

# Load Data       
adult = pd.read_hdf('datasets.hdf','adult')        
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values



madelon = pd.read_hdf('datasets.hdf','madelon')        
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values




adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     

# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
#alphas=[0]
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('DT',dtclf_pruned(random_state=55))])


pipeA = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params,'DT','madelon')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params,'DT','adult')        


#madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
#adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
madelon_final_params = madelon_clf.best_params_
adult_final_params = adult_clf.best_params_

pipeM.set_params(**madelon_final_params)
makeTimingCurve(madelonX,madelonY,pipeM,'DT','madelon')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'DT','adult')


DTpruningVSnodes(pipeM,alphas,madelon_trgX,madelon_trgY,'madelon')
DTpruningVSnodes(pipeA,alphas,adult_trgX,adult_trgY,'adult')