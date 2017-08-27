# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:55:38 2017

@author: JTay
"""
import numpy as np
from time import clock
import sklearn.model_selection as ms
import pandas as pd
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as dtclf


def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)

scorer = make_scorer(balanced_accuracy)    
    
def basicResults(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = ms.GridSearchCV(clfObj,n_jobs=1,param_grid=params,refit=True,verbose=10,cv=5,scoring=scorer)
    cv.fit(trgX,trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type,dataset),index=False)
    test_score = cv.score(tstX,tstY)
    with open('./output/test results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(clf_type,dataset,test_score,cv.best_params_))    
    N = trgY.shape[0]    
    curve = ms.learning_curve(cv.best_estimator_,trgX,trgY,cv=5,train_sizes=[50,100]+[int(N*x/10) for x in range(1,8)],verbose=10,scoring=scorer)
    curve_train_scores = pd.DataFrame(index = curve[0],data = curve[1])
    curve_test_scores  = pd.DataFrame(index = curve[0],data = curve[2])
    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type,dataset))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type,dataset))
    return cv

    
def iterationLC(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = ms.GridSearchCV(clfObj,n_jobs=1,param_grid=params,refit=True,verbose=10,cv=5,scoring=scorer)
    cv.fit(trgX,trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(clf_type,dataset),index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:        
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name:value})
        clfObj.fit(trgX,trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(balanced_accuracy(trgY,pred))
        clfObj.fit(trgX,trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(balanced_accuracy(tstY,pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(clf_type,dataset),index=False)
    return cv    
    
def add_noise(y,frac=0.1):
    np.random.seed(456)
    n = y.shape[0]
    sz = int(n*frac)
    ind = np.random.choice(np.arange(n),size=sz,replace=False)
    tmp = y.copy()
    tmp[ind] = 1-tmp[ind]
    return tmp
    
    
def makeTimingCurve(X,Y,clf,clfName,dataset):
    out = defaultdict(dict)
    for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:    
        X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=frac, random_state=42)
        st = clock()
        np.random.seed(55)
        clf.fit(X_train,y_train)
        out['train'][frac]= clock()-st
        st = clock()
        clf.predict(X_test)
        out['test'][frac]= clock()-st
        print(clfName,dataset,frac)
    out = pd.DataFrame(out)
    out.to_csv('./output/{}_{}_timing.csv'.format(clfName,dataset))
    return 
        
        
    
    
    
class dtclf_pruned(dtclf):        
    def remove_subtree(self,root):
        '''Clean up'''
        tree = self.tree_
        visited,stack= set(),[root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left =tree.children_left[v]
            right=tree.children_right[v]
            if left >=0:
                stack.append(left)
            if right >=0:
                stack.append(right)
        for node in visited:
            tree.children_left[node] = -1
            tree.children_right[node] = -1
        return 
        
    def prune(self):      
        C = 1-self.alpha
        if self.alpha <= -1: # Early exit
            return self
        tree = self.tree_        
        bestScore = self.score(self.valX,self.valY)        
        candidates = np.flatnonzero(tree.children_left>=0)
        for candidate in reversed(candidates): # Go backwards/leaves up
            if tree.children_left[candidate]==tree.children_right[candidate]: # leaf node. Ignore
                continue
            left = tree.children_left[candidate]
            right = tree.children_right[candidate]
            tree.children_left[candidate]=tree.children_right[candidate]=-1            
            score = self.score(self.valX,self.valY)
            if score >= C*bestScore:
                bestScore = score                
                self.remove_subtree(candidate)
            else:
                tree.children_left[candidate]=left
                tree.children_right[candidate]=right
        assert (self.tree_.children_left>=0).sum() == (self.tree_.children_right>=0).sum() 
        return self
        
    def fit(self,X,Y,sample_weight=None,check_input=True, X_idx_sorted=None):        
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) 
        self.trgX = X.copy()
        self.trgY = Y.copy()
        self.trgWts = sample_weight.copy()        
        sss = ms.StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=123)
        for train_index, test_index in sss.split(self.trgX,self.trgY):
            self.valX = self.trgX[test_index]
            self.valY = self.trgY[test_index]
            self.trgX = self.trgX[train_index]
            self.trgY = self.trgY[train_index]
            self.valWts = sample_weight[test_index]
            self.trgWts = sample_weight[train_index]
        super().fit(self.trgX,self.trgY,self.trgWts,check_input,X_idx_sorted)
        self.prune()
        return self
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 class_weight=None,
                 presort=False,
                 alpha = 0):
        super(dtclf_pruned, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_split=min_impurity_split,
            presort=presort)
        self.alpha = alpha
        
    def numNodes(self):
        assert (self.tree_.children_left>=0).sum() == (self.tree_.children_right>=0).sum() 
        return  (self.tree_.children_left>=0).sum() 