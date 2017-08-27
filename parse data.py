# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:11:29 2017

@author: jtay
"""

import pandas as pd
import numpy as np

# Preprocess with adult dataset
adult = pd.read_csv('./adult.data',header=None)
adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
adult['income'] = pd.get_dummies(adult.income)
print(adult.groupby('occupation')['occupation'].count())
print(adult.groupby('country').country.count())
#http://scg.sdsu.edu/dataset-adult_r/
replacements = { 'Cambodia':' SE-Asia',
                'Canada':' British-Commonwealth',
                'China':' China',
                'Columbia':' South-America',
                'Cuba':' Other',
                'Dominican-Republic':' Latin-America',
                'Ecuador':' South-America',
                'El-Salvador':' South-America ',
                'England':' British-Commonwealth',
                'France':' Euro_1',
                'Germany':' Euro_1',
                'Greece':' Euro_2',
                'Guatemala':' Latin-America',
                'Haiti':' Latin-America',
                'Holand-Netherlands':' Euro_1',
                'Honduras':' Latin-America',
                'Hong':' China',
                'Hungary':' Euro_2',
                'India':' British-Commonwealth',
                'Iran':' Other',
                'Ireland':' British-Commonwealth',
                'Italy':' Euro_1',
                'Jamaica':' Latin-America',
                'Japan':' Other',
                'Laos':' SE-Asia',
                'Mexico':' Latin-America',
                'Nicaragua':' Latin-America',
                'Outlying-US(Guam-USVI-etc)':' Latin-America',
                'Peru':' South-America',
                'Philippines':' SE-Asia',
                'Poland':' Euro_2',
                'Portugal':' Euro_2',
                'Puerto-Rico':' Latin-America',
                'Scotland':' British-Commonwealth',
                'South':' Euro_2',
                'Taiwan':' China',
                'Thailand':' SE-Asia',
                'Trinadad&Tobago':' Latin-America',
                'United-States':' United-States',
                'Vietnam':' SE-Asia',
                'Yugoslavia':' Euro_2'}
adult['country'] = adult['country'].str.strip()
adult = adult.replace(to_replace={'country':replacements,
                                  'employer':{' Without-pay': ' Never-worked'},
                                  'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})    
adult['country'] = adult['country'].str.strip()
print(adult.groupby('country').country.count())   
for col in ['employer','marital','occupation','relationship','race','sex','country']:
    adult[col] = adult[col].str.strip()
    
adult = pd.get_dummies(adult)
adult = adult.rename(columns=lambda x: x.replace('-','_'))

adult.to_hdf('datasets.hdf','adult',complib='blosc',complevel=9)

# Madelon
madX1 = pd.read_csv('./madelon_train.data',header=None,sep=' ')
madX2 = pd.read_csv('./madelon_valid.data',header=None,sep=' ')
madX = pd.concat([madX1,madX2],0).astype(float)
madY1 = pd.read_csv('./madelon_train.labels',header=None,sep=' ')
madY2 = pd.read_csv('./madelon_valid.labels',header=None,sep=' ')
madY = pd.concat([madY1,madY2],0)
madY.columns = ['Class']
mad = pd.concat([madX,madY],1)
mad = mad.dropna(axis=1,how='all')
mad.to_hdf('datasets.hdf','madelon',complib='blosc',complevel=9)