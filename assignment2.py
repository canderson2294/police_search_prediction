#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:28:58 2020

@author: camille
"""

import pandas as pd
df = pd.read_csv('police_stop_data.csv', index_col = 'OBJECTID')

df.info()
df.isna().sum()

to_keep = ['problem', 'personSearch', 'vehicleSearch', 'preRace', 
           'race', 'gender', 'policePrecinct']

df = df[to_keep]

#delete rows with nans
df = df.dropna()

df.to_csv('policeStop.csv')

df.isin(['Unknown']).sum()
df = df[~df.eq('Unknown').any(1)]



from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.3, random_state=0)


X_train.to_csv('policeStop_Train.csv')
X_test.to_csv('policeStop_Test.csv')


