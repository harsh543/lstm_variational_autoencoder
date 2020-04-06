# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 22:14:20 2020
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess(df):
    
    """returns normalized/standardized data.
    """
    
    df = np.asarray(df, dtype=np.float32)
    
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')
        
    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()
    
    #normalize data
    df = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df)
    print('Data normalized')
    
    return df
     
