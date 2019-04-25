import os
import re
import sys
from glob import glob
from typing import List
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn import preprocessing

# create dummy variables(use only once to modify the dataset)

def load_data(target):
    data_path = 'Data/KDD/kddcup10.csv'
    df = pd.read_csv(data_path)
    df_change = df[['protocol_type', 'service','flag']]
    df = df.drop(labels = ['protocol_type', 'service','flag'], axis = 1)
    df_change = pd.get_dummies(df_change, prefix=['protocol_type', 'service','flag'])
    df = df.merge(df_change,left_index=True, right_index=True)
    columns = df.columns
    y = df[target]
    X = df.drop(labels = target, axis = 1)
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    data = pd.DataFrame(np_scaled, columns = columns[:-1])
    df = data.append(y,ignore_index=True)
    df.to_csv('Data/KDD/kddcup10_dummied_normalized.csv', index = False)
    return df

if __name__ == "__main__":
    load_data('defects')
