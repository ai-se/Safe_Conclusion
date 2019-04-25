import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
import scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram
#Clustering birch
from freediscovery.cluster import birch_hierarchy_wrapper
from freediscovery.cluster import Birch,BirchSubcluster
#Sklearn
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Learners
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#Distance measure
from scipy.spatial.distance import euclidean

import warnings

import birch

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_data(path,target):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, prefix=['protocol_type', 'service','flag'])
    if path == 'data/jm1.csv':
        df = df[~df.uniq_Op.str.contains("\?")]
    y = df[target]
    X = df.drop(labels = target, axis = 1)
    X = X.apply(pd.to_numeric)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

    return train_X, test_X, train_y, test_y

def load_mutated_data(path,target):
    train_X, test_X, train_y, test_y = load_data(path,target)
    test_X = pd.concat([train_X,test_X])
    test_y = pd.concat([train_y,test_y])
    return test_X,test_y

# Cluster Driver
def cluster_driver(file,print_tree = True):
    train_X, test_X, train_y, test_y = load_data(file,'defects')
    cluster = birch.birch(branching_factor=20)
    cluster.fit(train_X,train_y)
    cluster_tree,max_depth = cluster.get_cluster_tree()
    #cluster_tree = cluster.model_adder(cluster_tree)
    if print_tree:
        cluster.show_clutser_tree()
    return cluster,cluster_tree,max_depth

if __name__ == "__main__":
    file = 'data/kddcup_10.csv'
    cluster,cluster_tree,max_depth = cluster_driver(file)