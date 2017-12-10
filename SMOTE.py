#   
#   Author: Jeff Brown
#   Date:   12/04/2017
#
#   SMOTE - Synthetic Minority Over-Sampling Technique
#   Create new datapoints for the minority class using KNN (K Nearest
#   Neighbour).

import pandas as pd
import numpy as np
import math
import data
from random import randint
from random import uniform



def transform(T, N, k, X, y):
    
    # This function has several required input parameters as shown below. It
    # returns a specified amount of synthesized vectors of the minority class.
    
    # Functions input parameters:
    # T = Number of minority class samples.
    # N = Amount of SMOTE to apply. This is a percentage. If set to 100
    #     and T = 50. Then 50 minority data-points will be generated.
    # k = Number of nearest neighbours to use.
    # X = Dataset.
    # y = Prediction feature.

    
    m_samples = get_minority(T, X, y)     # Gets minority vectors.
    num_attr = m_samples.shape[1]         # Attribute count for vector.
    
    # Sets the smote amount. This is the % of smote to apply based of the
    # number of minority samples in the dataset. 
    if N < 100:
        T = (N / 100.0) * T
        N = 100
    N = (N / 100) * T  
    N = int(N)    
    
    # Iterates through each minority class data-point.
    X_matrix = data.to_matrix(m_samples)
    nn_vectors = []
    for x in X_matrix:

        # Gets nearest neighbours for a data-point.
        nn_vectors.append(get_nn(m_samples, x, k))
      
    # Synthesizes data-points from nn vectors
    new_vectors = synth(N, X_matrix, nn_vectors, num_attr)
    return new_vectors


def synth(N, X, nn_vectors, attr_length):
    
    # N = The number of new data-points to create.
    # X = Dataset containing all the minority class samples.
    # nn_vectors = A list containing all the KNN for the 
    #   corresponding vector in X.
    # attr_length = The number of features in X.
    
    l = len(nn_vectors[0]) - 1      # Length of KNN array -1
    vectors = []                    # Will hold new vectors.
    
    # Keeps looping until all samples are synthesized.
    while N > 0:
        
        # Iterates through each KNN vector.
        for i in range(len(nn_vectors)):
   
            nn = randint(0, l)
            each_unit = []      
            
            # Creates vector
            for attr in range(attr_length):
                diff = float(nn_vectors[i][nn][attr] - X[i][attr])
                gap = uniform(0,1)
                each_unit.append(X[i][attr] + gap * diff)
            
            N -= 1
            vectors.append(each_unit)
            if N == 0:
                break
            
    return vectors
        
        
def get_nn(X, x, k):
    
    # Function Input Parameters
    # x = A minority class data-point
    # X = All minority class data-points
    # k = Number of nearest neighbours       
    dist = []
    vectors = []
    knn = []
    attr_length = len(x)
    X_matrix = data.to_matrix(X)
    
    # Iterates through each data-point in X
    for x_prime in X_matrix:
        distance = calc_distance(x, x_prime, attr_length)
        dist.append(distance)
        vectors.append(x_prime)
   
    # Selects k nereast vectors
    from heapq import nsmallest
    k_nearest_values = nsmallest(k + 1, dist)
    c = 0
    for d in dist:
        i = 0
        k_count = 0
        while k_count < k + 1:
            if d == k_nearest_values[i]:
                knn.append(vectors[c])
            i += 1
            k_count += 1
        c += 1
    del knn[0]
    
    return(knn)


def calc_distance(point_1, point_2, num_attr):
    
    # Calculates and returns the Euclidean distance between to points
    distance = 0.0
    for i in range(num_attr):
        distance += pow(float(point_1[i]) - float(point_2[i]), 2)
    distance = math.sqrt(distance)
    return distance


        
def get_minority(T, X, y):
    
    # This function will return a dataset composed of all the minority class
    # samples. Validation will be preformed against the specified size of the
    # minority class(T), and the found size.
    
    m_class = y.value_counts().index.tolist()
    m_class = m_class[len(m_class) - 1]
    m_samples = pd.DataFrame(columns=list(X.columns.values))
    for i, row in X.iterrows():
        if y[i] == m_class:
            m_samples = m_samples.append(row)
 
    if m_samples.shape[0] != T:
        msg = "\n\tValue Error: Specified minority class count (T): "
        msg += str(T) + "\n\t Found: " + str(m_samples.shape[0])
        print(msg)
        raise ValueError
    
    return m_samples
