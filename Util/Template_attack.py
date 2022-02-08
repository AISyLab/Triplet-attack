import os
#os.environ['PYTHONHASHSEED']=str(1)

import numpy as np
#np.random.seed(1)
from scipy.stats import multivariate_normal

def template_training(X,Y, pool=False):
    num_clusters = max(Y) + 1
    classes = np.unique(Y)
    # assign traces to clusters based on lables
    HW_catagory_for_traces = [[] for _ in range(num_clusters)]
    for i in range(len(X)):
        HW = Y[i]
        HW_catagory_for_traces[HW].append(X[i])
    HW_catagory_for_traces = [np.array(HW_catagory_for_traces[HW]) for HW in range(num_clusters)]

    # calculate Covariance Matrices
    # step 1: calculate mean matrix of POIs
    meanMatrix = np.zeros((num_clusters, len(X[0])))
    for i in range(num_clusters):
        meanMatrix[i] = np.mean(HW_catagory_for_traces[i], axis=0)
    # step 2: calculate covariance matrix
    covMatrix = np.zeros((num_clusters, len(X[0]), len(X[0])))
    for HW in range(num_clusters):
        for i in range(len(X[0])):
            for j in range(len(X[0])):
                x = HW_catagory_for_traces[HW][:, i]
                y = HW_catagory_for_traces[HW][:, j]
                covMatrix[HW, i, j] = np.cov(x, y)[0][1]
    if pool:
        covMatrix[:] = np.mean(covMatrix, axis=0)
    return meanMatrix, covMatrix, classes

# Calculate index of the most possible cluster for each traces
def template_attacking(meanMatrix, covMatrix, X_attack, classes):
    # launch attacks
    number_traces = X_attack.shape[0]
    prediction = np.zeros((number_traces))
    for i in range(number_traces):
        if(i % 2000 == 0):
            print(str(i) + '/' + str(number_traces))
        proba = np.zeros(classes.shape[0])
        for j,cl in enumerate(classes):
            rv = multivariate_normal(meanMatrix[j], covMatrix[j])
            proba[j] = rv.pdf(X_attack[i])
        sorted_index = np.argsort(proba)
        # Store the index of the most possible cluster
        prediction[i] = classes[sorted_index[-1]]

    return prediction

# Calculate probability of the most possible cluster for each traces
def template_attacking_proba(meanMatrix, covMatrix, X_test, classes):
    number_traces = X_test.shape[0]
    proba = np.zeros((number_traces,classes.shape[0]))
    rv_array = []
    m = 1e-6
    for idx in range(len(classes)):
        rv_array.append(multivariate_normal(meanMatrix[idx], covMatrix[idx]))

    for i in range(number_traces):
        if(i % 2000 == 0):
            print(str(i) + '/' + str(number_traces))
        proba[i] = [o.pdf(X_test[i]) for o in rv_array]
        # for j,cl in enumerate(classes):
        #     #rv = multivariate_normal(meanMatrix[j], covMatrix[j])
        #     # Store the probability of the most possible cluster
            
        #     proba[i,j] = rv_array[j].pdf(X_test[i])

    return proba

def acc_template(true, pred):
    acc = 0.0
    for i in range(pred.shape[0]):
        if int(pred[i])==int(true[i]):
           acc +=1

    return acc/float(pred.shape[0])

