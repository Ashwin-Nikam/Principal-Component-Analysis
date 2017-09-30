"""
-------------------------------------------------------------
            Principal Component Analysis
-------------------------------------------------------------
"""

import numpy as np
import pandas as pd

file = open("Input/pca_demo.txt", "r")
lines = file.readlines()
rows = len(lines)
diseases = []
for i in range(len(lines)):
    diseases.append(lines[i].split("\t")[len(lines[i].split("\t"))-1].strip())
diseases = np.reshape(diseases,(-1,1))

"""
------------------------------------------------------------
Method for converting Matrix to CSV
------------------------------------------------------------
"""

def convertToCSV(matrix,filename):
    matrix = np.hstack((matrix, diseases))
    df=pd.DataFrame(matrix)
    df.to_csv(filename)


"""
-------------------------------------------------------------
Method for calculating the number of features
-------------------------------------------------------------
"""

def findNumFeatures(lines):
    record = lines[0]
    nFeatures = 0
    for word in record:
        if word == "\t":
            nFeatures += 1
    return nFeatures

""" 
-------------------------------------------------------------
This method creates a feature matrix where each column represents
a feature and each row represents a record/
-------------------------------------------------------------
"""

def createMatrix():
    columns = findNumFeatures(lines)
    matrix = [[0 for x in range(columns)] for y in range(rows)]
    for row in range(rows):
        features = lines[row].split("\t")
        for column in range(columns):
            matrix[row][column] = features[column]
    findMean(matrix, rows, columns)

""" 
-------------------------------------------------------------
This method calculates the mean of each feature
and stores it in a list called mean.
-------------------------------------------------------------
"""
def findMean(matrix, rows, columns):
    mean = [0 for i in range(columns)]
    for column in range(columns):
        for row in range(rows):
            mean[column] += float(matrix[row][column])
    for i in range(len(mean)):
        mean[i] /= rows
    transformedMatrix(matrix, mean, rows, columns)

"""
-------------------------------------------------------------
This method creates a new matrix called as newMatrix in which each
feature of each record is subtracted by the mean of that feature.
After that covariance matrix is formed.
-------------------------------------------------------------
"""

def transformedMatrix(matrix, mean, rows, columns):
    newMatrix = [[0 for x in range(columns)] for y in range(rows)]
    for column in range(columns):
        for row in range(rows):
            newMatrix[row][column] = float(matrix[row][column]) - mean[column]
    covariance = np.cov(np.transpose(newMatrix))
    generateEigenValuesAndVectors(covariance, newMatrix)

"""
-------------------------------------------------------------
Generated the eigenvalues and eigenvectors from the covariance
matrix. Then selected the 2 eigenvectors corresponding to the 
maximum 2 eigenvalues.
-------------------------------------------------------------
"""

def generateEigenValuesAndVectors(covariance, newMatrix):
    values, vectors = np.linalg.eig(covariance)
    temp = []
    for i in range(len(values)):
        temp.append(values[i])
    temp.sort()
    temp = temp[::-1]
    maxEigenValue1 = temp[0]
    maxEigenValue2 = temp[1]
    for i in range(len(values)):
        if values[i] == maxEigenValue1:
            maxEigenVector1 = vectors[i]
        elif values[i] == maxEigenValue2:
            maxEigenVector2 = vectors[i]
    PCAImplementation(maxEigenVector1, maxEigenVector2, newMatrix)

"""
-------------------------------------------------------------
This is the main and final method for dimensionality reduction.
Multiplied each row in the newMatrix by the eigenvectors corresponding
to the maximum eigenvalues.
-------------------------------------------------------------
"""

def PCAImplementation(eigenVector1, eigenVector2, newMatrix):
    rows = np.shape(newMatrix)[0]
    columns = np.shape(newMatrix)[1]
    finalMatrix = [[0 for x in range(2)] for y in range(rows)]
    for row in range(rows):
        for column in range(columns):
            finalMatrix[row][0] += newMatrix[row][column] * eigenVector1[column]
            finalMatrix[row][1] += newMatrix[row][column] * eigenVector2[column]

    convertToCSV(finalMatrix,"PCA_Demo.csv")

"""
-------------------------------------------------------------
Now we apply existing packages to run SVD algorithm and 
visualize the data-points
-------------------------------------------------------------
"""

from sklearn.decomposition import TruncatedSVD

columns = findNumFeatures(lines)
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    features = lines[row].split("\t")
    for column in range(columns):
        matrix[row][column] = features[column]

def SVDReduction(matrix):
    svd = TruncatedSVD(n_components=2, n_iter=7)
    newMatrix = svd.fit_transform(matrix)
    convertToCSV(newMatrix,"SVD_Demo.csv")

"""
-------------------------------------------------------------
Now we apply existing packages to run TSNE algorithm and 
visualize the data-points
-------------------------------------------------------------
"""

from sklearn.manifold import TSNE

def TSNEReduction(matrix):
    newMatrix = TSNE(n_components=2).fit_transform(matrix)
    convertToCSV(newMatrix,"TSNE_Demo.csv")

"""
-------------------------------------------------------------
Main call to all the methods.
-------------------------------------------------------------
"""

createMatrix()
SVDReduction(matrix)
TSNEReduction(matrix)

#-----------------------------------------------------------------------------------------------------------------------