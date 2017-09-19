#Principal Component Analysis


file = open("../../Desktop/pca_a.txt", "r");
lines = file.readlines();
rows = len(lines);

"""
-------------------------------------------------------------
Method for calculating the number of features
-------------------------------------------------------------
"""

def findNumFeatures(lines):
    record = lines[0];
    nFeatures = 0;
    for word in record:
        if word == "\t":
            nFeatures += 1;
    return nFeatures;

""" 
-------------------------------------------------------------
This method creates a feature matrix where each column represents
a feature and each row represents a record/
-------------------------------------------------------------
"""

def createMatrix():           #Method for creating a feature matrix
    columns = findNumFeatures(lines);
    matrix = [[0 for x in range(columns)] for y in range(rows)];
    for row in range(rows):
        features = lines[row].split("\t");
        for column in range(columns):
            matrix[row][column] = features[column];
    findMean(matrix, rows, columns);

""" 
-------------------------------------------------------------
This method calculates the mean of each feature
and stores it in a list called mean.
-------------------------------------------------------------
"""
def findMean(matrix, rows, columns):
    mean = [0 for i in range(columns)];
    for column in range(columns):
        for row in range(rows):
            mean[column] += float(matrix[row][column]);
    for i in range(len(mean)):
        mean[i] /= rows;
    transformedMatrix(matrix, mean, rows, columns);

"""
-------------------------------------------------------------
This method creates a new matrix called as newMatrix in which each
feature of each record is subtracted by the mean of that feature.
-------------------------------------------------------------
"""

def transformedMatrix(matrix, mean, rows, columns):
    newMatrix = [[0 for x in range(columns)] for y in range(rows)];
    for column in range(columns):
        for row in range(rows):
            newMatrix[row][column] = float(matrix[row][column]) - mean[column];

createMatrix();