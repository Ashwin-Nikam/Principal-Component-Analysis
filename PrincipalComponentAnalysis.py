#Principal Component Analysis

#First we find out the number of features

file = open("../../Desktop/pca_a.txt", "r");
lines = file.readlines();
rows = len(lines);

def findNumFeatures(lines):   #Method for calculating the number of features
    record = lines[0];
    nFeatures = 0;
    for word in record:
        if word == "\t":
            nFeatures += 1;
    return nFeatures;

def createMatrix():
    columns = findNumFeatures(lines);
    matrix = [[0 for x in range(columns)] for y in range(rows)];
    for i in range(rows):
        features = lines[i].split("\t");
        for j in range(columns):
            matrix[i][j] = features[j];

createMatrix();

