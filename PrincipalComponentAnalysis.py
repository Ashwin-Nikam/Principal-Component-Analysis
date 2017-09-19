#Principal Component Analysis

#First we find out the number of features

file = open("../../Desktop/pca_c.txt", "r");
lines = file.readlines();
rows = len(lines);

def findNumFeatures(lines):   #Method for calculating the number of features
    record = lines[0];
    nFeatures = 0;
    for word in record:
        if word == "\t":
            nFeatures += 1;
    return nFeatures;

def createMatrix():           #Method for creating a feature matrix
    columns = findNumFeatures(lines);
    matrix = [[0 for x in range(columns)] for y in range(rows)];
    for row in range(rows):
        features = lines[row].split("\t");
        for column in range(columns):
            matrix[row][column] = features[column];
    findMean(matrix, rows, columns);

def findMean(matrix, rows, columns):  #Calculating mean of each feature
    mean = [0 for i in range(columns)];
    for column in range(columns):
        for row in range(rows):
            mean[column] += float(matrix[row][column]);
    print("Mean of all the features");
    for number in mean:
        number /= rows;
        print(number);

createMatrix();