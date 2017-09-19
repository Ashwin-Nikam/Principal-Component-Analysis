#Reading a file in python
"""

file = open("../../Desktop/pca_a.txt", "r");
print(file.read());

"""

#Principal Component Analysis

#First we find out the number of features
file = open("../../Desktop/pca_a.txt", "r");
lines = file.readlines();  #lines contains all the lines in the txt file
record = lines[0];
nFeatures = 0;
for word in record:
    if word == "\t":
        nFeatures+=1;

print(record);
print("Number of features are ",nFeatures);