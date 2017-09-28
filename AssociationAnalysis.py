# import matplotlib.pyplot as plt
#
# x=[1,2,3,4]
# y=[5,6,7,8]
# classes = [2,4,4,2]
# unique = list(set(classes))
#
# print(unique)
#
# colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
#
# print(colors)
#
#
# for i, u in enumerate(unique):
#     xi = [x[j] for j  in range(len(x)) if classes[j] == u]
#     yi = [y[j] for j  in range(len(x)) if classes[j] == u]
#     print(xi, yi)
#     print(colors[i])
#     print(str(u))
#     plt.scatter(xi, yi, c=colors[i], label=str(u))
# plt.legend()
#
# plt.show()



import matplotlib.pyplot as plt
from numpy.random import random

colors = ['b', 'c', 'y', 'm', 'r']

lo = plt.scatter(random(10), random(10), marker='x', color=colors[0])
ll = plt.scatter(random(10), random(10), marker='o', color=colors[0])
l  = plt.scatter(random(10), random(10), marker='o', color=colors[1])
a  = plt.scatter(random(10), random(10), marker='o', color=colors[2])
h  = plt.scatter(random(10), random(10), marker='o', color=colors[3])
hh = plt.scatter(random(10), random(10), marker='o', color=colors[4])
ho = plt.scatter(random(10), random(10), marker='x', color=colors[4])


plt.legend((lo, ll, l, a, h, hh, ho),
           ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)

#plt.show()