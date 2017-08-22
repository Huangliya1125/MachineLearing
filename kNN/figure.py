# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import kNN

dating_matrix,dating_labels = kNN.file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.scatter(dating_matrix[:,1],dating_matrix[:,2])
ax2 = fig.add_subplot(312)
ax2.scatter(dating_matrix[:,1],dating_matrix[:,2],15*np.array(dating_labels),15*np.array(dating_labels))
ax3 = fig.add_subplot(313)
ax3.scatter(dating_matrix[:,0],dating_matrix[:,1],15*np.array(dating_labels),15*np.array(dating_labels))
plt.show()
