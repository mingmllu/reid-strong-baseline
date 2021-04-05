import sys
import os
import numpy as np

P=1501 #751
list_dist = []
for pid in range(P):
    if not os.path.isfile('train_pid_%04d.npy'%(pid)):
        continue
    a = np.load('train_pid_%04d.npy'%(pid))
    N = a.shape[0]
    for i in range(N):
        for j in range(N):
            if j > i:
                dist = np.linalg.norm(a[i,:]-a[j,:])
                list_dist.append(dist)

title = 'within_class'
f = open('hist_' + title + '.txt', 'w')
list_stack_counts = []
b = np.array(list_dist)
counts, bins = np.histogram(b, bins=100, range=(0,2))
counts = counts / np.sum(counts)
list_stack_counts.append(counts)

counts = np.stack(list_stack_counts)
for i in range(1, bins.shape[0]):
    line = '%0.4f'%((bins[i-1]+bins[i])/2)
    for j in range(counts.shape[0]):
        line = line + '\t' + str(counts[j][i-1])
    f.write(line + '\n')

f = open('cdf_' + title + '.txt', 'w')
for i in range(bins.shape[0]):
    line = '%0.4f'%(bins[i])
    for j in range(counts.shape[0]):
        if i == 0:
            line = line + '\t' + str(0)
        else:
            line = line + '\t' + str(np.sum(counts[j][0:i]))
    f.write(line + '\n')
