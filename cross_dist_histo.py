import sys
import os
import numpy as np

P=1501 #751
dict_feat = {}
for pid in range(P):
    if not os.path.isfile('train_pid_%04d.npy'%(pid)):
        continue
    dict_feat[pid] = np.load('train_pid_%04d.npy'%(pid))

list_dist = []
dict_done = {}
for pid1 in dict_feat:
    for pid2 in dict_feat:
        if pid1 == pid2:
            continue
        if pid1 in dict_done and pid2 in dict_done:
            if dict_done[pid1] == pid2 or dict_done[pid2] == pid1:
                continue
        print("====================>", pid1, pid2)
        a = dict_feat[pid1]
        b = dict_feat[pid2]
        N1 = a.shape[0]
        N2 = b.shape[0]
        for i in range(N1):
            for j in range(N2):
                dist = np.linalg.norm(a[i,:]-b[j,:])
                list_dist.append(dist)
        dict_done[pid1] = pid2
        dict_done[pid2] = pid1

title = "cross_class"
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