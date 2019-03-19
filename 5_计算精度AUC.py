#coding:utf-8
from sklearn.metrics import auc
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import sklearn
import sklearn.metrics.pairwise as pw

def draw_roc_curve(fpr,tpr,title='cosine',save_name='roc_lfw'):
    #绘制roc曲线
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic using: '+title)
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig(save_name+'.png')

same=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_same')
not_same=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_not_same')
same=same.tolist()
not_same=not_same.tolist()

same_c=same[:]
not_same_c=not_same[:]
same.sort()
same_max=same[-1]
same_min=same[0]

not_same.sort()
not_same_max=not_same[-1]
not_same_min=not_same[0]

max=max(same_max,not_same_max)
min=min(same_min,not_same_min)
print same_max,not_same_max,same_min,not_same_min,max,min
norm_length=max-min
for i in range(len(same_c)):
    same_c[i]=(same_c[i]-min)/norm_length
for i in range(len(not_same_c)):
    not_same_c[i]=(not_same_c[i]-min)/norm_length
# print same
# print not_same
not_same=not_same_c
same=same_c
threshold = 0.01
accuracy={}
accuracy_same={}
accuracy_not_same={}
max=0
TPR=[]
FPR=[]

for i in range(len(not_same)):
    not_same[i]=1-not_same[i]
for i in range(len(same)):
    same[i]=1-same[i]
print 'not_same====',not_same
print 'same====',same
labels=[0]*len(same)+[1]*len(not_same)
print len(same)
print len(not_same)
distance_norm=not_same+same

fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, distance_norm)
# print 'fpr===',fpr
# print 'tpr===',tpr
# print 'thresholds=',thresholds
draw_roc_curve(fpr, tpr, title='cosine',save_name='lfw_evaluate')
print metrics.auc(fpr, tpr)
print sklearn.metrics.average_precision_score(labels, distance_norm)



















# import numpy as np
# from sklearn import metrics
# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# print(fpr,'\n',tpr,'\n',thresholds)
# print(metrics.auc(fpr,tpr))

