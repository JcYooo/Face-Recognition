#coding:utf-8
import numpy as np
a=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_same')
same=a.tolist()
a=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_not_same')
not_same=a.tolist()
same_sum=len(same)
not_same_sum=len(not_same)
all_sum=same_sum+not_same_sum
print len(same)
print len(not_same)

same.sort()
print same

threshold = -2
accuracy={}
accuracy_same={}
accuracy_not_same={}
max=0
while threshold <= 2 :
    sum=0
    sum_same=0
    sum_not_same=0
    for i in same:
        if i <= threshold:
            sum_same+= 1
        else:
            pass
    for i in not_same:
        if i > threshold:
            sum_not_same+= 1
        else:
            pass
    sum=sum_not_same+sum_same
    current_accuracy = sum*1.0 / all_sum
    if current_accuracy>max:
        max=current_accuracy
    accuracy[str(threshold)] = current_accuracy,sum_same*1.0/same_sum,sum_not_same*1.0/not_same_sum
    # accuracy_same[str(threshold)] = sum_same*1.0/3000
    # accuracy_not_same[str(threshold)] = sum_not_same*1.0/3000
    threshold = threshold + 0.001
print max

print sorted(accuracy.items(),key = lambda x:x[1],reverse = True)
print len(accuracy)