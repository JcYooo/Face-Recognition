#coding:utf-8
import numpy as np
import copy
a=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_same')
same=a.tolist()
a=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_not_same')
not_same=a.tolist()
# for i in range(len(same)):
#     same[i]=same[i]*0.7
# for i in range(len(not_same)):
#     not_same[i]=not_same[i]*1.3
print len(same)
print len(not_same)
l10=[]
for i in range(10):
    tmp=[]
    tmp.append(same[i*300:(i+1)*300])
    tmp.append(not_same[i*300:(i+1)*300])
    l10.append(tmp)
print len(l10),len(l10[0]),len(l10[0][1])
accuracy_sum=0
for i in l10:
    # print type(i)
    l10_c=l10[:]
    same=i[0]
    not_same=i[1]
    same.sort()


    threshold = -2
    accuracy={}
    accuracy_same={}
    accuracy_not_same={}
    max=0
    while threshold <= 2 :
        sum=0
        sum_same=0
        sum_not_same=0
        for ii in same:
            if ii <= threshold:
                sum_same+= 1
            else:
                pass
        for ii in not_same:
            if ii > threshold:
                sum_not_same+= 1
            else:
                pass
        sum=sum_not_same+sum_same
        current_accuracy = sum*1.0 / 600
        if current_accuracy>max:
            max=current_accuracy
            max_threshold=threshold
        accuracy[str(threshold)] = current_accuracy,sum_same*1.0/300,sum_not_same*1.0/300
        # accuracy_same[str(threshold)] = sum_same*1.0/3000
        # accuracy_not_same[str(threshold)] = sum_not_same*1.0/3000
        threshold = threshold + 0.001

    # print len(l10_c)
    # print i
    # print type(i)
    # print 'len(i)=',len(i)
    # for j in l10_c:
    #     if j==i:
    #         print 'harry pote'
    print max
    l10_c.remove(i)
    same=[]
    not_same=[]
    for i in l10_c:
        same+=i[0]
        not_same+=i[1]

    same.sort()
    # print same

    threshold = max_threshold
    accuracy = {}

    max = 0

    sum = 0
    sum_same = 0
    sum_not_same = 0
    for i in same:
        if i <= threshold:
            sum_same += 1
        else:
            pass
    for i in not_same:
        if i > threshold:
            sum_not_same += 1
        else:
            pass
    sum = sum_not_same + sum_same
    current_accuracy = sum * 1.0 / 5400

    accuracy_sum+=current_accuracy


accuracy=accuracy_sum/10
print 'accuracy=',accuracy


# print max
# print max_threshold
#
# print sorted(accuracy.items(),key = lambda x:x[1],reverse = True)
# print len(accuracy)