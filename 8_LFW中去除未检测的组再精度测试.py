#coding:utf-8
import numpy as np

same=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_same')
not_same=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_not_same')
same=same.tolist()
not_same=not_same.tolist()

pairs_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw_pairs'
lwf_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw/'
f = open(pairs_path, 'r')
lines = f.readlines()[1:]
index=0
l_same=[]
l_not_same=[]
sum=0
# print lines[300]
for ii in range(len(lines)):
    if ii/300%2==0:
        # l.append(float(lines[ii].split(' ')[-1].split('\n')[0]))
        lines[ii] = lines[ii].split(' ')
        lines[ii]=lines[ii][0].split('\t')
        lines[ii][2]=lines[ii][2].split('\n')[0]
        # print lines[ii]
        # l_pic = [l[1], l[2]]
        l_same.append([lwf_path+lines[ii][0]+'/'+lines[ii][0]+'_'+lines[ii][1].zfill(4)+'.jpg',lwf_path+lines[ii][0]+'/'+lines[ii][0]+'_'+lines[ii][2].zfill(4)+'.jpg'])
    if ii/300%2==1:
        # print lines[ii],index
        lines[ii] = lines[ii].split('\t')
        # lines[ii]=lines[ii][0].split('\t')
        # lines[ii][2]=lines[ii][2].split('\n')[0]
        l_not_same.append([lwf_path+lines[ii][0]+'/'+lines[ii][0]+'_'+lines[ii][1].zfill(4)+'.jpg',lwf_path+lines[ii][2]+'/'+lines[ii][2]+'_'+lines[ii][3].split('\n')[0].zfill(4)+'.jpg'])
f.close()

l_NoDetect=[]
f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/mtcnn_lfw_NoDetect.txt','r')
f1=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/mtcnn_lfw_NoDetect_sub.txt','r')
lines2=f.readlines()
lines1=f1.readlines()
lines=lines1+lines2
lines1_c=[]
lines2_c=[]
for i in lines1:
    lines1_c.append(i.split('\n')[0])
for i in lines2:
    lines2_c.append(i.split('\n')[0])
lines1=lines1_c[:]
lines2=lines2_c[:]
for i in lines:
    l_NoDetect.append(i.split('\n')[0])
print len(l_NoDetect)
# print l_NoDetect
#
# print len(l_NoDetect)
# print type(np.array(l_NoDetect))
# a=np.array(l_NoDetect)
# f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/dlib_NoDetect','a+')
# for i in l_NoDetect:
#     f.write(i+'\n')
# f.close()

not_same_c=not_same[:]
same_c=same[:]
test=[]
tmp1=0
tmp2=0
for i in range(len(l_same)):
    for j in l_same[i]:
        if j in lines1:
            tmp1+=1

        if j in lines2:
            tmp2+=1
            print 'same[i]', same[i]
        if j in l_NoDetect:
            # print j
            # test.append(same[i])
            same_c.remove(same[i])
            # same_c[i]=0
            test.append(same[i])
            # tmp1+=1
            break
# print test
test1=[]
for i in range(len(l_not_same)):
    for j in l_not_same[i]:
        if j in lines1:
            tmp1+=1
        if j in lines2:
            tmp2+=1
            print 'not_same[i]', not_same[i]
        if j in l_NoDetect:
            not_same_c.remove(not_same[i])
            # not_same_c[i]=1
            test1.append(not_same[i])
            # tmp2+=1
            break
# f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/same_lwf','a+')
# for i in same_c:
#     # print i
#     a=np.array(i)
#     f.write(a)
# f.close()
print 'tmp1=',tmp1
print 'tmp2=',tmp2
np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_删除未检测_same',same_c)
np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_删除未检测_not_same',not_same_c)


# print same_c
print 3000-len(same_c)
print 3000-len(not_same_c)
# print not_same_c
# print 3000-len(not_same_c)
#
# print test
# print len(test)
# print test1
# print len(test1)