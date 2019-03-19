#coding:utf-8
import os
import shutil
train_path='/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test1.txt'
new='/home/jc/caffe-face-caffe-face/face_example/data/new'
f=open(train_path,'r')
lines=f.readlines()
'''/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data_test/0060858776.jpg 0858776'''
for i in lines:
    print i
    print i
    newdir=os.path.join(new,i.split('/')[-1].split(' ')[-1]).split('\n')[0]
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    dir=i.split('/')[-1].split(' ')[-1]
    print 'dir===',dir
    # print '**********',i.split('/')[-1].split(dir)[0]
    newpath=os.path.join(newdir,i.split('/')[-1].split(dir)[0].split(' ')[0])
    print newpath

    shutil.copy(i.split(' ')[0],newpath)
