#coding:utf-8
import os
import shutil
# path_move='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data111_test/'
f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test.txt','r')
lines=f.readlines()
f.close()
for i in range(len(lines)):
    lines[i]=lines[i].split('mtcnn_data11_test')[0]+'mtcnn_data_test/'+lines[i].split('mtcnn_data11_test')[1].split('/')[-1]
f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test1.txt','a+')
f.writelines(lines)
f.close()

"""/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data11_test/3058260/0013058260.jpg 3058260"""