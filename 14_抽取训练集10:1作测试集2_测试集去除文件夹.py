#coding:utf-8
import os
import shutil
# path='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data1'
path='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data11_test/'
path_move='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data111_test/'
# f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test.txt','a+')
for _,dirs,_ in os.walk(path):
    for dir in dirs:
        dir_path=os.path.join(path,dir)
        for _,_,files in os.walk(dir_path):
            for file in files:
                file_path=os.path.join(dir_path,file)
                newpath=os.path.join(path_move,file)
                shutil.copy(file_path,newpath)