#coding:utf-8
import os
import shutil
# path_move='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data111_test/'
path='/home/jc/caffe-face-caffe-face/face_example/data/dlib_data3'
# path_test='/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data11_test/'
f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/train.txt','a+')

for _,dirs,_ in os.walk(path):
    for dir in dirs:
        dir_path=os.path.join(path,dir)
        for _,_,files in os.walk(dir_path):
            for file in files:
                file_path=os.path.join(dir_path,file)
                f.write(file_path+' '+dir+'\n')

f.close()



# f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test1.txt','a+')


"""/home/jc/caffe-face-caffe-face/face_example/data/mtcnn_data11_test/3058260/0013058260.jpg 3058260"""