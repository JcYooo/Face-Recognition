#coding:utf-8
import shutil
import os
import cv2

path='/home/jc/caffe-face-caffe-face/face_example/data/think/'
for _,dirs,_ in os.walk(path):
    for dir in dirs:
        dir_path=path+dir
        for _,_,files in os.walk(dir_path):
            for file in files:
                file_path=os.path.join(dir_path,file)
                im=cv2.imread(file_path)
                if im.shape[0] != 128:
                    new_path='/home/jc/caffe-face-caffe-face/face_example/data/think_未剪切移动/'+dir
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    print file_path
                    shutil.move(file_path,new_path+'/'+file)

