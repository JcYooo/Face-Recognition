#coding:utf-8
import os

path='/home/jc/caffe-face-caffe-face/face_example/data/dlib_data3/'
index=0
for _,dirs,_ in os.walk(path):
    for dir in dirs:
        print dir
        newname=str(index)
        newname_path=os.path.join(path,newname)
        old_path=os.path.join(path,dir)
        print old_path
        print newname_path
        os.rename(old_path,newname_path)
        index+=1
