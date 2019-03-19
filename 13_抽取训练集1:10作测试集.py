#coding:utf-8
import os
import shutil
path='/home/jc/caffe-face-caffe-face/face_example/data/dlib_data3'
path_test='/home/jc/caffe-face-caffe-face/face_example/data/dlib_data3_test/'
f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/test.txt','a+')
for _,dirs,_ in os.walk(path):
    for dir in dirs:
        dir_path=os.path.join(path,dir)
        for _,_,files in os.walk(dir_path):
            break
        print 'len(files)=',len(files)
        for i in xrange(len(files)/10):
            file=files[i*10]
            file_path=os.path.join(dir_path,file)
            newdir=path_test
            # newfile=os.path.join(newdir,file.split('.')[0]+'00'+dir+'.jpg')
            newfile=os.path.join(newdir,file)
            # if not os.path.exists(newdir):
            #     os.makedirs(newdir)
            shutil.move(file_path,newfile)
            f.write(newfile+' '+dir+'\n')