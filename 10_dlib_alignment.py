#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:11:48 2017

@author: jc
"""

#!---* coding: utf-8 --*--
#!/usr/bin/python

#----------------------------------------------------------------------------------------------
#
# Description: image process functions
# Author: WIll Wu
# Company: School of MicroElectronic. SJTU
#
#-----------------------------------------------------------------------------------------

import cv2
import dlib
import numpy as np
import os
import shutil
import time


MAIN_DIR = '/home/jc/Face-Recognition-Web-demo/recognition'

PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
BASEFILE = MAIN_DIR + '/baseline/base.jpg'
BASE_LANDMARK = MAIN_DIR + '/baseline/BASE_LANDMARK.txt'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class NoFaceError(Exception):
    '''
        No face detect in a picture, when occur this situation, we need to handle 
    '''
    def __init__(self,str):
        self.str = str
    def __str__(self):
        return self.str


def getPoints(landmark):
    '''
        when alignment, we need some point to be baseline
        choose 37 43 30 48 54 as the baseline point
    '''
    Points = np.float32([[landmark[37][0],landmark[37][1]],[landmark[43][0],landmark[43][1]],[landmark[30][0],landmark[30][1]],[landmark[48][0],landmark[48][1]],[landmark[54][0],landmark[54][1]]])
    return Points


def getlandmark(im):
    '''
        detect the lanmark of a face using dlib
    '''
    image = im.copy()
    rects = detector(image,2)
    if len(rects) == 0:
        raise NoFaceError("No face detect")
    # cv2.rectangle(image,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0),2)
    # face =image2[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]
    if len(rects)>1:
        max = 0
        l_area = []
        for i in range(len(rects)):
            area = (rects[i].bottom() - rects[i].top()) * (
                rects[i].right() - rects[i].left())
            l_area.append(area)
            if max < area:
                max = area
                tmp = rects[i]
                cv2.rectangle(im,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0))
                # face =im[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]
                # cv2.imshow('im',im)
                # cv2.waitKey(0)
        rects[0]=tmp
    landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    # for j in xrange(landmark.shape[0]):
    #     pos = (landmark[j][0],landmark[j][1])
        # cv2.putText(image, str(j), pos,
        #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #             fontScale=0.4,
        #             color=(0, 0, 255))
        # cv2.circle(image, pos, 1, color=(255, 0, 0))
    # face = image[dects_sub[0].top():dects_sub[0].bottom(), dects_sub[0].left():dects_sub[0].right()]
    l=[rects[0].top(),rects[0].bottom(), rects[0].left(),rects[0]]
    ll=[]
    # for i in l:
    #     if i<0:
    #         ll.append(0)
    #     elif i>249:
    #         ll.append(249)
    #     else:
    #         ll.append(i)


    return landmark,image,l



def compute_affine_transform(refpoints, points, w = None):
    '''
        Compute the affine transform matrix
    '''
    if w == None: 
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    
    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]
    #err = 0#lstsq[1]

    #R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R

#
# def alignment(imagefile):
#     '''
#         Input: an imagefile name
#         Output: an facefile name or None
#         Function:
#                alignment a picture, if no face detect, return None
#                Otherwise, return the facefile name
#
#     '''
#     base_dir='/home/jc/caffe-face-caffe-face/face_example/data/dlib_data/'
#     size = (96, 112)
#     facefilename = imagefile.split('.')[0]+'_face.jpg'
#     image = cv2.imread(imagefile)
#     # image = cv2.resize(image,(250,250))
#
#     base_landmark = np.loadtxt(BASE_LANDMARK)
#     base_landmark *= image.shape[0]
#
#     try:
#         landmark,image_show = getlandmark(image)
#     except NoFaceError,e:
#         print 'No Detect'
#         sum+=1
#         file = imagefile.split('CASIA-WebFace/')[1]
#         newdir1 = base_dir+file.split('/')[0]
#         newfile1=base_dir+file
#         if not os.path.exists(newdir1):
#             os.mkdir(newdir1)
#         # newfile1 = os.path.join(newdir1,file)
#         shutil.copyfile(imagefile,newfile1)
#         face = image
#         cv2.imshow('face', face)
#         image=cv2.resize(image,size)
#         return
#
#
#
#     srcPoints = getPoints(landmark)
#     dstPoints = getPoints(base_landmark)
#
#     M =compute_affine_transform(dstPoints,srcPoints)
#     image2 = cv2.warpAffine(image,M,(250,250))
#     rects = detector(image2,2)
#     if len(rects) == 0:
#         print 'why'
#         print 'No Detect'
#         sum+=1
#         file = imagefile.split('CASIA-WebFace/')[1]
#         newdir1 = base_dir + file.split('/')[0]
#         newfile1 = base_dir + file
#         if not os.path.exists(newdir1):
#             os.mkdir(newdir1)
#         face = image2[rects[0].top():rects[0].bottom(), rects[0].left():rects[0].right()]
#         face = cv2.resize(face, size)
#         cv2.imwrite(newfile1,face)
#         # cv2.imshow('face',face)
#         # cv2.waitKey(0)
#
#
#
#     face =image2[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()]
#     face = cv2.resize(face,size)
#     file = imagefile.split('CASIA-WebFace/')[1]
#     newdir1 = base_dir + file.split('/')[0]
#     newfile1 = base_dir + file
#     if not os.path.exists(newdir1):
#         os.mkdir(newdir1)
#
#     cv2.imwrite(newfile1,face)
#     #return facefilename
#     # return face

if __name__ == '__main__':
    # pic_path = '/home/jc/图片/图片1.png'
    # face=alignment(pic_path)
    # cv2.destroyAllWindows()
    path = '/home/jc/caffe-face-caffe-face/face_example/data/CASIA-WebFace/'
    # path_txt='/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/dlib_cacia_1stageNoDetect.txt'
    # path_txt_sub='/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/dlib_cacia_1stageNoDetect_sub.txt'
    base_dir = '/home/jc/caffe-face-caffe-face/face_example/data/dlib_data3/'
    # f = open(path_txt, 'a+')
    # f_sub = open(path_txt_sub, 'a+')
    # sum=0
    sum1=0
    sum2=0
    sum_index=0
    # for _,dirs_move,_ in os.walk(base_dir):
    #     break
    for _,dirs,_ in os.walk(path):
        for dir in dirs:
            # if dir not in dirs_move:
            dir_path=os.path.join(path,dir)
            for _,_,files in os.walk(dir_path):
                # sum +=len(files)
                for file in files:
                    # print dir,len(files)
                    sum_index+=1
                    imagefile=os.path.join(dir_path,file)

                    size = (96, 112)
                    # facefilename = imagefile.split('.')[0] + '_face.jpg'
                    image = cv2.imread(imagefile)
                    image_c=image.copy()
                    # image = cv2.resize(image,(250,250))

                    base_landmark = np.loadtxt(BASE_LANDMARK)
                    base_landmark *= image.shape[0]

                    try:
                        landmark, image_show,dects_sub = getlandmark(image)
                    except NoFaceError, e:
                        print 'No Detect'
                        # f.write(imagefile)
                        # f.write('\n')
                        sum1 += 1
                        file = imagefile.split('CASIA-WebFace/')[1]
                        newdir1 = base_dir + file.split('/')[0]
                        newfile1 = base_dir + file
                        if not os.path.exists(newdir1):
                            os.mkdir(newdir1)
                        image = cv2.resize(image, size)
                        cv2.imwrite(newfile1,image)

                        continue

                    srcPoints = getPoints(landmark)
                    dstPoints = getPoints(base_landmark)

                    M = compute_affine_transform(dstPoints, srcPoints)
                    image2 = cv2.warpAffine(image, M, (250, 250))
                    rects = detector(image2, 2)
                    if len(rects) == 0:
                        print 'why'
                        print 'No Detect'
                        # f_sub.write(imagefile)
                        # f_sub.write('\n')
                        sum2 += 1
                        file = imagefile.split('CASIA-WebFace/')[1]
                        newdir1 = base_dir + file.split('/')[0]
                        newfile1 = base_dir + file
                        if not os.path.exists(newdir1):
                            os.mkdir(newdir1)
                        print dects_sub[0],dects_sub[1], dects_sub[2],dects_sub[3]
                        face = image_c[dects_sub[0]:dects_sub[1], dects_sub[2]:dects_sub[3]]
                        face = cv2.resize(face, size)
                        cv2.imwrite(newfile1, face)
                        continue
                    l = [rects[0].top(), rects[0].bottom(), rects[0].left(), rects[0]]
                    ll = []
                    for i in l:
                        if i < 0:
                            ll.append(0)
                        elif i > 249:
                            ll.append(249)
                        else:
                            ll.append(i)

                    face = image2[ll[0]:ll[1], ll[2]:ll[3]]
                    face = cv2.resize(face, size)
                    file = imagefile.split('CASIA-WebFace/')[1]
                    newdir1 = base_dir + file.split('/')[0]
                    newfile1 = base_dir + file
                    if not os.path.exists(newdir1):
                        os.mkdir(newdir1)

                    cv2.imwrite(newfile1, face)

                print 'sum1,sum2=',sum1,sum2
                print 'sum_index=',sum_index
            # print 'sum=',sum
    # f.close()
    # f_sub.close()
    # print 'sum=======',sum
    # shutil.move(img,'/home/jc/Desktop/test')