#coding:utf-8
import numpy as np
import sys
import skimage
import sklearn.metrics.pairwise as pw
caffe_root = '/home/jc/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:11:48 2017

@author: jc
"""

# !---* coding: utf-8 --*--
# !/usr/bin/python

# ----------------------------------------------------------------------------------------------
#
# Description: image process functions
# Author: WIll Wu
# Company: School of MicroElectronic. SJTU
#
# -----------------------------------------------------------------------------------------

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

    def __init__(self, str):
        self.str = str

    def __str__(self):
        return self.str


def getPoints(landmark):
    '''
        when alignment, we need some point to be baseline
        choose 37 43 30 48 54 as the baseline point
    '''
    Points = np.float32(
        [[landmark[37][0], landmark[37][1]], [landmark[43][0], landmark[43][1]], [landmark[30][0], landmark[30][1]],
         [landmark[48][0], landmark[48][1]], [landmark[54][0], landmark[54][1]]])
    return Points


def getlandmark(im):
    '''
        detect the lanmark of a face using dlib
    '''
    image = im.copy()
    rects = detector(image, 2)
    if len(rects) == 0:
        raise NoFaceError("No face detect")
    # cv2.rectangle(image,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0),2)
    landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    # for j in xrange(landmark.shape[0]):
    #     pos = (landmark[j][0],landmark[j][1])
    # cv2.putText(image, str(j), pos,
    #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #             fontScale=0.4,
    #             color=(0, 0, 255))
    # cv2.circle(image, pos, 1, color=(255, 0, 0))

    return landmark, image


def compute_affine_transform(refpoints, points, w=None):
    '''
        Compute the affine transform matrix
    '''
    if w == None:
        w = [1] * (len(points) * 2)
    assert (len(w) == 2 * len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0] / w[n * 2], p[1] / w[n * 2 + 1]]
    A = []
    for n, p in enumerate(points):
        A.extend([[p[0] / w[n * 2], p[1] / w[n * 2], 0, 0, 1 / w[n * 2], 0],
                  [0, 0, p[0] / w[n * 2 + 1], p[1] / w[n * 2 + 1], 0, 1 / w[n * 2 + 1]]])

    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]
    # err = 0#lstsq[1]

    # R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R


def alignment(imagefile,sum1,sum2):
    '''
        Input: an imagefile name
        Output: an facefile name or None
        Function:
               alignment a picture, if no face detect, return None
               Otherwise, return the facefile name

    '''
    # facefilename = imagefile.split('.')[0] + '_face.jpg'
    # print 'imagefile========',imagefile
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (250, 250))

    base_landmark = np.loadtxt(BASE_LANDMARK)
    base_landmark *= image.shape[0]

    try:
        landmark, image_show = getlandmark(image)
    except NoFaceError, e:
        print 'No Detect'
        sum1+=1
        return sum1,sum2

    srcPoints = getPoints(landmark)
    dstPoints = getPoints(base_landmark)
    # print 'srcPoints====',srcPoints

    M = compute_affine_transform(dstPoints, srcPoints)
    image2 = cv2.warpAffine(image, M, (250, 250))
    rects = detector(image2, 2)
    if len(rects) == 0:
        print 'why'
        print 'No Detect'
        sum2 += 1
    return sum1,sum2

lwf_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw/'
pairs_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw_pairs'
model_def = '/home/jc/caffe-face-caffe-face/face_example/face_deploy.prototxt'
model_weights = '/home/jc/caffe-face-caffe-face/face_example/完整训练过程_原始网络/face_snapshot4/face_train_test_iter_69000.caffemodel'




if __name__=='__main__':
    f = open(pairs_path, 'r')
    lines = f.readlines()[1:]
    index=0
    l_same=[]
    l_not_same=[]
    sum1=0
    sum2=0
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
    f=open('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/LFW图片路径','a')
    print 'l_same=',l_same
    print 'l_not_same=',l_not_same
    for i in l_same:
        f.write(i)
    for i in l_not_same:
        f.write(l_not_same)
    f.close()
    l_same_result = []
    l_not_same_result = []
    l_feature = []
    print l_same
    error_path1=[]
    error_path2=[]
    for i in l_same:
        for j in i:
            # pic_path=path+i
            a,b=alignment(j,sum1,sum2)
            if b > sum2:
                sum2 = b
                error_path2.append(j)
                break
            if a > sum1:
                sum1 = a
                error_path1.append(j)
                break


        print 'sub_sum=', sum1,sum2
        print 'error_path1=',error_path1

            # l_feature=np.concatenate(l_feature,)
        # print l_feature,type(l_feature),type(l_feature[0])
        # print 'len(l_feature)============', len(l_feature)

    for i in l_not_same:
        for j in i:
            a,b = alignment(j, sum1,sum2)
            if b > sum2:
                sum2 = b
                error_path2.append(j)
                break
            if a > sum1:
                sum1 = a
                error_path1.append(j)
                break
    # print l_feature
    # print 'len(l_feature)============',len(l_feature)
    # np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/001.txt',l_feature)
    # a = np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/001.txt')
    # print len(a)
    print error_path1
    print error_path2
    print 'sum1=',sum1,
    print 'sum2=',sum2




