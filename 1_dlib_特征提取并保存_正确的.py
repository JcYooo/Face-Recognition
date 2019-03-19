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
BASEFILE = MAIN_DIR + '/home/jc/mtcnn/Rqdq-hmhafis2616038.jpg'
BASE_LANDMARK = MAIN_DIR + '/baseline/BASE_LANDMARK.txt'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
sum1=0
sum2=0

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
    # print 'rects=', rects, type(rects)
    # print 'rects[0]=',rects[0]
    print rects
    if len(rects) > 1:
        max = 0
        # l_area = []
        tmp = rects[0]
        for i in range(len(rects)):
            x1 = np.clip(rects[i].bottom(), 0, 249)
            x2 = np.clip(rects[i].top(), 0, 249)
            x3 = np.clip(rects[i].left(), 0, 249)
            x4 = np.clip(rects[i].right(), 0, 249)
            area = (x2 - x1) * (x4 - x3)
            # l_area.append(area)
            if max < area:
                max = area
                tmp = rects[i]
        rects[0] = tmp

    # print 'rects======',rects

    landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    return landmark, image, rects


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


def alignment(imagefile):
    '''
        Input: an imagefile name
        Output: an facefile name or None
        Function:
               alignment a picture, if no face detect, return None
               Otherwise, return the facefile name

    '''
    # facefilename = imagefile.split('.')[0] + '_face.jpg'
    # print 'imagefile========',imagefile
    global sum1,sum2
    print 'sum1,sum2=',sum1,sum2
    size=(96, 112)
    image = cv2.imread(imagefile)
    # image = cv2.resize(image, size)

    base_landmark = np.loadtxt(BASE_LANDMARK)
    base_landmark *= image.shape[0]

    try:
        landmark, image_show, rects = getlandmark(image)
    except NoFaceError, e:
        print 'No Detect'
        sum1 += 1
        image = cv2.resize(image, size)

        return image

    srcPoints = getPoints(landmark)
    dstPoints = getPoints(base_landmark)

    M = compute_affine_transform(dstPoints, srcPoints)
    image2 = cv2.warpAffine(image, M, (250, 250))
    rects1 = detector(image2, 2)
    if len(rects1) == 0:
        print 'why'
        print 'No Detect'
        x1 = np.clip(rects[0].top(), 0, 249)
        x2 = np.clip(rects[0].bottom(), 0, 249)
        x3 = np.clip(rects[0].left(), 0, 249)
        x4 = np.clip(rects[0].right(), 0, 249)
        sum2 += 1
        # landmark, image_show, rects = getlandmark(image)
        face = image[x1:x2, x3:x4]
        # cv2.imshow('face', face)
        face = cv2.resize(face, size)
        return face
    # print rects1
    x1 = np.clip(rects1[0].top(), 0, 249)
    x2 = np.clip(rects1[0].bottom(), 0, 249)
    x3 = np.clip(rects1[0].left(), 0, 249)
    x4 = np.clip(rects1[0].right(), 0, 249)

    # print 'x1,x2,x3,x4=', x1, x2, x3, x4
    face = image2[x1:x2, x3:x4]
    # cv2.imshow('face',face)
    # cv2.waitKey(0)
    face = cv2.resize(face, size)
    # cv2.waitKey(0)
    # face = cv2.resize(face, (128, 128))
    return face

lwf_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw/'
pairs_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw_pairs'
model_def = '/home/jc/caffe-face-caffe-face/face_example/4_mtcnn/face_deploy.prototxt'
model_weights = '/home/jc/caffe-face-caffe-face/face_example/5_dib/face_snapshot1/face_train_test_iter_28000.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)
def read_image(filelist):
    '''
        read an image file, transform, normlization
    '''
    word = filelist.split('\n')
    filename = word[0]
    # print 'filename========',filename
    face = alignment(filename)
    # new_filename = filename.split('.')[0] + '****' + '.jpg'
    # cv2.imwrite(new_filename, face)
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #
    # transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    # transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    # # transformer.set_raw_scale('data', 0.00390625)      # rescale from [0, 1] to [0, 255]
    # transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    #
    # # net.blobs['data'].reshape(1,        # batch sizze
    # #                           3,         # 3-channel (BGR) images
    # #                           128, 128)  # image size is 227x227
    #
    # image = caffe.io.load_image(new_filename)
    #
    # transformed_image = transformer.preprocess('data', image)
    mu=np.ones(face.shape)*127.5
    transformed_image=(face-mu)/128
    transformed_image = np.transpose(transformed_image, (2, 0, 1))
    return transformed_image


def compute_feature(path):
    '''
        compute the feature by caffe model
    '''
    caffe.set_mode_gpu()

    # use test mode (e.g., don't perform dropout)
    X = read_image(path)
    net.blobs['data'].data[...] = X

    ### perform classification
    out = net.forward()

    # feature = np.float64(out)
    feature = np.float64(out['fc5'])
    # print feature
    # print 'feature==========', type(feature)
    return feature

# path='/home/jc/caffe-face-caffe-face/face_example/data/think/1/'
# l=['/home/jc/caffe-face-caffe-face/face_example/data/think/67/001_face.jpg','/home/jc/caffe-face-caffe-face/face_example/data/think/2/003_face.jpg']
# l_feature=[]
# for i in l:
#     # pic_path=path+i
#     l_feature.append(compute_feature(i))
# print l_feature[0]
# distance  = pw.pairwise_distances(l_feature[0], l_feature[1], metric="cosine")[0][0]
# print distance

# print compute_feature('/home/jc/caffe-face-caffe-face/face_example/data/lfw/Chung_Mong-hun/Chung_Mong-hun_0002.jpg')


if __name__=='__main__':
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
    l_same_result = []
    l_not_same_result = []
    l_feature = []
    for i in l_same:

        for j in i:
            # pic_path=path+i
            if len(l_feature)==0:
                # j='/home/jc/图片/016.jpg'
                l_feature=compute_feature(j)
                print compute_feature(j)
            else:
                l_feature=np.concatenate((l_feature,compute_feature(j)))
        # print 'sub_sum=', sum

            # l_feature=np.concatenate(l_feature,)
        # print l_feature,type(l_feature),type(l_feature[0])
        # print 'len(l_feature)============', len(l_feature)

    for i in l_not_same:
        for j in i:
            # pic_path=path+i
            l_feature = np.concatenate((l_feature, compute_feature(j)))
    # print l_feature
    # print 'len(l_feature)============',len(l_feature)
    np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/dlib_训练缩放到(-1,1)_测试lfw_正确的.txt',l_feature)
    a = np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/dlib_训练缩放到(-1,1)_测试lfw_正确的.txt')
    print len(a)
    # print 'sum=',sum




