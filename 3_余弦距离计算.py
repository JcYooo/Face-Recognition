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
# import os
# import shutil
# import time
#
# MAIN_DIR = '/home/jc/Face-Recognition-Web-demo/recognition'
#
# PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
# BASEFILE = MAIN_DIR + '/baseline/base.jpg'
# BASE_LANDMARK = MAIN_DIR + '/baseline/BASE_LANDMARK.txt'
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(PREDICTOR_PATH)
#
#
# class NoFaceError(Exception):
#     '''
#         No face detect in a picture, when occur this situation, we need to handle
#     '''
#
#     def __init__(self, str):
#         self.str = str
#
#     def __str__(self):
#         return self.str
#
#
# def getPoints(landmark):
#     '''
#         when alignment, we need some point to be baseline
#         choose 37 43 30 48 54 as the baseline point
#     '''
#     Points = np.float32(
#         [[landmark[37][0], landmark[37][1]], [landmark[43][0], landmark[43][1]], [landmark[30][0], landmark[30][1]],
#          [landmark[48][0], landmark[48][1]], [landmark[54][0], landmark[54][1]]])
#     return Points
#
#
# def getlandmark(im):
#     '''
#         detect the lanmark of a face using dlib
#     '''
#     image = im.copy()
#     rects = detector(image, 2)
#     if len(rects) == 0:
#         raise NoFaceError("No face detect")
#     # cv2.rectangle(image,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),(0,255,0),2)
#     landmark = np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
#     # for j in xrange(landmark.shape[0]):
#     #     pos = (landmark[j][0],landmark[j][1])
#     # cv2.putText(image, str(j), pos,
#     #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#     #             fontScale=0.4,
#     #             color=(0, 0, 255))
#     # cv2.circle(image, pos, 1, color=(255, 0, 0))
#
#     return landmark, image
#
#
# def compute_affine_transform(refpoints, points, w=None):
#     '''
#         Compute the affine transform matrix
#     '''
#     if w == None:
#         w = [1] * (len(points) * 2)
#     assert (len(w) == 2 * len(points))
#     y = []
#     for n, p in enumerate(refpoints):
#         y += [p[0] / w[n * 2], p[1] / w[n * 2 + 1]]
#     A = []
#     for n, p in enumerate(points):
#         A.extend([[p[0] / w[n * 2], p[1] / w[n * 2], 0, 0, 1 / w[n * 2], 0],
#                   [0, 0, p[0] / w[n * 2 + 1], p[1] / w[n * 2 + 1], 0, 1 / w[n * 2 + 1]]])
#
#     lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
#     h11, h12, h21, h22, dx, dy = lstsq[1]
#     # err = 0#lstsq[1]
#
#     # R = np.array([[h11, h12, dx], [h21, h22, dy]])
#     # The row above works too - but creates a redundant dimension
#     R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
#     return R
#
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
#     # facefilename = imagefile.split('.')[0] + '_face.jpg'
#     # print 'imagefile========',imagefile
#     image = cv2.imread(imagefile)
#     image = cv2.resize(image, (250, 250))
#
#     base_landmark = np.loadtxt(BASE_LANDMARK)
#     base_landmark *= image.shape[0]
#
#     try:
#         landmark, image_show = getlandmark(image)
#     except NoFaceError, e:
#         print 'No Detect'
#         # dir1 = imagefile.split('/')[5]
#         # newdir1 = os.path.join('/home/jc/Desktop/think', dir1)
#         # if not os.path.exists(newdir1):
#         #     os.mkdir(os.path.join('/home/jc/Desktop/think', dir1))
#         # newfile1 = os.path.join(newdir1, facefilename.split('/')[6])
#         # shutil.copyfile(imagefile,newfile1)
#         # face = image
#         # cv2.imshow('face', face)
#         image = cv2.resize(image, (128, 128))
#         return image
#
#     srcPoints = getPoints(landmark)
#     dstPoints = getPoints(base_landmark)
#     # print 'srcPoints====',srcPoints
#
#     M = compute_affine_transform(dstPoints, srcPoints)
#     image2 = cv2.warpAffine(image, M, (250, 250))
#     # cv2.imshow('image2',image2)
#     # cv2.waitKey(0)
#     rects = detector(image2, 2)
#     if len(rects) == 0:
#         print 'why'
#         print 'No Detect'
#         # dir1 = imagefile.split('/')[5]
#         # newdir1 = os.path.join('/home/jc/Desktop/think', dir1)
#         # if not os.path.exists(newdir1):
#         #     os.mkdir(os.path.join('/home/jc/Desktop/think', dir1))
#         # newfile1 = os.path.join(newdir1, facefilename.split('/')[6])
#         # shutil.copyfile(imagefile,newfile1)
#         # cv2.imshow('face',face)
#         # cv2.waitKey(0)
#         image2 = cv2.resize(image2, (128, 128))
#         return image2
#
#     # print 'rects==',rects
#     # print 'rects[0].top():==',rects[0].top(),'rects[0].bottom():==',rects[0].bottom(),'rects[0].left():==',rects[0].left(),'rects[0].right():==',rects[0].right()
#     # print image2.shape
#     face = image2[rects[0].top():rects[0].bottom(), rects[0].left():rects[0].right()]
#     try:
#         face = cv2.resize(face, (128, 128))
#     except:
#         face = image2[rects[1].top():rects[1].bottom(), rects[1].left():rects[1].right()]
#         face = cv2.resize(face, (128, 128))
#     # cv2.imshow('face',face)
#     # cv2.waitKey(0)
#     # face = cv2.resize(face, (128, 128))
#     return face
#
# # lwf_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw/'
# # pairs_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw_pairs'
# # model_def = '/home/jc/caffe-face-caffe-face/face_example/face_deploy.prototxt'
# # model_weights = '/home/jc/caffe-face-caffe-face/face_example/face_snapshot4/face_train_test_iter_69000.caffemodel'
# #
# # net = caffe.Net(model_def,  # defines the structure of the model
# #                 model_weights,  # contains the trained weights
# #                 caffe.TEST)
# def read_image(filelist):
#     '''
#         read an image file, transform, normlization
#     '''
#     X = np.empty((1, 1, 128, 128))
#     word = filelist.split('\n')
#     filename = word[0]
#     # print 'filename========',filename
#     face=alignment(filename)
#     new_filename=filename.split('.')[0]+'****'+'.jpg'
#     cv2.imwrite(new_filename,face)
#     # cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     # face.astype(np.float32)
#     # face /= 255
#
#     im1 = skimage.io.imread(new_filename, as_grey=True)
#
#     # 归一化
#     image = skimage.transform.resize(im1, (128, 128)) * 255
#     X[0, :, :, :] = image[:, :]
#
#     X = X * 0.00390625
#     return X
#
#
# def compute_feature(path):
#     '''
#         compute the feature by caffe model
#     '''
#     caffe.set_mode_gpu()
#
#     # use test mode (e.g., don't perform dropout)
#     X = read_image(path)
#     out = net.forward_all(data=X)
#
#     # feature = np.float64(out)
#     feature = np.float64(out['fc5'])
#     # print 'feature==========', len(feature)
#     return feature


l_feature=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/PCA_1_4测试lfw_抽取特征向量.txt')
l_same_result = []
l_same=[]
l_not_same=[]

for i in xrange(0,6000,2):
    distance = pw.pairwise_distances([l_feature[i]], [l_feature[i + 1]], metric="cosine")[0][0]
    l_same.append(distance)
for i in xrange(6000,12000,2):
    distance = pw.pairwise_distances([l_feature[i]], [l_feature[i + 1]], metric="cosine")[0][0]
    l_not_same.append(distance)
    print 'distance=========',distance

# for i in xrange(0,12000,2):
#     distance = pw.pairwise_distances([l_feature[i]], [l_feature[i + 1]], metric="cosine")[0][0]
#     if i/600%2==0:
#         l_same.append(distance)
#     else:
#         l_not_same.append(distance)

print len(l_same),len(l_not_same)
print l_same
print l_not_same
np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_not_same',np.array(l_not_same))
np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4余弦距离_same',np.array(l_same))

# for i in l_same:
#     l_feature=[]
#     for j in i:
#         # pic_path=path+i
#         l_feature.append(compute_feature(j))
#     print type(l_feature[0])
#
#     distance  = pw.pairwise_distances(l_feature[0], l_feature[1], metric="cosine")[0][0]
#     # if distance>0.5:
#     #     print 'same:',distance,':',i[0].split('/')[-1],i[1].split('/')[-1]
#     l_same_result.append(distance)