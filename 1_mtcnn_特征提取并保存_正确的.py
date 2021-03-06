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

minsize = 20

caffe_model_path = "/home/jc/mtcnn/model"

threshold = [0.6, 0.7, 0.7]
factor = 0.709

caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)

path = '/home/jc/caffe-face-caffe-face/face_example/data/CASIA-WebFace/'
path_txt = '/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/mtcnn_lfw_NoDetect.txt'
path_txt_sub = '/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt/mtcnn_lfw_NoDetect_sub.txt'
f1 = open(path_txt, 'a+')
f1_sub = open(path_txt_sub, 'a+')
sum1 = 0
sum2 = 0
index_index = 0

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


def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        # print("reshape of reg")
        pass  # reshape of reg
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h

    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    # print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy()  # shit, value parameter!!!
    # print('#################')
    # print('boxes', boxes)
    # print('w,h', w, h)

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    # print('tmph', tmph)
    # print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1)
    dx = np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)

    # print("dy"  ,dy )
    # print("dx"  ,dx )
    # print("y "  ,y )
    # print("x "  ,x )
    # print("edy" ,edy)
    # print("edx" ,edx)
    # print("ey"  ,ey )
    # print("ex"  ,ex )


    # print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    # print('bboxA', bboxA)
    # print('w', w)
    # print('h', h)
    # print('l', l)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # read s using I

    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x

    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet

        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    # print("dx1.shape", dx1.shape)
    # print('map.shape', map.shape)


    score = map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T  # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T  # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    # print('(x,y)',x,y)
    # print('score', score)
    # print('reg', reg)

    return boundingbox_out.T


def drawBoxes(im, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
    return im


from time import time

_tstart_stack = []


def tic():
    _tstart_stack.append(time())


def toc(fmt="Elapsed: %s s"):
    print(fmt % (time() - _tstart_stack.pop()))


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0, 9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0 / minsize
    minl = minl * m

    # total_boxes = np.load('total_boxes.npy')
    # total_boxes = np.load('total_boxes_242.npy')
    # total_boxes = np.load('total_boxes_101.npy')


    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        if fastresize:
            im_data = (img - 127.5) * 0.0078125  # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws, hs))  # default is bilinear
        else:
            im_data = cv2.resize(img, (ws, hs))  # default is bilinear
            im_data = (im_data - 127.5) * 0.0078125  # [0,255] -> [-1,1]
        # im_data = imResample(img, hs, ws); print("scale:", scale)


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype=np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()

        boxes = generateBoundingBox(out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            # print(boxes[4:9])
            # print('im_data', im_data[0:5, 0:5, 0], '\n')
            # print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)

    # np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    # print("[1]:",total_boxes.shape[0])
    # print(total_boxes)
    # return total_boxes, []


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        # print("[2]:",total_boxes.shape[0])

        # revise and convert to square
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        t5 = total_boxes[:, 4]
        total_boxes = np.array([t1, t2, t3, t4, t5]).T
        # print("[3]:",total_boxes.shape[0])
        # print(regh)
        # print(regw)
        # print('t1',t1)
        # print(total_boxes)

        total_boxes = rerec(total_boxes)  # convert box to square
        # print("[4]:",total_boxes.shape[0])

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
        # print("[4.5]:",total_boxes.shape[0])
        # print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    # print(total_boxes.shape)
    # print(total_boxes)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        # print('tmph', tmph)
        # print('tmpw', tmpw)
        # print("y,ey,x,ex", y, ey, x, ex, )
        # print("edy", edy)

        # tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3))  # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) + 1, int(tmpw[k]) + 1, 3))

            # print("dx[k], edx[k]:", dx[k], edx[k])
            # print("dy[k], edy[k]:", dy[k], edy[k])
            # print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            # print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = img[int(y[k]):int(ey[k]) + 1,
                                                                          int(x[k]):int(ex[k]) + 1]
            # print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            # print("tmp", tmp.shape)

            tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))
            # tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            # print('tempimg', tempimg[k,:,:,:].shape)
            # print(tempimg[k,0:5,0:5,0] )
            # print(tempimg[k,0:5,0:5,1] )
            # print(tempimg[k,0:5,0:5,2] )
            # print(k)

        # print(tempimg.shape)
        # print(tempimg[0,0,0,:])
        tempimg = (tempimg - 127.5) * 0.0078125  # done in imResample function wrapped by python

        # np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        # print(tempimg[0,:,0,0])

        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        # print(out['conv5-2'].shape)
        # print(out['prob1'].shape)

        score = out['prob1'][:, 1]
        # print('score', score)
        pass_t = np.where(score > threshold[1])[0]
        # print('pass_t', pass_t)

        score = np.array([score[pass_t]]).T
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
        # print("[5]:",total_boxes.shape[0])
        # print(total_boxes)

        # print("1.5:",total_boxes.shape)

        mv = out['conv5-2'][pass_t, :].T
        # print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            # print('pick', pick)
            if len(pick) > 0:
                total_boxes = total_boxes[pick, :]
                # print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                # print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                # print("[8]:",total_boxes.shape[0])

        #####
        # 2 #
        #####
        # print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage

            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

            # print('tmpw', tmpw)
            # print('tmph', tmph)
            # print('y ', y)
            # print('ey', ey)
            # print('x ', x)
            # print('ex', ex)


            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1] = img[int(y[k]):int(ey[k]) + 1,
                                                                              int(x[k]):int(ex[k]) + 1]
                tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg - 127.5) * 0.0078125  # [0,255] -> [-1,1]

            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()

            score = out['prob1'][:, 1]
            points = out['conv6-3']
            pass_t = np.where(score > threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
            # print("[9]:",total_boxes.shape[0])

            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:, 3] - total_boxes[:, 1] + 1
            h = total_boxes[:, 2] - total_boxes[:, 0] + 1

            points[:, 0:5] = np.tile(w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:, 0], (5, 1)).T - 1
            points[:, 5:10] = np.tile(h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:, 1], (5, 1)).T - 1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:, :])
                # print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')

                # print(pick)
                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    # print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    # print("3:",total_boxes.shape)

    # print 'total=',total_boxes
    for j in range(len(total_boxes)):
        for i in range(len(total_boxes[j])):
            if 0 > total_boxes[j][i]:
                total_boxes[j][i] = 0
            if 249 < total_boxes[j][i]:
                total_boxes[j][i] = 249
    if len(total_boxes) > 0:
        max = 0
        index = 0
        for i in range(len(total_boxes)):
            area = (total_boxes[i][3] - total_boxes[i][1]) * (
                total_boxes[i][2] - total_boxes[i][0])
            if max < area:
                max = area
                # tmp = total_boxes[i]
                index = i

        # print 'total_boxes[index]=',total_boxes[index]
        total_boxes = np.array([total_boxes[index].tolist()])
        points = np.array([points[index].tolist()])

    return total_boxes, points


def initFaceDetector():
    minsize = 20
    caffe_model_path = "/home/duino/iactive/mtcnn/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)
    return (minsize, PNet, RNet, ONet, threshold, factor)


def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]

    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:, :, 2].copy()
    img_matlab[:, :, 2] = img_matlab[:, :, 0]
    img_matlab[:, :, 0] = tmp

    # tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    # toc()
    containFace = (True, False)[boundingboxes.shape[0] == 0]
    return containFace, boundingboxes
#------------------------------------------------------------------------------------------------

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


def alignment(imagefile):
    '''
        Input: an imagefile name
        Output: an facefile name or None
        Function:
               alignment a picture, if no face detect, return None
               Otherwise, return the facefile name

    '''
    global sum1
    global sum2
    global index_index
    global f1
    global f1_sub
    index_index+=1
    size = (96, 112)

    img = cv2.imread(imagefile)
    img_matlab = img.copy()
    tmp = img_matlab[:, :, 2].copy()
    img_matlab[:, :, 2] = img_matlab[:, :, 0]
    img_matlab[:, :, 0] = tmp

    # check rgb position
    # tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    # print 'boundingboxes, points =',boundingboxes, points
    if len(boundingboxes) == 0:
        f1.write(imagefile)
        f1.write('\n')
        print imagefile
        sum1 += 1
        face = cv2.resize(img_matlab, size)
        # l_whole.append(imagefile)
    else:
        srcPoints = np.float32(
            [[points[0][0], points[0][5]], [points[0][1], points[0][6]],
             [points[0][2], points[0][7]],
             [points[0][3], points[0][8]], [points[0][4], points[0][9]]])
        dstPoints = np.float32(
            [[101.09627533, 96.81816864], [152.47973633, 97.22710419], [127.95368195, 120.1177063],
             [104.06752014, 147.86920166], [149.44421387, 149.58815002]])

        M = compute_affine_transform(dstPoints, srcPoints)
        image2 = cv2.warpAffine(img, M, (250, 250))
        # cv2.imshow('img',image2)
        # cv2.waitKey(0)

        boundingboxes1, points1 = detect_face(image2, minsize, PNet, RNet, ONet, threshold, False,
                                              factor)
        if len(boundingboxes1) == 0:
            f1_sub.write(imagefile)
            f1_sub.write('\n')
            sum2 += 1
            face = img[int(boundingboxes[0][1]):int(boundingboxes[0][3]),
                   int(boundingboxes[0][0]):int(boundingboxes[0][2])]
            face = cv2.resize(face, size)
        else:
            face = image2[int(boundingboxes1[0][1]):int(boundingboxes1[0][3]),
                   int(boundingboxes1[0][0]):int(boundingboxes1[0][2])]
            face = cv2.resize(face, size)
    print 'sum1,sum2=', sum1, sum2
    print 'index=', index_index
    return face

lwf_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw/'
pairs_path='/home/jc/caffe-face-caffe-face/face_example/data/lfw_pairs'
model_def = '/home/jc/caffe-face-caffe-face/face_example/4_mtcnn/face_deploy.prototxt'
model_weights = '/home/jc/caffe-face-caffe-face/face_example/1/face_snapshot4_2_3/face_train_test_iter_28000.caffemodel'

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
    # sum1 = 0
    # sum2 = 0
    # index_index = 0
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
    print l_feature
    print 'len(l_feature)============',len(l_feature)
    np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4测试lfw_抽取特征向量.txt',l_feature)
    a = np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4测试lfw_抽取特征向量.txt')
    print len(a)
    # print 'sum=',sum
    f.close()
    f1.close()
    f1_sub.close()




