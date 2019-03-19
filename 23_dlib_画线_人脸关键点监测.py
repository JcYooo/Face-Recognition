#coding:utf-8
import numpy as np
import cv2
import dlib
import numpy
from skimage import io

detector = dlib.get_frontal_face_detector()
MAIN_DIR = '/home/jc/Face-Recognition-Web-demo/recognition'

PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# cv2读取图像
# img = cv2.imread('/home/jc/mtcnn/Rqdq-hmhafis2616038.jpg')
#
# # 取灰度
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
win = dlib.image_window()
img = io.imread('/home/jc/mtcnn/timg_1.jpg')

win.clear_overlay()
win.set_image(img)

 #与人脸检测程序相同,使用detector进行人脸检测 dets为返回的结果
dets = detector(img, 2)

#dets的元素个数即为脸的个数
print("Number of faces detected: {}".format(len(dets)))

#使用enumerate 函数遍历序列中的元素以及它们的下标
#下标k即为人脸序号
#left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
#top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
for k, d in enumerate(dets):
    print("dets{}".format(d))
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    k, d.left(), d.top(), d.right(), d.bottom()))

    #使用predictor进行人脸关键点识别 shape为返回的结果
    shape = predictor(img, d)

    #获取第一个和第二个点的坐标（相对于图片而不是框出来的人脸）
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),  shape.part(1)))

    #绘制特征点
    win.add_overlay(shape)

#绘制人脸框
win.add_overlay(dets)



#也可以这样来获取（以一张脸的情况为例）
#get_landmarks()函数会将一个图像转化成numpy数组，并返回一个68 x2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标。
def get_landmarks(im):

    rects = detector(im, 3)

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

#多张脸使用的一个例子
def get_landmarks_m(im):

    dets = detector(im, 3)

    #脸的个数
    print("Number of faces detected: {}".format(len(dets)))

    for i in range(len(dets)):

        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])

        for i in range(68):

            #标记点
            im[facepoint[i][1]][facepoint[i][0]] = [232,28,8]
            # im[facepoint[i][1]][facepoint[i][0]] = [255,255,0]

    return im

img=get_landmarks_m(img)
io.imsave('/home/jc/mtcnn/timg_11.jpg',img)
#打印关键点矩阵
print("face_landmark:")

print(get_landmarks(img))

#等待点击
dlib.hit_enter_to_continue()