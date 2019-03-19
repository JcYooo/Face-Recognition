#coding:utf-8
import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
MAIN_DIR = '/home/jc/Face-Recognition-Web-demo/recognition'

PREDICTOR_PATH = MAIN_DIR + '/model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# cv2读取图像
img = cv2.imread('/home/jc/图片/人脸/2033466575.jpg')

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
rects = detector(img_gray, 2)
print rects
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)
        # x1 = np.clip(rects[i].top(), 0, 249)
        # x2 = np.clip(rects[i].bottom(), 0, 249)
        # x3 = np.clip(rects[i].left(), 0, 249)
        # x4 = np.clip(rects[i].right(), 0, 249)
        # print '(x3,x1),(x4,x2)=',(x3,x1),(x4,x2)
        # print rects[i]
        # print rects[i].top()
        # print rects[i].bottom()
        # cv2.rectangle(img,(rects[i].left(),rects[i].bottom()),(rects[i].right(),rects[i].top()),(255, 255, 0),2)

        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 6, (0, 255, 0),-1)
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.6, (0, 0, 255), 2,cv2.LINE_AA)


cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.imwrite('/home/jc/图片/人脸/20334665751.jpg',img )
cv2.waitKey(0)