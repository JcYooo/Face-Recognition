#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
#X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/1_4测试lfw_抽取特征向量.txt')
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')
# plt.show()
#先不降维，只对数据进行投影，看看投影后的三个维度的方差分布
from sklearn.decomposition import PCA
pca1 = PCA(n_components=450)
pca1.fit(X)

#将降维后的2维数据进行可视化
X_new = pca1.transform(X)
print len(X_new),len(X_new[0])
np.savetxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/PCA_1_4测试lfw_抽取特征向量.txt',X_new)
a=np.loadtxt('/home/jc/pycharm-2017.2.3/PycharmProjects/demoDetect/毕设/txt_final/PCA_1_4测试lfw_抽取特征向量.txt')
print len(a),len(a[0])
# plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
# plt.show()
