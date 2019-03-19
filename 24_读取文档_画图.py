# coding: utf-8
import matplotlib.pyplot as plt

# figsize = 11, 9
# figure, ax = plt.subplots(figsize = figsize)

# x1 =[0,5000,10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
# y1=[0, 223, 488, 673, 870, 1027, 1193, 1407, 1609, 1791, 2113, 2388]
# x2 = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
# y2 = [0, 214, 445, 627, 800, 956, 1090, 1281, 1489, 1625, 1896, 2151]

# y1=[9.2883, 8.95847, 8.85225, 8.75826, 8.50599, 8.18309, 7.77338, 7.38235, 6.83893, 6.61936, 6.50865, 6.42913, 6.30992, 6.23999, 6.20566]
# y2=[9.95165, 6.57493, 4.24437, 6.01034, 13.6406, 20.7462, 32.9122, 37.9254, 58.1309, 55.7304, 58.2912, 59.1009, 64.6366, 69.5219, 72.8841]
def readData(file_path):
    # type: (object) -> object
    l=[]
    f = open(file_path, 'r')
    lines = f.readlines()
    for ii in range(len(lines)):
        if 'Iters' not in lines[ii]:
            #l.append(float(lines[ii].split(' ')[-1].split('\n')[0]))
            lines[ii]=lines[ii].replace('    ', ',')
            lines[ii] = lines[ii].replace('   ', ',')
            lines[ii] = lines[ii].replace('  ', ',')
            lines[ii] = lines[ii].replace(' ', ',')
            print lines[ii]
            try:
                print '**',lines[ii].split(',')[-1].split('\n')[0]
                # l.append(float(lines[ii].split(',')[-2]))
                l.append(float(lines[ii].split(',')[-1].split('\n')[0]))
            except:
                pass
    f.close()
    return l[:-2]
def readData_train(file_path):
    # type: (object) -> object
    l=[]
    f = open(file_path, 'r')
    lines = f.readlines()
    for ii in range(len(lines)):
        if 'LearningRate' not in lines[ii]:
            #l.append(float(lines[ii].split(' ')[-1].split('\n')[0]))
            lines[ii]=lines[ii].replace('    ', ',')
            lines[ii] = lines[ii].replace('   ', ',')
            lines[ii] = lines[ii].replace('  ', ',')
            lines[ii] = lines[ii].replace(' ', ',')
            print lines[ii]
            try:
                print '********',lines[ii].split(',')[-1].split('\n')[0]
                l.append(float(lines[ii].split(',')[-2]))
                # l.append(float(lines[ii].split(',')[-1].split('\n')[0]))
            except:
                pass
    f.close()
    return l[:-1]
# y1=readData('/home/jc/caffe-face-caffe-face/adamlenet1.log.test')
# y1=y1+readData('/home/jc/caffe-face-caffe-face/adamlenet3.log.test')
# y1=y1+readData('/home/jc/caffe-face-caffe-face/adamlenet4.log.test')

y1=readData('/home/jc/caffe-face-caffe-face/adamlenet1_dlib.log.test')
y2=readData('/home/jc/caffe-face-caffe-face/face_example/4_mtcnn/adamlenet4_2_1.log.test')
print 'y1=',y1
print 'y2=',y2
for i in y2:
    if i>10:
        print i
# 设置输出的图片大小
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)

# 在同一幅图片上画两条折线
# A, = plt.plot(x1, y1, '-r', label='A', linewidth=5.0)
# B, = plt.plot(x2, y2, 'b-.', label='B', linewidth=5.0)

# A, = plt.plot(range(0,30000,2000), y1, '-r', label='A', linewidth=5.0)
# B, = plt.plot(range(0,30000,2000), y2, '-r', label='A', linewidth=5.0)
print len(y2)
A, = plt.plot(range(0,2000*len(y1),2000), y1, color='#0000CD', linestyle='-', label='MTCNN', linewidth=2.0)
# B, = plt.plot(range(0,100*len(y2),100), y2, color='red', linestyle='-', label='TrainLoss0', linewidth=2.0)
B, = plt.plot(range(0,2000*len(y2),2000), y2, color='red', linestyle='-', label='Dlib', linewidth=2.0)
# C, = plt.plot(range(0,2000*len(y5),2000), y5, color='black', linestyle='-', label='TestLoss2', linewidth=2.0)
# D, = plt.plot(range(0,100*len(y4),100), y4, color='darkred', linestyle='-', label='TrainLoss1', linewidth=2.0)
# print range(0,30000,2000),len(range(0,30000,2000))
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }
# legend = plt.legend(handles=[A, B], prop=font1)
legend = plt.legend(handles=[A,B,], prop=font1)
# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
plt.xlabel('Iters', font2)
# plt.ylabel('TestAccuracy', font2)
plt.ylabel('Loss', font2)
plt.show()