# coding=utf-8
import kNN
import matplotlib.pyplot as plt
import numpy as np

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
print (datingDataMat,datingLabels)


# 没有样本类别标签的约会数据散点图，难以辨识图中的点究竟属于哪个样本分类
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.set_xlabel("Percentage of time spent playing video games")
ax.set_ylabel("Percentage of games played per week")


fig2 = plt.figure()
bx = fig2.add_subplot(111)
bx.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
#添加后两个参数设置点的差异化，第三个是设置点的大小，第四个设置点颜色
bx.set_xlabel("Percentage of time spent playing video games")
bx.set_ylabel("Percentage of games played per week")
plt.show()

