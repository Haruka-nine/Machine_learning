# coding=utf-8
import os
import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


#2-1 k-近邻算法
def classify0(intX, dataSet, labels,k ):
    dataSetSize = dataSet.shape[0]  #输出行的个数，也就是数据的数量
    diffMat = np.tile(intX,(dataSetSize,1)) - dataSet #tile是将intX进行dataSize-1次的复制 ，然后相减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #每一行数的和
    distances = sqDistances**0.5
    sortedDistIndicies  = distances.argsort() #argsort()是进行从小到大的排序，返回原数组的索引
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#2-2 将文本记录转换Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # readlines是读取文件所有行，返回一个list，list的元素就是行
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))   #生成一个numberOfLines行，3列的全是0的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()    # strip--没写参数，就是删除前后的空格，写了参数，就是删除前后的对应参数字符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector


#2-2-3归一化数值：我们的三个数值如果相差很大，玩游戏时间就1 2 小时，飞行里程则是2 3w公里，那我进行近邻算法时飞行里程就会占很大比例，但这是不对的
# newValue = (oldValue-min)/(max-min)

#2-3 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#2-4 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.08
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount +=1.0
        print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))


#2-5 约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])


#将图像转换为向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#2-6手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits') #读取文件目录
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)   #添加对应的label
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)  #将文件转化为向量，形成mat
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest =img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print ("the classifier came bake with : %d,the real answer is :%d"%(classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print ("\nthe total number of errors is : %d"%errorCount)
    print ("\nthe total error rate is : %f"%(errorCount/float(mTest)))


