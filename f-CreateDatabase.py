import cv2
import numpy as np


#数据处理，将图片存入矩阵中
def CreateDatabase():
    TrainNum = 20
    T = np.zeros((20,98304))
    for i in range(0,TrainNum):

        imgPath = '..\\TrainDatabase\\'+str(i+1)+'.jpg'
        # print(imgPath)
        img = cv2.imread(imgPath,0)
        irow = img.shape[0]
        icol = img.shape[1]

        # imgVector = np.zeros((1,irow*icol))
        imgVector = np.reshape(img,(1,irow*icol))
        # print(imgVector)
        # imgVector = imgVector.T
        T[i,:] = imgVector
        # print(i)
    return T