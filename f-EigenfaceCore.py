import cv2
import numpy as np


def EigenfaceCore(T):

    imgNum = np.shape(T)[0] #20 98304

    m = T.mean(axis = 0)
    irow,icol = np.shape(T)
    # A Matrix of centered image vectors
    A = T - m
    A = np.mat(A)

    # 得到协方差矩阵
    L = A * A.T
    # 求协方差矩阵的特征值和特征向量
    D, V = np.linalg.eig(L) #
    # print(D,V)
    # 得到的特征向量转变成list
    V = list(V.T)
    for i in range(imgNum):
        if D[i] < 1:
            V.pop(i)
    V = np.array(V)
    V = np.mat(V).T
    # print(V,D)

    Eigenfaces = A.T * V
    # print(Eigenfaces)
    # print(Eigenfaces.shape)
    return Eigenfaces,m,A
