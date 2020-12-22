# 整合代码


prompt = "Enter test image name (a number between 1 to 10):"
dlg_title = "Input of PCA-Based Face Recognition System"

num_lines= 1
d = 1

import  cv2
import numpy as np


from tkinter import *


#数据处理，将图片存入矩阵中
def CreateDatabase():
    TrainNum = 20
    T = np.zeros((20,98304))
    for i in range(0,TrainNum):

        imgPath = 'TrainDatabase\\'+str(i+1)+'.jpg'
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

def recognize2(TestImage,T,Eigenface ,m ,A):
    # m = T.mean(0)

    # diffMatrix = T - m
    imgSize, num = np.shape(Eigenface)

    projectedImages = Eigenface.T * A.T
    img = cv2.imread(TestImage,0)

    irow = img.shape[0]
    icol = img.shape[1]
    imgVector = np.reshape(img, (1, irow * icol))
    imgVector = np.squeeze(imgVector)
    # 求最小欧式距离对应的图片
    Difference = imgVector - m
    # 这里将numpy(98304,1)的数据转换成(98304,)
    #否则会数据超限
    Difference = np.array(Difference)
    Difference = np.mat(Difference)

    projectedTestImage = Eigenface.T * Difference.T

    Euc_dist = []
    for i in range(0, num):
        q = projectedImages[:, i]
        #计算最小距离
        temp = np.linalg.norm(projectedTestImage - q)
        #欧氏距离平方计算
        Euc_dist.append(temp*temp)

    Euc_dist_min = min(Euc_dist)
    index = Euc_dist.index(Euc_dist_min)
    # 返回匹配的图像编号
    return index + 1


def imgRead(TestImage):
    T = CreateDatabase()
    Eigenface,m,A = EigenfaceCore(T)
    id =  recognize2(TestImage,T,Eigenface,m,A)
    return id





master = Tk()
e = Entry(master)
e.pack()

e.focus_set()

def callback():
    # print(e.get()) # This is the text you may want to use later
    x = e.get()
    TestImage = 'TestDatabase\\'+str(x)+'.jpg'
    im = cv2.imread(TestImage)

    idx = imgRead(TestImage)
    s_img = 'TrainDatabase\\'+str(idx)+'.jpg'
    cv2.imshow("original",im)
    cv2.imshow("searched",cv2.imread(s_img))
    # print(TestImage)

master.title('PCA')
Label(master, text="请输入要匹配的图像的编号",font=('Arial 12 bold'),width=20,height=5).pack()
b = Button(master, text = "OK", width = 10, command = callback).pack()


mainloop()
