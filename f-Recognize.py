
import cv2
import numpy as np

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