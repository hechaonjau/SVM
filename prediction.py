import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVR
from skimage import feature as skft
import torch
from train import photo, loadPicture, texture_detect


# 预测
def prediction(n,path,path1):
    """
    n:验证图片数量
    path：验证数据集路径
    path1:处理后的数据集存放路径
    """
    train_hist = torch.load('train.mat')
    train_label = torch.load('label.mat')
    photo(n, path, path1)
    mdata, mlabel = loadPicture(n, path1)
    hist = texture_detect(n, 1, mdata)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model = torch.load('1.pkl')
    clf = model.fit(train_hist, train_label)

    predicted = model.predict(hist)
    print(predicted)


prediction()
