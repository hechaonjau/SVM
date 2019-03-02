import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVR
from skimage import feature as skft
import torch
from train import photo, loadPicture, texture_detect


# 预测
def prediction():
    train_hist = torch.load('train.mat')
    train_label = torch.load('label.mat')
    photo(6, '1/', 'hist/')
    mdata, mlabel = loadPicture(6, 'hist/')
    hist = texture_detect(6, 1, mdata)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model = torch.load('1.pkl')
    clf = model.fit(train_hist, train_label)

    predicted = model.predict(hist)
    print(predicted)


prediction()
