import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import pandas as pd
import torch


# ROI提取正样本
def positive_roi(par, path, task):
    """
    par:bbox文件
    path：需要存储正样本的路径
    task:原图
    """
    x1 = []  # 左
    x2 = []
    y1 = []  # 上
    y2 = []
    img = cv2.imread(task)

    with open(par) as txt:  # 打开文件
        for line in txt:
            line = line.strip()  # 去除多余空格
            line1 = line.split()
            line2 = list(map(lambda x: float(x), line1))  # 将list中的字符串转化为数字
            y1.append(round(line2[2]))
            y2.append(round(line2[4]))
            x1.append(round(line2[1]))
            x2.append(round(line2[3]))

    for a, b, c, d, i in zip(y1, y2, x1, x2, range(0, 24)):
        roi = img[a:b, c:d]  # 上下，左右
        # roi[150:500, 0:1000] = 0   #添加黑边
        cv2.imwrite(path + '%d.jpg' % i, roi)


# 处理数据集
# 训练集和测试集要按照正负样本的顺序命名（从0.jpg开始）
def photo(n, path0, path1):
    """
    n:训练集、测试集或验证集所含图片总数
    path0:训练集、测试集或验证集所在路径
    path1：处理后训练集、测试集或验证集存放路径
    """
    for i in range(n):
        img = cv2.imread(path0 + '%d.jpg' % i)
        img1 = cv2.resize(img, (512, 512))             # resize成（512,512）大小
        # print(img1)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转为灰色图像（此方法只支持二维图像）
        # print(img2)
        # print(img2.shape)
        cv2.imwrite(path1 + '%d.tiff' % i, img2)       # 以tiff格式存储


# 导入数据集
def loadPicture(n, path):
    """
    n:训练集、测试集或验证集所含图片总数
    path：处理后训练集、测试集或验证集存放路径
    """
    index = 0
    data = np.zeros((n, 512, 512))
    label = np.zeros((n))
    for i in np.arange(n):
        image = mpimg.imread(path + str(i) + '.tiff')
        vdata = np.zeros((512, 512))
        vdata[0:image.shape[0], 0:image.shape[1]] = image
        if i < (n / 2):
            data[index, :, :] = vdata
            label[index] = 0
            index += 1
        else:
            data[index, :, :] = vdata
            label[index] = 1
            index += 1
    return data, label


# 使用LBP方法提取图像的纹理特征
def texture_detect(n, radius, data):
    """
    n:训练集、测试集或验证集所含图片总数
    radius：半径
    data:需要提取的数据
    """
    n_point = radius * 8
    hist = np.zeros((n, 256))
    for i in np.arange(n):
        # 使用LBP方法提取图像的纹理特征
        lbp = skft.local_binary_pattern(data[i], n_point, radius, 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return hist


if __name__ == '__main__':
    photo(50, 'train1/', 'train/')
    photo(10, 'qwe/', 'test/')
    train_data, train_label = loadPicture(50, 'train/')
    test_data, test_label = loadPicture(10, 'test/')
    train_hist = texture_detect(50, 1, train_data)
    test_hist = texture_detect(10, 1, test_data)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    model = OneVsRestClassifier(svr_rbf, -1)
    clf = model.fit(train_hist, train_label)

    # 测试模型
    clf.score(test_hist, test_label)

    # 保存模型及参数
    torch.save(model, '1.pkl')
    torch.save(train_hist, 'train.mat')
    torch.save(train_label, 'label.mat')
