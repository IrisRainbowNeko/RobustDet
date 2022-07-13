import os
import numpy as np
import cv2

filepath = ['../datas/VOCdevkit/VOC2007/JPEGImages', '../datas/VOCdevkit/VOC2012/JPEGImages']  # 数据集目录
img_list = []
for x in filepath:
    img_list.extend([os.path.join(x, f) for f in os.listdir(x)])

R_channel = 0
G_channel = 0
B_channel = 0
num = len(img_list)
for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path) / 255.0
    pix_num = np.prod(img.shape[:2])
    R_channel += np.sum(img[:, :, 2])/pix_num
    G_channel += np.sum(img[:, :, 1])/pix_num
    B_channel += np.sum(img[:, :, 0])/pix_num

    if i%200==0:
        print(f'mean: {i}/{num}')

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))

R_channel = 0
G_channel = 0
B_channel = 0
for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path) / 255.0
    pix_num = np.prod(img.shape[:2])
    R_channel += np.sum((img[:, :, 2] - R_mean) ** 2)/pix_num
    G_channel += np.sum((img[:, :, 1] - G_mean) ** 2)/pix_num
    B_channel += np.sum((img[:, :, 0] - B_mean) ** 2)/pix_num

    if i % 200 == 0:
        print(f'std: {i}/{num}')

R_var = R_channel / num
G_var = G_channel / num
B_var = B_channel / num
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))