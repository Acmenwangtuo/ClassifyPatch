import cv2
import numpy as np  
import os
import glob

images_dir = '../cls_data/diagnosis/'


images_list = glob.glob(os.path.join(images_dir,'*.png'))

print(images_list[0])
img = cv2.imread(images_list[0],0)

img = cv2.GaussianBlur(img,(11,11),0)

canny = cv2.Canny(img,50,200) 

cv2.imwrite('./Canny.png',canny)
# img = cv2.imread("D:/lion.jpg", 0)  # 由于Canny只能处理灰度图，所以将读取的图像转成灰度图
 
 
# img = cv2.GaussianBlur(img,(3,3),0) # 用高斯平滑处理原图像降噪。若效果不好可调节高斯核大小
 
# canny = cv2.Canny(img, 50, 150)     # 调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
 
 
# cv2.imshow('Canny', canny)