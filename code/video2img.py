#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-26 16:12:41
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
import matplotlib.pyplot as plt
import os 
import pickle

root = r"E:\postgraduate\渔机所\烟台东方海洋出差\2018_10出差\GOPR6437.MP4"
savepath = r"E:\postgraduate\论文\thesis\img"

cap = cv2.VideoCapture(root)
print("info:\n width:{}, height:{}, fps:{}".format(cap.get(3), cap.get(4), cap.get(5)))
start = 1000
diff1 = []
diff2 = []
count = 0
num = 60
col = 6
row = 5

cap.set(0,10000)
ret, last = cap.read()
print(last.shape)
last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
last = cv2.resize(last, (192,108),interpolation=cv2.INTER_CUBIC)
out = last
img1 = np.zeros((row*108, col*192))
img2 = np.zeros((row*108, col*192))


while cap.isOpened():
	ret,frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (192,108),interpolation=cv2.INTER_CUBIC)
	diff1.append((frame-last).sum()/(108*192))
	last = frame
	img1[count//col*108:count//col*108+108,(count%col)*192:(count%col+1)*192] = frame
	count = count +1
	if count >= 30:
		break
	print(count)

count = 0
while cap.isOpened():
	ret,frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (192,108),interpolation=cv2.INTER_CUBIC)
	diff2.append((frame-last).sum()/(108*192))
	last = frame
	img2[count//col*108:count//col*108+108,(count%col)*192:(count%col+1)*192] = frame
	count = count +1
	if count >= 30:
		break
	print(count)

print("diff1:{},diff2:{}".format(sum(diff1)/len(diff1),sum(diff2)/len(diff2)))
# cv2.imshow("img",img)
# cv2.waitKey(0)
# plt.figure(figsize=(15, 15))
# cv2.imwrite("fast.png",img1)
# cv2.imwrite("slow.png",img2)
plt.plot(diff1, label="v1")
plt.plot(np.ones(len(diff1))*sum(diff1)/len(diff1),'--', label="v1 diff mean", color='B')
plt.plot(diff2, label="v2")
plt.plot(np.ones(len(diff2))*sum(diff2)/len(diff2),'--', label="v2 diff mean", color='orange')
plt.xlabel("frame index", fontsize=18)
plt.ylabel("inner-frame difference", fontsize=18)
plt.legend()
plt.show()