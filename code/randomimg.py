#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-01-12 17:41:21
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import numpy as np 
import cv2
import matplotlib.pyplot as plt

def randimg():
	img = np.random.randint(255, size=(512,521,3))
	# cv2.imshow("img",img)
	plt.imshow(img)
	print(img)
	plt.show()

def shufflepixel():
	img = cv2.imread(r"../img/2_2_RAS.jpg")
	# plt.figure()
	# plt.imshow(img)
	cv2.imshow("1",img)
	size = img.shape
	# img.flatten()
	np.random.shuffle(img)
	# new_img = np.resize(img, size)
	# plt.figure()
	cv2.imshow("2", img)
	cv2.waitKey(0)
	# plt.imshow(new_img)
	# plt.show()

if __name__ == '__main__':
	shufflepixel()
