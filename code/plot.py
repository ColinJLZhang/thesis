#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-01-02 17:05:18
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math,pickle

def plothist(img):
	# n,m, _ = img.shape
	# print(m,n)
	print(len(img))
	chans = cv2.split(img)
	colors = ('B', 'G', 'R')
	plt.figure()
	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])/4
		plt.axis([0, 256, 0, 5000])
		plt.plot(hist, color=color, label=color)
		plt.xlim([0, 256])
		plt.legend()
	plt.show()

def calentroy(img):
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print(img.shape)
	m, n = img.shape
	hist = cv2.calcHist(img, [0], None, [256], [0, 256])/(m*n)
	entr = 0
	for val in hist:
		if val:
			entr -= val * (math.log(val) / math.log(2.0))
	return entr


def video_entroy(p):
	with open(p, 'rb') as f:
		sample = pickle.load(f)
	entrl = []
	for frame in sample[:149]:
		cv2.imshow('sample', frame)
		entrl.append(calentroy(frame))
	return entrl

def mean_frame(p):
	with open(p, "rb") as f:
		data = pickle.load(f)
	fmean = []
	for frame in data[:149]:
		if len(frame.shape) == 3:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fmean.append(np.mean(gray))
	return fmean

def std2_frame(p):
	with open(p, "rb") as f:
		data = pickle.load(f)
	np.random.shuffle(data)
	fstd2m = []
	for frame in data[:149]:
		if len(frame.shape) == 3:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fstd2m.append(np.std(gray)/np.mean(gray))
	return fstd2m

def mknamelist(p):
	names = [os.path.join(p, name) for name in os.listdir(p)]
	print("Name list at {} done!".format(p))
	np.random.shuffle(names)
	return names

def mean_sample(p):
	names = mknamelist(p)
	vmean = []
	for i in range(100):
		print("Processing:#", i)
		vmean.append(np.mean(mean_frame(names[i])))
	return vmean

# def std2_sample(p):
# 	names = mknamelist(p)
# 	std2 = []
# 	for i in range(100):
# 		print("Processing:#", i)

# 		std2.append(mean_sample(names[i]))
# 	return std2	

		
if __name__ == '__main__':
	# img = cv2.imread("../img/lena_std.tif")
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print(np.std(gray)/np.mean(gray))
	# img = cv2.imread("../img/fish.JPG")
	# plothist(img)
	# print(calentroy(img))
	ne_path = r"D:\1dataset\dataset\pkl\none-eating\GH076787_2_.pkl"
	e_path = r"H:\dataset\pkl\eating\GH076787_1_.pkl"
	# ne_path = r"H:\dataset\pkl\eating"
	# e_path = r"H:\dataset\pkl\none-eating"
	ne = std2_frame(ne_path)
	print("---------------noneating finished----------------------")
	e = std2_frame(e_path)
	print("------------------eating finished----------------------")
	np.savetxt("ne-std2.txt", ne)
	np.savetxt("e-std2.txt", e)
	plt.figure()
	plt.plot(ne, label="noneating")
	plt.plot(e, label="eating")
	plt.xlabel("sample index", fontsize=18)
	plt.ylabel("sample cv values", fontsize=18)
	# plt.yticks(np.arange(50,150,10))
	plt.legend(fontsize=18)
	plt.tick_params(labelsize=18)
	plt.show()