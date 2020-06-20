#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-22 13:41:17
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import os
import numpy as np
import matplotlib.pyplot as plt

def framediff():
	data = np.loadtxt("ne-std2.txt")
	data = data * 255
	data.astype(int)
	plt.plot(data)
	plt.xlabel("index of randoms images")
	plt.ylabel("gray mean values")
	plt.title("random frame mean values")
	plt.ylim([0, 256])
	plt.show()
	return 0

if __name__ == '__main__':
	print(framediff())