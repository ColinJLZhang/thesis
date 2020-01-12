#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-01-12 17:41:21
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import numpy as np 
import cv2
import matplotlib.pyplot as plt

img = np.random.randint(255, size=(512,521,3))
# cv2.imshow("img",img)
plt.imshow(img)
print(img)
plt.show()