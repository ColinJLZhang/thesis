#!/usr/bin/python
# coding:utf8

import numpy as np
import cv2

root = r"E:\postgraduate\渔机所\烟台东方海洋出差\2018_10出差\GOPR6437.MP4"
savepath = r"E:\postgraduate\论文\thesis\img"

cap = cv2.VideoCapture(root)
print("info:\n width:{}, height:{}, fps:{}".format(cap.get(3), cap.get(4), cap.get(5)))
start = 20030
count = 0
num = 60
col = 6
row = 5
step=5

img1 = np.zeros((row*108, col*192, 3))
img2 = np.zeros((row*108, col*192, 3))

def mean_flow(flow):
	res = []
	for c in flow:
		for e in c:
			res.append(np.sqrt(e[0]**2 + e[1]**2))
	mf = np.sum(res)
	return mf


if __name__ == '__main__':
    cam = cv2.VideoCapture(root)
    cam.set(0,start)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.resize(prevgray, (192,108), interpolation=cv2.INTER_CUBIC)
    flows = []

    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (192,108), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(mean_flow(flow))
        prevgray = gray
        # 绘制线
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        line = []
        for l in lines:
            if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
                line.append(l)

        cv2.polylines(img, line, 0, (0,255,255))
        cv2.imshow('flow', img)
        
        img1[count//col*108:count//col*108+108,(count%col)*192:(count%col+1)*192,:] = img
        count = count +1
        if count >= 30:
        	break

        print(count)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()

    cv2.imwrite("flow31.png",img1)
    print("mean flow:",np.mean(flows))