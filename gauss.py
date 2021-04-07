# GaussianBlur

import cv2
import numpy as np
import math
import os

def gfunc(x,y,sigma):
    return (math.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2, j-size[1]//2, sigma )
    return out/np.sum(out)



def conv(img, filter):
    H, W = img.shape
    fH, fW = filter.shape

    output = np.zeros((H, W))

    x = int(fH/2)

    for k in range(x):
        img = np.insert(img, (0, img.shape[0]), 0, axis=0)
        img = np.insert(img, (0, img.shape[1]), 0, axis=1)

    for h in range(H):
            for w in range(W):
                output[h, w] = np.sum(filter * img[h:h+fH , w:w+fW])

    return output


# handler fn
def gauss(img, kernel=(1,1), sigma=1):
    gaussianFilter = gaussFilter(kernel, sigma)
    gaussian_image = conv(img, gaussianFilter)
    return gaussian_image
