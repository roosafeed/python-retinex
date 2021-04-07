# Multi Scale Retinex
# Reference: http://www.ipol.im/pub/art/2014/107/
import numpy as np
import cv2
import os
from gauss import gauss

cv2_gaussblur = False       #whether or not to use cv2.GaussianBlur

def SSR(img, sigma, kernel):
    if(cv2_gaussblur):
        return np.log10(img) - np.log10(cv2.GaussianBlur(img, kernel, sigma))

    out = np.zeros(img.shape)
    for i in range(img.shape[2]):
        out[:, :, i] = gauss(img[:, :, i], kernel, sigma)
    out = out + 1   #prevent division by zero
    return np.log10(img) - np.log10(out)
    
def MSR(img, sigmas, weight, kernel):
    retinex = np.zeros(img.shape)

    for s in sigmas:
        retinex += SSR(img, s, kernel)

    retinex = weight * retinex
    return retinex

def CRF(img, alpha, beta):
    # Color restoration function
    img_sum = np.sum(img, axis=2, keepdims=True)

    cr = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return cr


def MSRCR(img, sigma_list, G, b, alpha, beta, weight=0.3, kernel_size=(0,0)):
    # Color restoration
    ep = 1   #to prevent division by zero
    img = np.float64(img) + ep

    retinex = MSR(img, sigma_list, weight, kernel_size)    
    cl_res_fn = CRF(img, alpha, beta)    
    msrcr = G * (retinex * cl_res_fn - b)

    for i in range(msrcr.shape[2]):
        msrcr[:, :, i] = (msrcr[:, :, i] - np.min(msrcr[:, :, i])) / (np.max(msrcr[:, :, i]) - np.min(msrcr[:, :, i])) * 255
    
    msrcr = np.uint8(msrcr)  

    return msrcr
