# Multi Scale Retinex
# Reference: http://www.ipol.im/pub/art/2014/107/
import numpy as np
import cv2
import os
from gauss import gauss

# constants
sig_list = [15, 80, 250]    #list of sigma
w = 1/3                     #weight
k = (0, 0)                  #kernel size
a = 125                     #alpha
G = 192
b = -30
bet = 46                    #beta
cv2_gaussblur = False       #whether or not to use cv2.GaussianBlur

def SSR(img, sigma):
    if(cv2_gaussblur):
        return np.log10(img) - np.log10(cv2.GaussianBlur(img, k, sigma))

    out = np.zeros(img.shape)
    for i in range(img.shape[2]):
        out[:, :, i] = gauss(img[:, :, i], k, sigma)
    out = out + 1   #prevent division by zero
    return np.log10(img) - np.log10(out)
    
def MSR(img, sigmas, weight):
    retinex = np.zeros(img.shape)

    for s in sigmas:
        retinex += SSR(img, s)

    retinex = weight * retinex
    return retinex

def CRF(img, alpha, beta):
    # Color restoration function
    img_sum = np.sum(img, axis=2, keepdims=True)

    cr = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return cr


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    # Color restoration
    ep = 1   #to prevent division by zero
    img = np.float64(img) + ep

    retinex = MSR(img, sigma_list, w)    
    cl_res_fn = CRF(img, alpha, beta)    
    msrcr = G * (retinex * cl_res_fn - b)

    for i in range(msrcr.shape[2]):
        msrcr[:, :, i] = (msrcr[:, :, i] - np.min(msrcr[:, :, i])) / (np.max(msrcr[:, :, i]) - np.min(msrcr[:, :, i])) * 255
    
    msrcr = np.uint8(msrcr)  

    return msrcr

# 
file = '1.jpg'
img = cv2.imread('img/' + file)

out = MSRCR(img, sig_list, G, b, a, bet, 0.01, 0.99)

path = 'output/'
if not os.path.exists(path):
    os.makedirs(path)
path = path + 'MSRCR_'
if(cv2_gaussblur):
    path = path + 'cv2_'
iw = cv2.imwrite(path + file.split('.')[0] + '.png', out)
if(iw):
    print('image saved to ' + path)


cv2.imshow('Retinex Colour Restored', cv2.resize(out, (500,500)))
cv2.imshow('Original', cv2.resize(img, (500,500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
