#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:40:33 2020

@author: jaimecalderonocampo
Segmentacion de venas 
"""
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, util, exposure, morphology, filters
from scipy.interpolate import lagrange


Ojo = io.imread('imagenes/01_test.fit')
Ojo_green = Ojo[:, :, 1]
Ojo_invesa = util.invert(Ojo_green)
Ojo_EHA = exposure.equalize_adapthist(Ojo_invesa)

plt.figure(1)
plt.subplot(221).imshow(Ojo)
plt.axis("off")
plt.subplot(222).imshow(Ojo_green, cmap='gray')
plt.axis("off")
plt.subplot(223).imshow(Ojo_invesa, cmap='gray')
plt.axis("off")
plt.subplot(224).imshow(Ojo_EHA, cmap='gray')
plt.axis("off")
plt.show()
# ----------------------------------------------------------------------------

Morfologia = morphology.grey.opening(Ojo_EHA, morphology.disk(8))
Resta = Ojo_EHA - Morfologia
Filtro = filters.median(Resta, morphology.disk(5))
Otsu = filters.threshold_otsu(Filtro)
Final = Filtro > Otsu

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    np.uint8(Final), None, None, None, 8, cv2.CV_32S)
areas = stats[1:, cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 200:  # keep
        result[labels == i + 1] = 255

out = morphology.remove_small_objects(
    result.astype(bool), min_size=750, connectivity=2)

plt.figure(2)
plt.subplot(221).imshow(Morfologia, cmap='gray')
plt.axis("off")
plt.subplot(222).imshow(Resta, cmap='gray')
plt.axis("off")
plt.subplot(223).imshow(Filtro, cmap='gray')
plt.axis("off")
plt.subplot(224).imshow(Final, cmap='gray')
plt.axis("off")
plt.show()

plt.figure(3)
plt.imshow(out, cmap='gray')
out = out.astype('uint8')
cv2.imwrite('Ojo.jpg', out * 255)

# --------------------------- Ecuacion matematica de segmentacion de ojo ----------------
