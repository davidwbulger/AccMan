# Change the WTFi image's background to transparent.

import cv2
import numpy as np

im = cv2.imread('WTFi_opaque.png',0)  #  0 for greyscale

# The method is as follows. We're going to partition this image conceptually
# into an inside part and an outside part, using a boundary that is a little
# bit inward of the logo's outer edge. Inside, the colours are unchanged, and
# alpha=1. Outside, alpha is set from the colour, and then the colour is set to
# black.

d=12  #  kernel diameter, in pixels
(x,y) = np.meshgrid(*(np.arange(d)-(d-1)/2,)*2)
kernel = (x*x+y*y<d*d/4).astype(np.uint8)
imdil = cv2.dilate(im,kernel)
(contours, _) = cv2.findContours(255-imdil,
  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #  external only, no smoothing

# 'mask' distinguishes the inside from the outside:
mask = np.zeros_like(im)  #  new image, same size
cv2.fillPoly(mask, pts=contours, color=255)

# I'm not immediately seeing whether PNG can handle 2-channel images. Making
# this RGBA only doubles its size, & will save some research.
outA = mask + (mask==0)*(255-im)
outRGB = im*(mask>0)

pngArray = np.stack([outRGB]*3+[outA],axis=-1)
cv2.imwrite('WTFi.png', pngArray)
