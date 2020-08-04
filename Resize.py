#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html#brief
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#sift-intro
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb

import cv2
import numpy as np

im_gray = cv2.imread('Sample7.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 50
#thresh = 127
bestthresh = 0
tempthresh = 0
threshstart = 100
threshbreak = 25
for i in range(threshstart,250,5):
    bestthresh = i
    im_bw = cv2.threshold(im_gray, i, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(str(len(contours)) + " --> " + str(i))
    #print((len(contours) - tempthresh)/len(contours))
    if (len(contours) - tempthresh) > threshbreak and i != threshstart:
        bestthresh = bestthresh -5
        break
    tempthresh =  len(contours)

im_bw = cv2.threshold(im_gray, bestthresh, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(im_bw.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

tempx = 0
tempy = 0
tempw = 0
temph = 0
tempwh =0 

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    print(x,y,w,h, w*h)
    if x > 20 and y > 20 and w >50 and h> 50:
        if (w*h) > tempwh:
            tempx = x
            tempy = y
            tempw = w
            temph = h
            tempwh = w*h

print("final " + str(tempw * temph))
print(im_bw)

tempx2 = tempx
tempy2 = tempy
tempw2 = tempw
temph2 = temph


rowcount, colcount = im_bw.shape

tempcal =0
tempcal2 =0
for k in range(tempx + tempw,colcount):
    tempcal = np.count_nonzero(im_bw[tempy:tempy + temph,tempx + tempw:k] != 255)
    #if (tempcal2 - tempcal) 
    print(np.count_nonzero(im_bw[tempy:tempy + temph,tempx + tempw:k] != 255))

print(rowcount)
print(im_bw[tempy:tempy + temph,tempx + tempw:tempx + tempw + 10])
im_bw[tempy:tempy + temph,tempx + tempw: colcount] = 1

#rowstart
#im_bw[tempy:tempy+1,:] = 1
#row end
#im_bw[tempy + temph:tempy + temph+1,:] = 1
#colstart
#im_bw[:,tempx:tempx+1] = 1
#col end
#im_bw[:,tempx + tempw:tempx + tempw+1] = 1

#left
#im_bw[tempy:tempy + temph,tempx + tempw:] = 1

#right
#im_bw[tempy:tempy + temph,:tempx] = 1

#top
#im_bw[:tempy,tempx:tempx + tempw+1] = 1

#bottom
#im_bw[tempy + temph:,tempx:tempx + tempw+1] = 1


roi = im_bw[tempy:tempy+temph, tempx:tempx+tempw]
cv2.imwrite('Test1.png', roi)
#break
cv2.imwrite('Test.png', im_bw)