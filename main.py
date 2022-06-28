# Samin Mahdipour - 9839039
#Final Project - part 2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray


# load the image
addressReference = "Reference.jpg"
addressSource = "Source.jpg"

Reference = Image.open(addressReference)
Source = Image.open(addressSource)
# convert image to numpy array
reference = asarray(Reference)
source = asarray(Source)

sizeReference = reference.size
widthReference = Reference.width
heightRefrence = Reference.height

sizeSource = source.size
widthSource = Source.width
heightSource = Source.height

def dfB(img, deb=""):
    values = np.zeros((256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[round(img[i][j][2])] += 1

    return values


def dfG(img, deb=""):
    values = np.zeros((256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[round(img[i][j][1])] += 1

    return values


def dfR(img, deb=""):
    values = np.zeros((256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[round(img[i][j][0])] += 1

    return values

plt.figure(0)
plt.title('Reference image')
plt.imshow(Reference)

plt.figure(1)
plt.title('Histogram of Red in Reference image')
plt.plot(dfR(reference),color="red")

plt.figure(2)
plt.title('Histogram of Green in Reference image')
plt.plot(dfG(reference),color="green")

plt.figure(3)
plt.title('Histogram of Blue in Reference image')
plt.plot(dfB(reference),color="blue")

plt.figure(4)
plt.title('Source image')
plt.imshow(Source)

plt.figure(5)
plt.title('Histogram of Red in Source image')
plt.plot(dfR(source),color="red")

plt.figure(6)
plt.title('Histogram of Green in Source image')
plt.plot(dfG(source),color="green")

plt.figure(7)
plt.title('Histogram of Blue in Source image')
plt.plot(dfB(source),color="blue")

plt.show()