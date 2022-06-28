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


def cdf(hist):
    cdf = np.zeros((256))
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf = [ele * 255 / cdf[-1] for ele in cdf]
    return cdf


def equalize_image_Red(image):
    equa = cdf(dfR(image))
    image_equalized = np.zeros_like(image)
    # image_equalized = np.interp(x=image, xp=range(0,256), fp=equa)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][0]
            image_equalized[x][y][0] = equa[pixel_val]

    return image_equalized

def equalize_image_Green(image):
    equa = cdf(dfG(image))
    image_equalized = np.zeros_like(image)
    # image_equalized = np.interp(x=image, xp=range(0,256), fp=equa)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][1]
            image_equalized[x][y][1] = equa[pixel_val]

    return image_equalized

def equalize_image_Blue(image):
    equa = cdf(dfR(image))
    image_equalized = np.zeros_like(image)
    # image_equalized = np.interp(x=image, xp=range(0,256), fp=equa)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][2]
            image_equalized[x][y][2] = equa[pixel_val]

    return image_equalized



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

#let's equalize reference:
equalizeRefRed =equalize_image_Red(reference)
equalizeRefGreen =equalize_image_Green(reference)
equalizeRefBlue =equalize_image_Blue(reference)
#let's equalize reference's histogram:
eqRefRedHist = cdf(dfR(equalizeRefRed))
eqRefGreenHist = cdf(dfR(equalizeRefGreen))
eqRefBlueHist = cdf(dfR(equalizeRefBlue))
#let's get histogram of equalized ones!
histRefRed = dfR(equalizeRefRed)
histRefGreen = dfG(equalizeRefGreen)
histRefBlue = dfB(equalizeRefBlue)

#let's equalize source:
equalizeSourceRed =equalize_image_Red(source)
equalizeSourceGreen =equalize_image_Green(source)
equalizeSourceBlue =equalize_image_Blue(source)
#let's equalize Source's histogram:
eqSourceRedHist = cdf(dfR(equalizeSourceRed))
eqSourceGreenHist = cdf(dfR(equalizeSourceGreen))
eqSourceBlueHist = cdf(dfR(equalizeSourceBlue))
#let's get histogram of equalized ones!
histSourceRed = dfR(equalizeSourceRed)
histSourceGreen = dfG(equalizeSourceGreen)
histSourceBlue = dfB(equalizeSourceBlue)

plt.figure(8)
plt.title('Equalized Red of Reference image')
plt.imshow(equalizeRefRed)

plt.figure(9)
plt.title('Equalized Green of Reference image')
plt.imshow(equalizeRefGreen)

plt.figure(10)
plt.title('Equalized Blue of Reference image')
plt.imshow(equalizeRefBlue)

plt.figure(11)
plt.title('Red Histogram of equalized Refrence image')
plt.plot(eqRefRedHist)

plt.figure(12)
plt.title('Green Histogram of equalized Refrence image')
plt.plot(eqRefGreenHist)

plt.figure(13)
plt.title('Blue Histogram of equalized Refrence image')
plt.plot(eqRefBlueHist)

plt.figure(14)
plt.title('Equalized Red of Source image')
plt.imshow(equalizeSourceRed)

plt.figure(15)
plt.title('Equalized Green of Source image')
plt.imshow(equalizeSourceGreen)

plt.figure(16)
plt.title('Equalized Blue of Source image')
plt.imshow(equalizeSourceBlue)

plt.figure(17)
plt.title('Red Histogram of equalized Source image')
plt.plot(eqSourceRedHist)

plt.figure(18)
plt.title('Green Histogram of equalized Source image')
plt.plot(eqSourceGreenHist)

plt.figure(19)
plt.title('Blue Histogram of equalized Source image')
plt.plot(eqSourceBlueHist)
plt.show()
