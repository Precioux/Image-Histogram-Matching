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
    hist=dfR(image)
    equa = cdf(hist)
    image_equalized = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][0]
            image_equalized[x][y][0] = equa[pixel_val]
    return image_equalized

def equalize_image_Green(image):
    equa = cdf(dfG(image))
    image_equalized = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][1]
            image_equalized[x][y][1] = equa[pixel_val]
    return image_equalized

def equalize_image_Blue(image):
    equa = cdf(dfB(image))
    image_equalized = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y][2]
            image_equalized[x][y][2] = equa[pixel_val]

    return image_equalized

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

plt.figure(0)
plt.title('Reference image')
plt.imshow(Reference)

plt.figure(1)
plt.title('Histogram of Red in Reference image')
plt.plot(dfR(reference),color="red")

plt.figure(2)
plt.title('Histogram of Green in Reference image')
plt.plot(dfG(reference),color="green")

plt.figure(1)
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
#
plt.show()

#let's equalize reference:
equalizeRefRed =equalize_image_Red(reference)
equalizeRefGreen =equalize_image_Green(reference)
equalizeRefBlue =equalize_image_Blue(reference)
new = np.zeros_like(reference)
for i in range(reference.shape[0]):
    for j in range(reference.shape[1]):
        r = equalizeRefRed[i][j][0]
        g = equalizeRefGreen[i][j][1]
        b = equalizeRefBlue[i][j][2]
        new[i][j][0] = r
        new[i][j][1] = g
        new[i][j][2] = b

plt.figure(8)
plt.title('Refrence Image Equalized')
plt.imshow(new)
plt.show()

#let's equalize reference's histogram:
eqRefRedHist = cdf(dfR(equalizeRefRed))
eqRefGreenHist = cdf(dfG(equalizeRefGreen))
eqRefBlueHist = cdf(dfB(equalizeRefBlue))


#let's get histogram of equalized ones!
histRefRed = dfR(equalizeRefRed)
histRefGreen = dfG(equalizeRefGreen)
histRefBlue = dfB(equalizeRefBlue)


#let's equalize source:
equalizeSourceRed =equalize_image_Red(source)
equalizeSourceGreen =equalize_image_Green(source)
equalizeSourceBlue =equalize_image_Blue(source)

new2 = np.zeros_like(source)
for i in range(source.shape[0]):
    for j in range(source.shape[1]):
        r = equalizeSourceRed[i][j][0]
        g = equalizeSourceGreen[i][j][1]
        b = equalizeSourceBlue[i][j][2]
        new2[i][j][0] = r
        new2[i][j][1] = g
        new2[i][j][2] = b

plt.figure(9)
plt.title('Source Image Equalized')
plt.imshow(new2)
plt.show()

#let's equalize Source's histogram:
eqSourceRedHist = cdf(dfR(equalizeSourceRed))
eqSourceGreenHist = cdf(dfG(equalizeSourceGreen))
eqSourceBlueHist = cdf(dfB(equalizeSourceBlue))

#let's get histogram of equalized ones!
histSourceRed = dfR(equalizeSourceRed)
histSourceGreen = dfG(equalizeSourceGreen)
histSourceBlue = dfB(equalizeSourceBlue)


plt.figure(10)
plt.title('Red Histogram of equalized Refrence image')
plt.plot(histRefRed)

plt.figure(11)
plt.title('Green Histogram of equalized Refrence image')
plt.plot(histRefGreen)

plt.figure(12)
plt.title('Blue Histogram of equalized Refrence image')
plt.plot(histRefBlue)

plt.figure(13)
plt.title('Red Histogram of equalized Source image')
plt.plot(histSourceRed)

plt.figure(14)
plt.title('Green Histogram of equalized Source image')
plt.plot(histSourceGreen)

plt.figure(15)
plt.title('Blue Histogram of equalized Source image')
plt.plot(histSourceBlue)
plt.show()

# #mapping histograms...
mappedHistRed = np.zeros_like(histSourceRed)
mappedHistBlue = np.zeros_like(histSourceGreen)
mappedHistGreen = np.zeros_like(histSourceBlue)
matched_image = np.zeros_like(source)

for i in range(256):
    if (eqSourceRedHist[i] != 0):
        idx = find_nearest(eqRefRedHist, eqSourceRedHist[i])
        mappedHistRed[i] = histRefRed[idx]

    if (eqSourceGreenHist[i] != 0):
        idx = find_nearest(eqRefGreenHist, eqSourceGreenHist[i])
        mappedHistGreen[i] = histRefGreen[idx]

    if (eqSourceBlueHist[i] != 0):
        idx = find_nearest(eqRefBlueHist, eqSourceBlueHist[i])
        mappedHistBlue[i] = histRefBlue[idx]

match_equa_Red = cdf(mappedHistRed)
match_equa_Green = cdf(mappedHistGreen)
match_equa_Blue = cdf(mappedHistBlue)

for x in range(matched_image.shape[0]):
    for y in range(matched_image.shape[1]):
        pixel_val_red =  source[x][y][0]
        pixel_val_green = source[x][y][1]
        pixel_val_blue = source[x][y][2]
        matched_image[x][y][0] = match_equa_Red[pixel_val_red]
        matched_image[x][y][1] = match_equa_Green[pixel_val_green]
        matched_image[x][y][2] = match_equa_Blue[pixel_val_blue]

plt.figure(16)
plt.title('Matched new image Red Histogram')
plt.plot(dfR(matched_image))
plt.figure(17)
plt.title('Matched new image Green Histogram')
plt.plot(dfG(matched_image))
plt.figure(18)
plt.title('Matched new image Blue Histogram')
plt.plot(dfB(matched_image))
plt.figure(19)
plt.title('Matched new image')
plt.imshow(matched_image)
plt.show()


