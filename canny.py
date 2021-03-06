#!/usr/bin/env python
import numpy as np
from PIL import Image
from time import time

class Canny():
    def __init__(self, fileName, thresholds=(.1, .2)):
        '''
        fileName -      name of image file
        thresholds -    lower and upper thresholds contained in array or tuple. 
                        both threshold values must be between 0 and 1.
        
        __init__ is used to run all necessary methods for edge detection, 
        then saves the result to a new image file.
        '''
        self.kernel = np.array(
                [
                    [
                        [1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], 
                    [
                        [1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]
                    ])
        self.smooth = np.array(
                [
                    [1./16, 1./8, 1./16],
                    [1./8, 1./4, 1./8],
                    [1./16, 1./8, 1./16]
                    ])
        self.fileName = fileName

        self.startTime = time()

        '''get the image'''
        self.getImg()

        '''convolve image with sobel kernels and save as images'''
        self.convolve()
        yGradient = self.convolvedArrayY
        Image.fromarray(yGradient.astype('uint8'), "L").save("cannyImages/canny_yGradient_{0}".format(self.fileName))
        xGradient = self.convolvedArrayX
        Image.fromarray(xGradient.astype('uint8'), "L").save("cannyImages/canny_xGradient_{0}".format(self.fileName))
        gradient = self.gradientArray
        Image.fromarray(gradient.astype('uint8'), "L").save("cannyImages/canny_gradient_{0}".format(self.fileName))
        direction = self.directionArray
        Image.fromarray(direction.astype('uint8'), "L").save("cannyImages/canny_direction_{0}".format(self.fileName))

        '''round the direction of the gradient to 0, 45, 90, or 135 degrees and save as image'''
        self.getDirection()
        roundedDirection = self.directionArray
        Image.fromarray(roundedDirection.astype('uint8'), "L").save("cannyImages/canny_roundedDirection_{0}".format(self.fileName))

        '''supress false edges and save as image'''
        self.suppress()
        suppressedGradient = self.gradientArray
        Image.fromarray(suppressedGradient.astype('uint8'), "L").save("cannyImages/canny_suppressedGradient_{0}".format(self.fileName))

        '''separate weak and strong edges with threshold values and save as images'''
        self.threshold(thresholds)
        strongEdges = self.upperArray
        Image.fromarray(strongEdges.astype('uint8'), "L").save("cannyImages/canny_strongEdges_{0}".format(self.fileName))
        weakEdges = self.lowerArray
        Image.fromarray(weakEdges.astype('uint8'), "L").save("cannyImages/canny_weakEdges_{0}".format(self.fileName))

        '''add weak edges is they are linked to strong edges and save final edge detected image'''
        self.link()
        final = self.upperArray
        Image.fromarray(final.astype('uint8'), "L").save("cannyImages/canny_finalEdges_{0}".format(self.fileName))

        print "Total time: ", time() - self.startTime
        return

    def getImg(self):
        '''
        open image and convert to a numpy array
        '''
        self.img = Image.open(self.fileName)
        self.img = self.img.convert('L')
        self.imgArray = np.array(self.img, dtype = 'uint8')
        self.shape = self.imgArray.shape

        print "Got image. Execution Time: ", time() - self.startTime
        self.time = time()
        return

    def _convolve(self, kernel):
        '''
        helper function to convolve images with 3x3 kernel
        '''
        convolvedArray = np.zeros(self.shape)
        kernelIterator = [-1, 0, 1]
        for x in range(1, self.shape[0] - 1):
            for y in range(1, self.shape[1] -1):
                for i in kernelIterator:
                    for j in kernelIterator:
                        convolvedArray[x, y] +=  (self.imgArray[x + i, y + j]*
                                kernel[1 + i, 1 + j])

        print "Convolved. ET: ", time() - self.time
        self.time = time()
        return convolvedArray

    def convolve(self):
        '''
        -smooth the image by convolution with gaussian kernel
        -calculate the gradient of the image by convolution with sobel
        -calculate the direction of the gradient
        '''
        self.imgArray = self._convolve(self.smooth)

        self.convolvedArrayX = self._convolve(self.kernel[0])
        self.convolvedArrayY = self._convolve(self.kernel[1])
        self.gradientArray = np.sqrt(self.convolvedArrayX**2 + self.convolvedArrayY**2)
        self.directionArray = np.arctan2(self.convolvedArrayY, self.convolvedArrayX)*180/np.pi
        return

    def getDirection(self):
        '''
        round gradient direction to 0, 45, 90, or 135
        '''
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if (self.directionArray[x][y]<22.5 and self.directionArray[x][y]>=0) or \
                   (self.directionArray[x][y]>=157.5 and self.directionArray[x][y]<202.5) or \
                   (self.directionArray[x][y]>=337.5 and self.directionArray[x][y]<=360):
                    self.directionArray[x][y]=0
                elif (self.directionArray[x][y]>=22.5 and self.directionArray[x][y]<67.5) or \
                     (self.directionArray[x][y]>=202.5 and self.directionArray[x][y]<247.5):
                    self.directionArray[x][y]=45
                elif (self.directionArray[x][y]>=67.5 and self.directionArray[x][y]<112.5)or \
                     (self.directionArray[x][y]>=247.5 and self.directionArray[x][y]<292.5):
                    self.directionArray[x][y]=90
                else:
                    self.directionArray[x][y]=135

        print "Rounded directions. ET: ", time() - self.time
        self.time = time()
        return

    def suppress(self):
        '''
        supress noise via the the following algorithm:
            - for a pixel, look at it's neighbors in the direction indicated by the direction array
            - if the pixel's value is less than that of either of the neighboring pixels, delete it
        '''
        for x in range(1, self.shape[0] - 1):
            for y in range(1, self.shape[1] - 1):
                if self.directionArray[x][y] == 0:
                    neighbor1 = self.gradientArray[x][y + 1]
                    neighbor2 = self.gradientArray[x][y - 1]
                elif self.directionArray[x][y] == 45:
                    neighbor1 = self.gradientArray[x + 1][y - 1]
                    neighbor2 = self.gradientArray[x - 1][y + 1]
                elif self.directionArray[x][y] == 90:
                    neighbor1 = self.gradientArray[x + 1][y]
                    neighbor2 = self.gradientArray[x - 1][y]
                elif self.directionArray[x][y] == 135:
                    neighbor1 = self.gradientArray[x + 1][y + 1]
                    neighbor2 = self.gradientArray[x - 1][y - 1]

                if self.gradientArray[x][y] <= max(neighbor1, neighbor2):
                    self.gradientArray[x][y] = 0

        print "Supressed false edges. ET: ", time() - self.time
        self.time = time()
        return

    def threshold(self, thresholds):
        '''
        thresholds - tuple pair or list pair containing upper and lower threshold decimals

        the weak and strong edges are separated into two arrays. 
        strong edges are those with values greater than the upper threshold
        weak edges are those with values between the lower and upper threshold
        '''
        span = (np.amax(self.gradientArray) + np.amin(self.gradientArray))
        upper = np.amax(thresholds)*span
        lower = np.amin(thresholds)*span

        self.upperArray = np.zeros(self.shape)
        self.lowerArray = np.zeros(self.shape)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.gradientArray[x][y] >= upper:
                    self.upperArray[x][y] = 255
                if self.gradientArray[x][y] < upper:
                    if  self.gradientArray[x][y] >= lower:
                        self.lowerArray[x][y] = 255

        print "Separated weak/strong edges. ET: ", time() - self.time
        self.time = time()
        return

    def link(self):
        '''
        if an edge pixel in the weak edge array is adjacent to a pixel in the strong edge array,
        it is added to the strong edge array.
        '''
        def traverseEdges(x, y):
            '''
            - check neighbors of a weak pixel to see if it's connected. 
            - if it's connected, recurse on the newly found edge until it ends 
            '''
            adjacentInd = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for i, j in adjacentInd:
                if not self.upperArray[x + i][y + j]:
                    if self.lowerArray[x + i][y + j]:
                        self.upperArray[x + i][y + j] = 255
                        traverseEdges(x + i, y + j)
            return

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.upperArray[x][y]:
                    traverseEdges(x, y)

        print "Added linked weak edges. ET: ", time() - self.time
        self.time = time()
        return


def main():
    import argparse as ap
    parser = ap.ArgumentParser(
            description = "This module filters images according to the options below",
            formatter_class = ap.RawTextHelpFormatter
            )
    parser.add_argument("image",
            type = str,
            help = '''            Choose an image file for processing. 
            Currently tested filetypes include: .bmp, .png, .jpg'''
            )
    parser.add_argument("--thresholds", "-t",
            nargs = "+",
            type = float,
            help = '''            Choose a lower and upper threshold between 0 and 1. Ex:
            $ canny.py lena.jpg -t .1 .2'''
            )
    args = parser.parse_args()
    
    if args.thresholds:
        Canny(args.image, args.thresholds)
    else:
        Canny(args.image)

    return

if __name__ == '__main__':
    main()
