#!/usr/bin/env python
"""
This module transforms a given image by a specified kernel or set of kernels.
The module is self-contained and is controlled via a command-line interface.
The goal is to apply any 3x3 kernel (including pairs of 3x3 kernels such as the sobel operator)
as well as to apply artsy algorithms to an edge-detected image.

Some ideas for "artsy filters":
    - connect edges "correctly" and then change their shape (ex. make them zig-zag)
    - turn edge pixels into a new shape (ex. pixel --> circle of pixels)
    - connect edges to make enclosed spaces and color each space to create a new color image
    - connect edges incorrectly to create impossible geometry (!!!!)
        - maybe by first connecting them "correctly" and then swapping things out
    - apply randomly generated rules for connecting edges
    - apply specified rules for connecting/deleting edges 
        - ex. only connect edges that are in the same horizontal stripe of specified width
    - connect the edges "correctly" and then color the edges based on location

"""

import numpy as np
from PIL import Image

class EdgeDetection():
    def __init__(self, fileName, kernel):
        self.fileName = fileName
        self.kernel = kernel
        self.getImg()
        array = self.convolve()
        array = self.binarize(array)

        newImg = Image.fromarray(array.astype('uint8'), "L")
        newImg.save('edge_detected_{0}'.format(self.fileName))

    def getImg(self):
        self.img = Image.open(self.fileName)
        self.img = self.img.convert('L')
        self.imgArray = np.array(self.img, dtype = 'uint8')
        self.shape = self.imgArray.shape

    def _convolve(self, kernel):
        """helper function to convolve images with 1d kernel"""
        convolvedArray = np.zeros(self.shape)
        kernalIterator = [-1, 0, 1]
        for x in range(1, self.shape[0] - 1):
            for y in range(1, self.shape[1] -1):
                for i in kernalIterator:
                    for j in kernalIterator:
                        convolvedArray[x, y] +=  (self.imgArray[x + i, y + j]*
                                kernel[1 + i, 1 + j])
        return convolvedArray

    def convolve(self):
        """colvolve the image matrix with the kernel"""

        if len(self.kernel.shape) == 2 or self.kernel.shape[0] == 1: 
            convolvedArray = self._convolve(self.kernel)
        elif self.kernel.shape[0] == 2:
            kernelX = self.kernel[0]
            kernelY = self.kernel[1]
            convolvedArrayX = self._convolve(kernelX)
            convolvedArrayY = self._convolve(kernelY)
            convolvedArray = np.sqrt(convolvedArrayX**2 + convolvedArrayY**2)
        
        return convolvedArray    

    def binarize(self, array):
        """convert array into black and white image"""
        avg = (np.amax(array) + np.amin(array))/2
        for i in xrange(array.shape[0]):
            for j in xrange(array.shape[1]): 
                if array[i][j] < avg:
                    array[i][j] = 0
                elif array[i][j] >= avg:
                    array[i][j] = 255
        return array


def main():
    import argparse as ap
    parser = ap.ArgumentParser(
            description = "This module filters images according to the options below",
            formatter_class = ap.RawTextHelpFormatter
            )
    parser.add_argument("image",
            type = str,
            help = "Choose an image file for processing. Currently tested filetypes include: .bmp, .png, .jpg")
    parser.add_argument("kernel",
            type = str,
            help = '''
            Choose a kernel for filtering. Options are:
            identity - 3x3 kernel, does not change image
            edgeA - 3x3 kernel, edge detection option A
            edgeB - 3x3 kernel, edge detection option B
            edgeC - 3x3 kernel, edge detection option C
            sobel - pair of 3x3 kernels, sobel edge detection
            sharpen - 3x3 kernel, sharpens image
            boxBlur - 3x3 kernel, blur option A
            gaussBlur - 3x3 kernel, blur option B
            '''
            )
    args = parser.parse_args()

    identity = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edgeDetectA = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edgeDetectB = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    edgeDetectC = np.array([[1, 0, -1],[0, 0, 0],[-1, 0, 1]])
    sobel = np.array([[[1, 0, -1],[2, 0, -2],[1, 0, -1]], [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]])
    sharpen = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    boxBlur = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
    gaussBlur = np.array([[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]])

    if args.kernel == "identity":
        kernel = identity
    elif args.kernel == "edgeA":
        kernel = edgeDetectA
    elif args.kernel == "edgeB":
        kernel = edgeDetectB
    elif args.kernel == "edgeC":
        kernel = edgeDetectC
    elif args.kernel == "sobel":
        kernel = sobel
    elif args.kernel == "sharpen":
        kernel = sharpen
    elif args.kernel == "boxBlur":
        kernel = boxBlur
    elif args.kernel == "gaussBlur":
        kernel = gaussBlur
    else:
        print "please choose an available kernel for filtering"
        return

    try:
        print "filtering in process"
        EdgeDetection(args.image, kernel)
    except IOError:
        print "please make sure the image file you are trying to filter exists"

    return

if __name__ == '__main__':
    main()
    #fileName = 'lena.jpg'
    # kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    #kernel = np.array([[[1, 0, -1],[2, 0, -2],[1, 0, -1]], [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]])
    #EdgeDetection(fileName, kernel)
