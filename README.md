Edge_Detection
==============
This module lets you filter images using edge detection as well as some other cool basic kernels. Eventually we hope to add in some more creative filters.
## Dependencies:
    - PIL 

## Usage:
```sh
$ python kernel.py --help
usage: kernel.py [-h] image kernel

This module filters images according to the options below

positional arguments:
  image       Choose an image file for processing. Currently tested filetypes include: .bmp, .png, .jpg
  kernel      Choose a kernel for filtering. Options are:
                          identity - 3x3 kernel, does not change image
                          edgeA - 3x3 kernel, edge detection option A
                          edgeB - 3x3 kernel, edge detection option B
                          edgeC - 3x3 kernel, edge detection option C
                          sobel - pair of 3x3 kernels, sobel edge detection
                          sharpen - 3x3 kernel, sharpens image
                          boxBlur - 3x3 kernel, blur option A
                          gaussBlur - 3x3 kernel, blur option B

optional arguments:
  -h, --help  show this help message and exit

$ python kernel "lena.jpg" "sobel"
```
