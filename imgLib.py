import numpy as np

def linearTrans(x,a,b):
    return a*x+b

def reduceContrast_centered(im, k, norm=False):
    threshhold = int((k/100)*255)

    k1 = threshhold

    # Linear transfer function calculated
    if(norm):
        k2 = 1 - threshhold
        a = (k2 - k1)
    else:
        k2 = 255-threshhold
        a = (k2 - k1) / (255)

    b = k1

    # Using general linear function
    response = linearTrans(im, a, b)

    return response

# Kernels

def meanKernel(hs):
    size = 2 * hs + 1

    kernel = np.ones((size, size))

    return kernel / size**2


def gaussianKernel(hs, sig):
    size = 2 * hs + 1

    kernel = np.ones((size, size))

    for i in range(size):
        for j in range(size):
            u = i - hs
            v = j - hs
            kernel[i][j] = np.exp((-u ** 2 - v ** 2) / (2 * sig ** 2)) / (2 * np.pi * sig ** 2)

    return kernel / np.sum(kernel)