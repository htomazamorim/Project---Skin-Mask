from neuralNetwork import *
from imgLib import meanKernel, gaussianKernel
from skimage import io
from scipy import ndimage, stats
from skimage.transform import resize
import matplotlib.pyplot as plt

NN = NeuralNetwork([3, 5, 15, 1])
NN.loadNN()

Xb = np.load('./X_testing.npy', allow_pickle=True)
yb = np.load('./y_testing.npy', allow_pickle=True)

print("Validation on training set")
validationData(NN, Xb, yb)

qtdImg = 1
i = 1

f= plt.figure()

while(1):
    plt.close(f)
    i = input('Digite a imagem:')

    if(i == '0'):
        break

    im = io.imread('imgs/'+i+'.jpg')
    im2 = np.zeros(im.shape)

    # Passing through Mean filter
    for k in range(3):
        im2[:,:,k] = ndimage.convolve(im[:,:,k], meanKernel(4))

    mask2 = np.zeros((im.shape[0], im.shape[1]))

    # Calculating mask
    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
            X = np.array([im2[x,y,2], im2[x,y,1], im2[x,y,0]])/255

            if(NN.forwardProp(X.reshape(-1,1)) > 0.8):
                mask2[x,y] = 255


    # Blurring mask
    k = gaussianKernel(5, 2)

    mask2 = ndimage.convolve(mask2, k)

    # Plotting figure
    f = plt.figure()

    plt.subplot(qtdImg,3,1)
    plt.title('Original Image')
    plt.imshow(im)
    plt.axis('off')

    plt.subplot(qtdImg,3,2)
    plt.title('Mean-filtered image')
    plt.imshow(im2/255)
    plt.axis('off')

    plt.subplot(qtdImg,3,3)
    plt.title('Mask')
    plt.imshow(mask2, cmap='gray')
    plt.axis('off')

    plt.show()