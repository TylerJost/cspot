# %%
opencv = 0
try:
    import cv2 as cv
    opencv = 1
    print('OpenCV installed')
except:
    print('OpenCV not installed')

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread
from skimage import io, filters, img_as_ubyte
from skimage.color import rgb2gray
from skimage.util import img_as_float
# %%
def filterOpenCV(path):
    image = cv.imread(str(path.resolve()), cv.IMREAD_GRAYSCALE)
    print(image.shape)
    blur = cv.GaussianBlur(image, ksize=(3,3), sigmaX=1, sigmaY=1)
    ret3,th3 = cv.threshold(blur,0,1,cv.THRESH_OTSU)
    print(ret3)
    return th3

def filterSkimage(path):
    image = imread(path.resolve())

    if image.ndim == 3:
        image = rgb2gray(image)
    blur = filters.gaussian(image, sigma=1)
    thresholdValue = filters.threshold_otsu(blur)
    binaryImage = (blur > thresholdValue).astype(np.int)
    return binaryImage
# %%
path = Path('./data/50_img.tif')
if opencv:
    th3 = filterOpenCV(path)
    cv.imwrite('./data/img_opencv.png', th3)

binaryImage = filterSkimage(path)
imsave('./data/img_skimage.png', img_as_ubyte(binaryImage))
# %%
# %%
plt.subplot(121)
imgCV = imread('./data/img_opencv.png').astype(np.float32)
plt.imshow(imgCV)
plt.title('opencv')
plt.subplot(122)
imgSkimage = imread('./data/img_skimage.png').astype(np.float32)
plt.imshow(imgSkimage)
plt.title('skimage')

# plt.savefig('./data/imgComp.png', dpi = 500, bbox_inches='tight')
# %%
diff = np.linalg.norm(imgCV - imgSkimage)
