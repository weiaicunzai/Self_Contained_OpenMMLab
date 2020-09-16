import cv2
import math
from feature import Feature




# default number of sampled intervals per octave
SIFT_INTVLS = 3

# double image size before pyramid construction?
SIFT_IMG_DBL = True

# assumed gaussian blur for input image
SIFT_INIT_SIGMA = 0.5

# width of border in which to ignore keypoints
SIFT_IMG_BORDER = 5

# maximum steps of keypoint interpolation before failure
SIFT_MAX_INTERP_STEPS = 5

img = cv2.imread('/home/baiyu/Downloads/CV_Algorithm/Lenna_(test_image).png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sift = cv2.SIFT_create()
#kp = sift.detect(gray, None)
#print(dir(kp[0]))
#print(kp[0].response)
#print(kp[0].size)
#kp,des = sift.compute(gray,kp)
#print(des.shape)
#img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('test', img)
#cv2.waitKey(0)

#def sift_features(img):
#    return _sift_features(img, )

def _sift_features(img, intvls, sigma, contr_thr, curv_thr, img_dbl, descr_width, descr_hist_bins):
    i = 0
    n = 0

    img = img.astype('float32')
    init_img = create_init_img(img, img_dbl, sigma)
    octvs = int(math.log(min(img.shape[0], img.shape[1])) / math.log(2) - 2)
    gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma)
    dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls)

def create_init_img(img, img_dbl, sigma):
    """
    Converts an image to 8-bits gray-scale and Gaussian-smooths it. The image
    is optionally doubled prior to smoothing

    Args:
        img: input image
        img_dbl: if true, the image is doubled
        sigma: total std of Gaussian smoothing
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Of course, if we pre-smooth the image before extrema
    # detection, we are effectively discarding the highest
    # spatial frequencies. Therefore, to make full use of
    # the input, the image can be expanded to create more
    # sample points than were present in the original. We
    # double the size of the input image using linear
    # interpolation prior to building the first level of
    # the pyramid.

    if img_dbl:

        sig_diff = math.sqrt(sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4)
        # The image doubling increases the number of stable
        # keypoints by almost a factor of 4, but no significant
        # further improvements were found with a larger
        # expansion factor
        dbl = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        dbl = cv2.GaussianBlur(dbl, (0, 0), sigmaX=sig_diff, sigmaY=sig_diff)

        return dbl

    else:
        sig_diff = math.sqrt(sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA)
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sig_diff, sigmaY=sig_diff)

        return gray


def build_gauss_pyr(base, octvs, intvls, sigma):
    """Builds Gaussian scale space pyramid from an image

    Args:
        base: base image of pyramid
        octvs: number of octaves of scale space
        intvls: number of intervals per octave
        sigma: amount of Gaussian smoothing per octave

    Returns:
        returns a Gaussian scale space pymramid as an octvs x (intvls + 3)
        array
    """
    gauss_pyr = [[None for _ in range(intvls)] for _ in range(octvs)]

    # We choose to divide each octave of scale space (i.e., doubling of σ )
    # into an integer number, s, of intervals, so k = 2 1/s
    k = 2 ** (1 / intvls)

    # The initial image is incrementally convolved with Gaussian to produce
    # images seperated by a constand factor k in scale space
    sig = [sigma, sigma * math.sqrt(k*k - 1)]

    # the last two image of a octave have a larger scale than 2*sigma
    for i in range(2, intvls + 3):
        sig[i] = sig[i - 1] * k

    for o in range(octvs):
        for i in range(intvls + 3):
            if o == 0 and i == 0:
                gauss_pyr[o][i] = base

            # scales in the first octave:
            # [sigma, 2^(1/3) * sigma, 2^(2/3) * sigma, 2^(3/3) * sigma,
            #  2^(4/3) * sigma, 2^(5/3) * sigma]
            # the intvls-th scale is 2sigma scale
            elif i == 0:
                gauss_pyr[o][i] = cv2.resize(
                                gauss_pyr[o][intvls],
                                (0, 0),
                                fx=0.5,
                                fy=0.5,
                                interpolation=cv2.INTER_NEAREST)

            else:
                gauss_pyr[o][i] = cv2.GaussianBlur(
                    gauss_pyr[o][i-1],
                    sigmaX=sigma[i],
                    sigmaY=sigma[i]
                )

    return gauss_pyr


def build_dog_pyr(gauss_pyr, octvs, intvls):

    dog_pyr = [[None for _ in range(intvls + 2)] for _ in range(octvs)]
    for o in range(octvs):
        for i in range(intvls + 2):
            dog_pyr[o][i] = cv2.subtract(gauss_pyr[o][i+1], gauss_pyr[o][i])

    return dog_pyr

def scale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr):
    features = []

    for o in range(octvs):
        for i in range(1, intvls + 1):
            for r in range(SIFT_IMG_BORDER):
                pass

#def hello():
#    a = hello1(
#    print(a)
#
#def hello1():
#    return 4


#print(hello())


def test():
    a = [-1]

    a = [a[-1] + i for i in range(10)]

    print(a)
    return a

test()

print(math.log(min(512, 512)) / math.log(2) - 2)

#dbl = cv2.resize(img, (0, 0), fx=2, fy=2)
cv2.imshow('1test', img)
dbl = img
dbl = cv2.GaussianBlur(dbl, (0, 0), sigmaX=0.5, sigmaY=0.5)
print(dbl.shape)

import dis

def test():

    #dbl = cv2.GaussianBlur(dbl, (0, 0), sigmaX=0.5, sigmaY=0.5)
    print('test')
    min(51, 33)
dis.dis(test)

#cv2.imshow('test', dbl)
#cv2.waitKey(0)


import numpy as np

n1 = np.zeros((10, 10))
n2 = np.zeros((10, 10))

n1.fill(100)
n2.fill(200)

print(n1 - n2)
print(cv2.subtract(n1, n2))

img = np.zeros((5, 10))
print(img[4, 9])
print(img.shape)
