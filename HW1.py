import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, img_as_ubyte
from skimage import exposure, filters, color
import cv2 as cv


def powerLaw(image, gama, c):
    image_pw = exposure.adjust_gamma(image, gama, c)
    return image_pw
    #


def histEQ(image, thresh=0):
    dr_image = img_as_ubyte(color.rgb2gray(image))
    freq, bins = exposure.cumulative_distribution(dr_image)
    target_bins = np.arange(thresh * 255, 255)
    target_freq = np.linspace(thresh, 1, len(target_bins))
    # here we try to fit CDF of image to desired CDF of uniform distribution
    # this is the process of hist_equalization
    my_lut = np.interp(freq, target_freq, target_bins)
    cr_image = cv.LUT(dr_image, my_lut)

    return cr_image


def kernelNineHist(image):
    h, w = image.shape

    thrd_w = w // 3
    thrd_h = h // 3
    # th
    # is will be the first column
    images = []
    part_11 = image[:thrd_h, :thrd_w]
    part_12 = image[:thrd_h, thrd_w:thrd_w * 2]
    part_13 = image[:thrd_h, 2 * thrd_w:]
    part_21 = image[thrd_h:2 * thrd_h, :thrd_w]
    part_22 = image[thrd_h:2 * thrd_h, thrd_w:thrd_w * 2]
    part_23 = image[thrd_h:2 * thrd_h, 2 * thrd_w:]
    part_31 = image[thrd_h * 2:, :thrd_w]
    part_32 = image[thrd_h * 2:, thrd_w:thrd_w * 2]
    part_33 = image[thrd_h * 2:, 2 * thrd_w:]
    images.append(part_11)
    images.append(part_12)
    images.append(part_13)
    images.append(part_21)
    images.append(part_22)
    images.append(part_23)
    images.append(part_31)
    images.append(part_32)
    images.append(part_33)

    return images


def print_hi():
    my_list = [0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]
    pow_law = np.arange(256, dtype=np.dtype('uint8'))
    r = pow_law / 255
    c = 1
    for g in my_list:
        plt.plot(pow_law, 255 * c * r ** g)
        # gama < 1  above line gama = 1, gama > 1 under
        # for gama = 1, identity function
        # c is the system gain
    plt.show()
    image = plt.imread("mri_spine.jpg")
    plt.show()
    plt.hist((image.flatten()), bins=256, range=(0, 255))
    plt.show()
    gama = 0.3
    c = 1
    f, axs = plt.subplots(2, 3, sharey=True)
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('OG_spine')

    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0: continue
            gama = round(np.random.uniform(0, 1), 2)  # because image is originally dark, we need gama < 1
            lut = [np.power(x / 255.0, gama) * 255.0 * c for x in range(256)]
            lut = np.round(np.array(lut)).astype(np.uint8)
            new_image = cv.LUT(image, lut)
            axs[i, j].imshow(new_image, cmap='gray')
            axs[i, j].set_title(' gama=' + str(gama))
    plt.show()
    print('c is chosen=1 for all , because original image grayscale max value was already=255\
           my favorite gama was 0.46')
    gama = 0.46
    lut = [np.power(x / 255.0, gama) * 255.0 * c for x in range(256)]
    lut = np.round(np.array(lut)).astype(np.uint8)
    new_image = cv.LUT(image, lut)
    plt.imshow(new_image, cmap='gray')
    plt.show()
    eq_image = histEQ(image)
    plt.imshow(eq_image, cmap='gray')
    plt.show()
    # this is the eq_spine, result is better than gama, white parts no longer saturated
    eq_image = histEQ(image, 0.6)
    plt.imshow(eq_image, cmap='gray')
    plt.show()
    # this is the eq_spine with threshold=0.6, black parts are not touched in this way

    imgs = kernelNineHist(image)
    f, axs = plt.subplots(3, 3, sharey=True)
    axs = axs.ravel()
    for k, ax in enumerate(axs):
        axs[k].imshow(imgs[k], cmap='gray')
    plt.show()

    t, hist_axs = plt.subplots(3, 3, sharey=True)
    hist_axs = hist_axs.ravel()
    for i, hist_x in enumerate(hist_axs):
        hist_x[i].hist(imgs[i].flatten())
    plt.show()
    return


if __name__ == '__main__':
    print_hi()
