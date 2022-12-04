import numpy as np
import matplotlib.pyplot as plt
import cv2

MAX_INTENSITY = 256
GRAY_PLOT_KWARGS = {'cmap': 'gray', 'vmin': 0, 'vmax': 255}
GRAY_SHADES = np.arange(MAX_INTENSITY)

"""
Part 1. Point operations and histogram
"""


def gamma_correction(c=1):
    """
    a)
    gama < 1  above line gama = 1,
    gama > 1 under line gamma = 1,
    for gama = 1, identity function.
    c is the system gain. Here, c=1 and no gain in the output signal.
    """
    gamma = [0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]
    r = np.arange(MAX_INTENSITY, dtype=np.dtype('uint8'))
    r_normed = r / MAX_INTENSITY
    for g in gamma:
        plt.plot(r_normed, MAX_INTENSITY * c * r_normed ** g)

    plt.title('Gamma correction')
    plt.xlabel('r')
    plt.ylabel('s(r)')

    plt.legend([f'γ = {g}' for g in gamma])
    plt.show()


def power_law_transform(path=r'../mri_spine.jpg'):
    """
    b)
    After applying gamma transform with gamma=0.7, dark areas became clearer.
    The histogram stretched in the low intensities area.
    c = 1 not to cause noise in the bright areas of the bone.
    Histogram plots are done in logarithmic scale in y-axis, due to extremely high frequency of black.
    """

    spine_f = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    hist_f = cv2.calcHist([spine_f], [0], None, [256], [0, 256]).ravel()

    gamma = 0.7
    c = 1
    r = np.arange(256, dtype='uint8')

    lut = (((r / MAX_INTENSITY) ** gamma) * MAX_INTENSITY * c).astype('uint8')

    spine_g = cv2.LUT(spine_f, lut)
    hist_g = cv2.calcHist([spine_g], [0], None, [256], [0, 256]).ravel()

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].set_title('Original image')
    ax[0, 0].imshow(spine_f, **GRAY_PLOT_KWARGS)

    ax[0, 1].set_title('Histogram of original image')
    ax[0, 1].bar(GRAY_SHADES, hist_f, width=0.3)

    ax[1, 0].set_title('Transformed image')
    ax[1, 0].imshow(spine_g, **GRAY_PLOT_KWARGS)

    ax[1, 1].set_title('Histogram of trasformed image')
    ax[1, 1].bar(GRAY_SHADES, hist_g, width=0.3)

    ax[0, 1].set_yscale('log')
    ax[1, 1].set_yscale('log')

    plt.show()


"""
T=3;
    Maria_hist=imhist(Maria);
    norm_Maria_hist=(Maria_hist(T:size(Maria_hist)).')./sum(Maria_hist(T:size(Maria_hist)));
    HE_lut_temp=cumsum(norm_Maria_hist+T);
    HE_lut=[1:T-1 HE_lut_temp];
    HE_Maria=uint8(round(HE_lut(double(Maria)+1)));
    """


def histogram_eq(path=r'../mri_spine.jpg', T=0):
    """Relusting image probabilty distribution is not uniformal. This is because in discrete image,
    in this case 8-bit,  not all possible intensities exist in the image, and we have to round
    the values while mapping to CDF, which close to integer discrete values we can express with 8 bits.
"""

    spine_f = cv2.imread(path)

    hist_f = cv2.calcHist([spine_f], [0], None, [256], [0, 256]).ravel()[T:]
    num_pixels = np.sum(hist_f[T:])
    hist_f_normed = hist_f / num_pixels

    lut = np.floor(255 * np.cumsum(hist_f_normed + T)).astype('uint8')  # CDF
    T_pad = np.arange(T)
    lut = np.concatenate((lut, T_pad)).astype('uint8')

    # mapping an image to probability density function, effectively - histogram equalization
    spine_g = cv2.LUT(spine_f, lut)

    hist_g = cv2.calcHist([spine_g], [0], None, [256], [0, 256]).ravel()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].set_title('Equalized image')
    ax[0].imshow(spine_g, **GRAY_PLOT_KWARGS)

    ax[1].set_title('Equalized histogram')
    ax[1].bar(GRAY_SHADES, hist_g, width=0.3)
    ax[1].set_yscale('log')

    ax[2].set_title('Equalization LUT')
    ax[2].plot(lut / 255)

    plt.show()


def histogram_eq_threshold(path=r'../mri_spine.jpg', T=250):
    """
    Applying high threshold due to overall high darkness of the image.
    Achieving linear CDF and more uniformal PDF.
    """
    histogram_eq_threshold(path, T)


def main():
    # gamma_correction()
    # power_law_transform()
    # histogram_eq(T=0)
    histogram_eq(T=250)


if __name__ == '__main__':
    main()
