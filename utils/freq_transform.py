'''
Perform DCT or FFT transform to get frequency specturm.
'''

import os
import cv2
import numpy as np
from PIL import Image

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage

import torch
import torchvision.transforms as transforms
import torch_dct

import ipdb

def log_scale(array, scale_factor=1):
    """Log scale the input array.
    """
    return scale_factor * np.log10(abs(array))

def dct2d(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array

def fft2d(array):
    array = fftpack.fft2(array)
    array = fftpack.fftshift(array)
    return array

def normalize(image, mean, std):
    image = (image - mean) / std
    return image

def scale_image(image):
    if not image.flags.writeable:
        image = np.copy(image)

    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    image /= 127.5
    image -= 1.
    return image

def load_image(path, img_size, grayscale=False):
    x = Image.open(path)

    x = x.resize((img_size, img_size))

    if grayscale:
        x = x.convert("L")
    return np.asarray(x)

def highpass_median_filter(image, filter_size=3):
    '''
    a simple form of high-pass filtering,
    subtracting the image from its median blurred.
    '''
    image_blur = ndimage.median_filter(image, size=filter_size)
    image_highpass = abs(image - image_blur)

    return image_highpass

def dct2d_wrapper(image, scale_factor=1, highpass=False):
    image = np.asarray(image)
    if highpass:
        image = highpass_median_filter(image)
    image = dct2d(image)
    image = log_scale(image, scale_factor)
    return image

def fft2d_wrapper(image, scale_factor=1, highpass=False):
    image = np.asarray(image)
    if highpass:
        image = highpass_median_filter(image)
    image = fft2d(image)
    image = log_scale(image, scale_factor)
    return image

def fft2d_tensor_wrapper(image, scale_factor=1):
    image = torch.fft.fft2(image)
    image = torch.fft.fftshift(image)
    image = scale_factor * torch.log(abs(image))
    # image = torch.einsum("bchw->bhwc", image).cpu().numpy()
    # image = log_scale(image, scale_factor)
    return image

def dct2d_tensor_wrapper(image, scale_factor=1):
    '''
    DCT is a special case of the Discrete Fourier Transform (DFT) where the input signal is real-valued.
    Therefore, instead of using a separate DCT function, we can use the rfft function in PyTorch to perform the DCT transform on the tensor.
    TODO: change to torch_dct
    '''
    image = torch.rfft(image, signal_ndim=2, normalized=True, onesided=True)
    image = torch.fft.fftshift(image)
    image = scale_factor * torch.log(abs(image))
    return image

def fft2d_tensor_wrapper_v2(image, shift=False, logscale_factor=None):
    # TODO: move highpass inside fft transform (in dataset now)
    image = torch.abs(torch.fft.fft2(image)) + 1e-6
    if shift:
        image = torch.fft.fftshift(image)
    if logscale_factor is not None:
        image = logscale_factor * torch.log(image)
    return image

def dct2d_tensor_wrapper_v2(image, logscale_factor=None):
    # TODO: log scale as fft
    image = torch_dct.dct_2d(image)
    if logscale_factor is not None:
        image = logscale_factor * torch.log(torch.abs(image))
    return image

def idct2d_tensor_wrapper_v2(image, logscale_factor=None):
    # TODO: log scale as fft
    image = torch_dct.idct_2d(image)
    if logscale_factor is not None:
        image = logscale_factor * torch.log(torch.abs(image))
    return image


if __name__ == "__main__":
    savepath = "/data/tmpfiles/test_fft"
    path = "frame00120.png"
    image = load_image(path=path, img_size=256, grayscale=True)

    norm_transform_gray = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    scale_factor = 20
    image_tensor = norm_transform_gray(image)

    image_dct = dct2d_tensor_wrapper_v2(image_tensor, logscale_factor=None)
    cv2.imwrite(os.path.join(savepath, "dct_tensor_log20.png"), torch.einsum("chw->hwc", image_dct).numpy())

    image_idct = idct2d_tensor_wrapper_v2(image_dct, logscale_factor=None)
    cv2.imwrite(os.path.join(savepath, "idct_tensor_log20.png"), torch.einsum("chw->hwc", image_idct).numpy())

    # scale_factor = 10
    # image_fft = fft2d_wrapper(image, scale_factor=scale_factor, highpass=False)
    # image_dct = dct2d_wrapper(image, scale_factor=scale_factor, highpass=False)

    # cv2.imwrite(os.path.join(savepath, "fft.png"), image_fft)
    # cv2.imwrite(os.path.join(savepath, "dct.png"), image_dct)