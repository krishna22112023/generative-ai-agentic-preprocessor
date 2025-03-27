import cv2
import numpy as np

def dehaze(image, filter_size=7, sigma_color=75, sigma_space=75):
    """
    Simple dehazing using bilateral filtering.
    
    Args:
        image (np.ndarray): Input image.
        filter_size (int): Diameter of each pixel neighborhood.
        sigma_color (int): Filter sigma in the color space.
        sigma_space (int): Filter sigma in the coordinate space.
        
    Returns:
        np.ndarray: Dehazed image.
    """
    # Bilateral filter preserves edges while smoothing color variations.
    dehazed = cv2.bilateralFilter(image, filter_size, sigma_color, sigma_space)
    return dehazed

def denoise(image, method="gaussian", kernel_size=3):
    """
    Remove noise from an image.
    
    Args:
        image (np.ndarray): Input image.
        method (str): 'gaussian' or 'median'.
        kernel_size (int): Kernel size (must be odd for median blur).
        
    Returns:
        np.ndarray: Denoised image.
    """
    if method == "gaussian":
        # GaussianBlur expects kernel size tuple.
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        denoised = cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("Unsupported denoising method. Choose 'gaussian' or 'median'.")
    return denoised

def enhance_contrast(image, clipLimit=2.0, tileGridSize=(8,8)):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image (np.ndarray): Input image in BGR format.
        clipLimit (float): Threshold for contrast limiting.
        tileGridSize (tuple): Size of grid for histogram equalization.
        
    Returns:
        np.ndarray: Contrast enhanced image.
    """
    # Convert from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def sharpen(image, kernel_size=3):
    """
    Sharpen the image using a kernel convolution.
    
    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Size of the kernel used for sharpening.
        
    Returns:
        np.ndarray: Sharpened image.
    """
    # A common sharpening kernel for kernel_size=3.
    if kernel_size == 3:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
    else:
        # For other kernel sizes, create a normalized Laplacian kernel and add identity.
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[(kernel_size // 2), (kernel_size // 2)] = 2.0
        avg_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        kernel = kernel - avg_kernel

    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
