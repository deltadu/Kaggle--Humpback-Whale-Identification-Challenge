import numpy as np

def normalizeInput(inputImages):
    """
    Normalize input such that the mean is 0.0 and std is 1.0 for each channel

    Args:
        inputImages: an Numpy array that contains all the images. Assume the input is 3D or 4D array
                         3D: (imageIdx, x, y);
                         4D: (imageIdx, x, y, colorChannel)
    Returns:
        Numpy array: in-place modified image array
    """
    dimension = len(inputImages.shape)

    # single color channel
    if dimension == 3:
        imageMean = np.mean(inputImages)
        imageStd = np.std(inputImages)

        inputImages -= imageMean
        inputImages /= imageStd
    # multiple color channels
    else:
        # normalize for each channel
        for channelIdx in range(inputImages.shape[dimension-1]):
            imageMean = np.mean(inputImages[:,:,:,channelIdx])
            imageStd = np.std(inputImages[:,:,:,channelIdx])

            inputImages[:,:,:,channelIdx] -= imageMean
            inputImages[:,:,:,channelIdx] /= imageStd
    return inputImages
