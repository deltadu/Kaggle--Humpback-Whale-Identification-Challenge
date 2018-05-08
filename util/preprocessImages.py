import numpy as np

def normalizeInput(inputImages, precalculatedMeans = None, precalculatedStds = None):
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
    imageMeans = None
    imageStds = None

    # single color channel
    if dimension == 3:
        if precalculatedMeans is None:
            imageMean = np.mean(inputImages)
            imageStd = np.std(inputImages)
        else:
            imageMean = precalculatedMeans[0]
            imageStd = precalculatedStds[0]

        inputImages -= imageMean
        inputImages /= imageStd
        imageMeans = np.array([imageMean])
        imageStds = np.array([imageStd])
    # multiple color channels
    else:
        # normalize for each channel
        imageMeans = np.zeros(inputImages.shape[dimension-1])
        imageStds = np.zeros(inputImages.shape[dimension-1])
        for channelIdx in range(inputImages.shape[dimension-1]):
            if precalculatedMeans is None:
                imageMeans[channelIdx] = np.mean(inputImages[:,:,:,channelIdx])
                imageStds[channelIdx] = np.std(inputImages[:,:,:,channelIdx])
            else:
                imageMeans[channelIdx] = precalculatedMeans[channelIdx]
                imageStds[channelIdx] = precalculatedStds[channelIdx]

            inputImages[:,:,:,channelIdx] -= imageMeans[channelIdx]
            inputImages[:,:,:,channelIdx] /= imageStds[channelIdx]
    return inputImages, imageMeans, imageStds
