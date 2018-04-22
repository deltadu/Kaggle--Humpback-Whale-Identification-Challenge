import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import collections
import os

def getImageHash(imagePath):
    """
    Fuction to calculate the image's hash value given the image path
    
    Args:
        imagePath: the path to the image

    Returns:
        str: hash value of the image.
    """
    with Image.open(imagePath) as img:
        imgHash = imagehash.phash(img)
        return str(imgHash)

def getImageMetaData(targetCSVFile, targetImageFolder):
    """
    Fuction to load the CSV file and append the hash value of the corresponding images
    
    Args:
        targetCSVFile: the path to the CSV file
        targetImageFolder: the folder of the images

    Returns:
        pandas.DataFrame: data of the CSV with index (autoincrement); Image (filename); Id (label); Hash
    """
    image_input = pd.read_csv(targetCSVFile)
    
    imgHashes = image_input.Image.apply(lambda imageFile: getImageHash(os.path.join(targetImageFolder,imageFile)))
    image_input["Hash"] = [hashValue for hashValue in imgHashes]
    
    return image_input

def getHashWithDuplicate(dataWithHash):
    hashAndCounts = dataWithHash.Hash.value_counts()
    return hashAndCounts.loc[hashAndCounts>1]

def showDuplicateData(dataWithHash, ignoredNewWhale = True):
    """
    Fuction to show duplicate data in the dataset (duplicate images with different labels)
    
    Args:
        dataWithHash: data obtained from get_train_input (included hash value)
        ignoredNewWhale: the folder of the images
    """
    hashWithDuplicate = getHashWithDuplicate(dataWithHash)
    
    newWhaleCount = 0
    numOfDuplicate = 0
    _numOfDuplicate = 0
    for hashValue in hashWithDuplicate.index:
        duplicatedInfo = dataWithHash[dataWithHash.Hash==hashValue]
        shownId = set(duplicatedInfo.Id)
        
        numOfDuplicate = numOfDuplicate + (len(duplicatedInfo)-len(shownId))
        _numOfDuplicate = _numOfDuplicate + len(duplicatedInfo)
        if 'new_whale' in shownId:
            newWhaleCount += 1
            numOfDuplicate -= 1
        if ignoredNewWhale:
            shownId.discard('new_whale')
            
        if len(shownId) >= 1:
            print("Duplicate images: {}, set of Ids: {}".format(duplicatedInfo.Image.tolist(), shownId))
    if not ignoredNewWhale:
        print("Number of 'new whale' in duplicate images: {}".format(newWhaleCount))
    print("Number of duplicates: {}".format(numOfDuplicate))
    print("Number of _duplicates: {}".format(_numOfDuplicate))
            
def inconsistentDataIndex(dataWithHash, newWhaleOnly = True):
    """
    Fuction to show inconsistent data in the dataset (duplicate images with different labels)
    
    Args:
        dataWithHash: data obtained from get_train_input (included hash value)
        ignoredNewWhale: the folder of the images

    Returns:
        pandas.DataFrame: data of the CSV with index (autoincrement); Image (filename); Id (label); Hash
    """
    hashWithDuplicate = getHashWithDuplicate(dataWithHash)
    
    targetList = []
    for hashValue in hashWithDuplicate.index:
        duplicatedInfo = dataWithHash[(dataWithHash.Hash==hashValue) & ((dataWithHash.Id == 'new_whale') | (not newWhaleOnly))]
        
        if len(duplicatedInfo.index.values)>0:
            targetList += duplicatedInfo.index.values.tolist()
    return targetList

def removeInconsistentData(dataWithHash, targetCSVFile, targetImageFolder, newWhaleOnly = True):
    rowIndices = inconsistentDataIndex(dataWithHash, newWhaleOnly)
    
    # remove images
    for imagePath in dataWithHash.loc[dataWithHash.index.isin(rowIndices)].Image.tolist():
        os.remove(os.path.join(targetImageFolder, imagePath))
    
    # drop rows in CSV
    dataWithHash.drop(rowIndices, inplace=True)
    dataWithHash.to_csv(targetCSVFile, index=False, columns=['Image', 'Id'])

def removeDuplicateData(dataWithHash, targetCSVFile, targetImageFolder):
    # remove images
    for imagePath in dataWithHash[dataWithHash.duplicated(subset=['Id','Hash'], keep='first')].Image.tolist():
        os.remove(os.path.join(targetImageFolder, imagePath))
    
    # drop rows in CSV
    dataWithHash.drop_duplicates(subset=['Id','Hash'], keep='first', inplace=True)
    dataWithHash.to_csv(targetCSVFile, index=False, columns=['Image', 'Id'])
