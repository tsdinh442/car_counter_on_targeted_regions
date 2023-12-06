import numpy as np
import cv2

def masking(image, polygons, opacity):

    '''
    Parameters
    ----------
    image: str - path to an image
    polygons: list - list of vertices (x, y)
    opacity: float - opacity value

    Return: np array - image with mask overlaid
    -------
    '''

    # create a blank white image with same dimension as the org image
    blank = np.ones_like(image, dtype=np.uint8) * 255

    # create a blank black image with the same dimension as the ori image
    mask = np.zeros_like(image, dtype=np.uint8)

    # fill the polygon with white color on the blank black image
    cv2.fillPoly(mask, polygons, (255, 255, 255))

    # blend the mask image with the original image
    blended = cv2.addWeighted(src1=image, alpha=opacity, src2=blank, beta=1 - opacity, gamma=0)

    # perform masking
    result = cv2.bitwise_and(blended, 255 - mask) + cv2.bitwise_and(image, cv2.bitwise_not(255 - mask))


    return result