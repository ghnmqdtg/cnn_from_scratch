# ----------------------------------------------------------------------------
# Filename    : utils.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ---------------------------------------------------------------------------
import os
import cv2
import numpy as np


def create_folder(dir) -> None:
    """
    Create a folder if it doesn't already exist
    """

    if not os.path.exists(dir):
        os.makedirs(dir)


def load_picture(src_path: str, as_float=False) -> np.array:
    """
    Load and resize picture and return is as np.array

    Parameters
    ----------
    src_path : str
        The path to the source picture
    """

    img = cv2.imread(src_path)

    if as_float:
        img = img.astype(np.float32)

    return img


def save_picture(dest_path: str, img: np.array) -> None:
    """
    Save np.array image data in JPG

    Parameters
    ----------
    dest_path : str
        The path to save the picture
    img : np.array
        Image data in np.array
    """

    result = cv2.imwrite(f'{dest_path}.jpg', img)


def parse_folder(folder: str):
    """
    Parse a folder and return the path of files in it

    Parameters
    ----------
    folder: str
        Path to the parent folder
    """

    filepaths = []

    for dirPath, dirNames, fileNames in os.walk(folder, topdown=True):
        # Skip the root directory that os.walk returns
        dirPath = dirPath.replace('\\', '/')
        for f in [f for f in fileNames]:
            filepaths.append(
                os.path.join(dirPath, f).replace('\\', '/'))
        filepaths.sort()

    return filepaths
