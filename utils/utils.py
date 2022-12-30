# ----------------------------------------------------------------------------
# Filename    : utils.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ---------------------------------------------------------------------------
import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import config


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
            # Rename files
            basename = os.path.basename(f)
            filename, extension = os.path.splitext(basename)

            if extension != '.csv':
                desire_name = os.path.join(dirPath, f'{filename:0>4}{extension}').replace(
                    '\\', '/')

                os.rename(
                    os.path.join(dirPath, f).replace('\\', '/'),
                    desire_name
                )

                filepaths.append(desire_name)
            else:
                filepaths.append(os.path.join(dirPath, f).replace('\\', '/'))

        filepaths.sort()

    return filepaths


def plot_results(file_paths, preds, labels):
    """
    Plot and save the results

    Parameters
    ----------
    file_paths : list
        List of file paths
    preds : list
        List of predictions
    labels : list
        List of labels
    """

    fig, axs = plt.subplots(3, 5, figsize=(12, 6))
    fig.suptitle('Sampling Test', fontsize=20)

    for idx, (file_path, pred, true) in enumerate(zip(file_paths, preds, labels)):
        row = idx // 5
        col = idx % 5
        filename = int(os.path.splitext(os.path.basename(file_path))[0])

        img = load_picture(file_path)
        axs[row, col].imshow(img)
        axs[row, col].tick_params(top=False, bottom=False, left=False, right=False,
                                  labelleft=False, labelbottom=False)

        if pred != true:
            axs[row, col].set_title(f'{pred} ({filename})', color='r')
        else:
            axs[row, col].set_title(f'{pred} ({filename})')

    # plt.show()
    plt.savefig(f'{config.DST_FOLDER}/result.jpg')


def plot_confusion_matrix(confusion_matrix):
    """
    Plot and save the confusion matrix

    Parameters
    ----------
    confusion_matrix : list
        List of confusion matrices
    """

    df = pd.DataFrame(confusion_matrix,
                      index=[i for i in "01"],
                      columns=[i for i in "01"])

    plt.figure(figsize=(10, 7))
    plt.title('Confusion matrix', fontdict={'fontsize': 28})
    plt.tick_params(labelsize=24)
    sn.heatmap(df, annot=True, annot_kws={'size': 32})
    plt.savefig(f'{config.DST_FOLDER}/confusion_matrix.jpg')
