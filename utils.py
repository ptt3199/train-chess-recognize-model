import os
from shutil import rmtree
import cv2


def into_black_white(image, thresh=150, show=False):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow("Binary image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def clean(path):  # delete all in path
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            