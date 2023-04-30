import numpy as np
import cv2
import glob
import g_params

chessboardSize = (7, 7)
frame_size = g_params.resolution


def save_calibration_result():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = 25
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('.\\calib_camera\\*.jpg')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(400)

    cv2.destroyAllWindows()
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    np.savetxt('.\\calib_camera\\camera_matrix.txt', camera_matrix)
    np.savetxt('.\\calib_camera\\dist.txt', dist)

# pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
# pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
# pickle.dump(dist, open("dist.pkl", "wb"))


def calibrate_image(image):
    # load calibration matrix, vector
    camera_matrix = np.loadtxt('.\\calib_camera\\camera_matrix.txt')
    dist = np.loadtxt('.\\calib_camera\\dist.txt')

    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(image, camera_matrix, dist, None, new_camera_matrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def calibrate_remap_image(image):
    # load calibration matrix, vector
    camera_matrix = np.loadtxt('.\\calib_camera\\camera_matrix.txt')
    dist = np.loadtxt('.\\calib_camera\\dist.txt')

    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, new_camera_matrix, (w, h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    return dst

# # Reprojection Error
# mean_error = 0
#
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
#     mean_error += error
#
# print("total error: {}".format(mean_error / len(objpoints)))


def capture_image(save_path, prefix):
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    num = 0
    while True:
        _, image = capture.read()
        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(save_path + prefix + str(num) + '.jpg', image)
            print("image saved!" + str(num))
            num += 1
        show_image = cv2.resize(image, (960, 540))
        cv2.imshow('Img', show_image)
    # Release and destroy all windows before termination
    capture.release()
    cv2.destroyAllWindows()


def main():
    save_path = '.\\webcam\\'
    prefix = 'test'
    # capture_image(save_path, prefix)
    save_calibration_result()
    # image = cv2.imread('.\\webcam\\test.jpg', 0)
    # image = calibrate_remap_image(image)
    # cv2.imwrite('caliResult2.jpg', image)


if __name__ == '__main__':
    main()
