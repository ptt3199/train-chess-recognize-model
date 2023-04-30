import cv2


def capture_image():
    name = '.\\webcam\\'
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    num = 0
    while True:
        _, image = capture.read()
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite(name + str(num) + '.jpg', image)
            print("image saved!" + str(num))
            num += 1
            cv2.imshow('Img', image)
        # Release and destroy all windows before termination
        capture.release()
        cv2.destroyAllWindows()


def main():
    capture_image()


if __name__ == '__main__':
    main()
