import cv2
import numpy
from matplotlib import pyplot as plt

def set_res(cap, x,y):
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(y))

def undistort_params(frame, calibration):
    ret, mtx, dist, rvecs, tvecs = calibration
    height, width = frame.shape[:2]
    newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    return ret, mtx, dist, newcamera_mtx, roi

def undistort_frame(frame, mtx, dist, newcamera_mtx, roi):
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcamera_mtx)
    x, y, w, h = roi
    
    return undistorted_frame[y:y+h, x:x+w]

def run():
    cap1 = cv2.VideoCapture(1)
    set_res(cap1, 1280, 720)
    cap2 = cv2.VideoCapture(2)
    set_res(cap2, 1280, 720)

    LIMIT = 20

    cv2.namedWindow('Frame 1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Frame 2', cv2.WINDOW_AUTOSIZE)
        
    object_points1 = []
    image_points1 = []

    object_points2 = []
    image_points2 = []

    mtx1, dist1, newcamera_mtx1, roi1 = numpy.load('cal1.npy')
    mtx2, dist2, newcamera_mtx2, roi2 = numpy.load('cal2.npy')

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        height, width = frame1.shape[:2]

        frame1 = undistort_frame(frame1, mtx1, dist1, newcamera_mtx1, roi1)
        frame2 = undistort_frame(frame2, mtx2, dist2, newcamera_mtx2, roi2)

        cv2.imshow('Frame 1', frame1)
        cv2.imshow('Frame 2', frame2)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=64, SADWindowSize=19)
        disparity = stereo.compute(gray2, gray1, disptype=cv2.CV_32F)
        norm_coeff = 255 / disparity.max()
        
        cv2.imshow('Depth', disparity * norm_coeff / 255)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
