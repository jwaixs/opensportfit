import cv2
import numpy
import time
from matplotlib import pyplot as plt

def set_res(cap, x,y):
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(y))

def detect_chessboard(frame, pattern_size=(7,7), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    pattern_points = numpy.zeros((numpy.prod(pattern_size), 3), numpy.float32)
    pattern_points[:,:2] = numpy.indices(pattern_size).T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(frame, pattern_size)
    
    if ret:
        #print('Chessboard found.')
        cv2.cornerSubPix(frame, corners, (5,5), (-1,-1), criteria)
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        return True, frame, pattern_points, corners
    else:
        #print('Chessboard not found.')
        return False, frame, None, None

def webcam_calibration(cap, limit=8):
    '''Calibrate webcam with chessboard method.'''

    # Arrays to store object points and image points of all the images.
    object_points = [] # 3d points in real world space.
    image_points = [] # 2d points in image plane.

    height, width = 0, 0

    while len(image_points) != limit:
        time.sleep(0.5)

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        cv2.imshow('Chessboard', frame)

        ret, chessboard, object_point, image_point = detect_chessboard(gray)

        if ret:
            object_points.append(object_point)
            image_points.append(image_point)
            print 'Found {0} chessboards.'.format(len(image_points))

    return cv2.calibrateCamera(object_points, image_points, (width, height), None, None)

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

    cv2.namedWindow('Frame 1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Frame 2', cv2.WINDOW_AUTOSIZE)

    i = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        cv2.imshow('Frame 1', frame1)
        cv2.imshow('Frame 2', frame2)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('chess_1_{0}.jpg'.format(str(i)), frame1)
        cv2.imwrite('chess_2_{0}.jpg'.format(str(i)), frame2)

        time.sleep(0.5)

        i += 1

        if i == 30:
            break

        #ret1, chess1, _, _ = detect_chessboard(gray1)
        #ret2, chess2, _, _ = detect_chessboard(gray2)

        #cv2.imshow('Chess 1', chess1)
        #cv2.imshow('Chess 2', chess2)

        #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=10*16, SADWindowSize=5)
        #disparity = stereo.compute(gray2, gray1, disptype=cv2.CV_32F)
        #norm_coeff = 255 / disparity.max()
        #
        #cv2.imshow('Depth', disparity * norm_coeff / 255)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
