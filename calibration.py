import cv2
import numpy
from matplotlib import pyplot as plt
import time

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

    LIMIT = 4

    cv2.namedWindow('Frame 1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Frame 2', cv2.WINDOW_AUTOSIZE)
        
    object_points1 = []
    image_points1 = []

    object_points2 = []
    image_points2 = []

    time_old = time.time()
    time_new = time.time()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        height, width = frame1.shape[:2]

        #cv2.imshow('Frame 1', frame1)
        #cv2.imshow('Frame 2', frame2)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        time_new = time.time()

        if (time_new - time_old) > 1:
            ret1, chess1, object_point1, image_point1 = detect_chessboard(gray1)
            ret2, chess2, object_point2, image_point2 = detect_chessboard(gray2)

            cv2.imshow('Chess 1', chess1)
            cv2.imshow('Chess 2', chess2)

            if ret1 and ret2 and len(image_points1) < LIMIT:
                object_points1.append(object_point1)
                image_points1.append(image_point1)

                print 'Found {0} chessboards for 1.'.format(len(image_points1))

                object_points2.append(object_point2)
                image_points2.append(image_point2)
                
                print 'Found {0} chessboards for 2.'.format(len(image_points2))
            else:
                print 'Did not found chessboards.'

            time_old, time_new = time.time(), time.time()


        if len(image_points1) == LIMIT and len(image_points2) == LIMIT:
            calibration1 = cv2.calibrateCamera(object_points1, image_points1, (width, height), None, None)
            calibration2 = cv2.calibrateCamera(object_points2, image_points2, (width, height), None, None)
    
            ret, mtx1, dist1, newcamera_mtx1, roi1 = undistort_params(frame1, calibration1)
            ret, mtx2, dist2, newcamera_mtx2, roi2 = undistort_params(frame2, calibration2)

            numpy.save('cal1', [mtx1, dist1, newcamera_mtx1, roi1])
            numpy.save('cal2', [mtx2, dist2, newcamera_mtx2, roi2])

            break


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
