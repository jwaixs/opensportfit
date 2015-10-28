# Put a lot of comments, because this is a test program.

import cv2
import numpy
import time
import sys
import matplotlib
from matplotlib import pyplot as plt

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

def drawlines(frame1, frame2, lines, pts1, pts2):
    print frame1
    r, c = frame1.shape

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(numpy.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c)/r[1]])

        frame1 = cv2.line(frame1, (x0, y0), (x1, y1), color, 1)
        frame1 = cv2.circle(frame1, tuple(pt1), 5, color, -1)
        
        frame2 = cv2.circle(frame2, tuple(pt2), 5, color, -1)

def epipolar_geometry(frame1, frame2):
    #sift = cv2.SIFT()

    # Find the keypoints and descriptors with SIFT
    #kp1, des1 = sift.detectAndCompute(frame1, None)
    #kp2, des2 = sift.detectAndCompute(frame2, None)

    # Trying ORB instead of SIFT
    orb = cv2.ORB()

    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    des1, des2 = map(numpy.float32, (des1, des2))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0 
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good, pts1, pts2 = [], [], []

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    return F, mask

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1, _ = drawlines(frame1, frame2, lines1, pts1, pts2)
    
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)
    img2, _ = drawlines(frame2, frame1, lines2, pts2, pts1)

    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.imshow(img1)
    matplotlib.pyplot.subplot(122)
    matplotlib.pyplot.imshow(img2)
    matplotlib.show()
    


def run():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    cv2.namedWindow('Frame 1', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Frame 1', 1920, 0)

    cv2.namedWindow('Frame 2', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Frame 2', 1920+640, 0)

    if len(sys.argv) > 2:
        if sys.argv[1] == '-c' and sys.argv[2] == '1':
            print('Calibrating webcam 1')
            calibration1 = webcam_calibration(cap1)
            ret1, frame1 = cap1.read()
            ret, mtx1, dist1, newcamera_mtx1, roi1 = undistort_params(frame1, calibration1)
            numpy.save('cal1', [mtx1, dist1, newcamera_mtx1, roi1])
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            return
        if sys.argv[1] == '-c' and sys.argv[2] == '2':
            print('Calibrating webcam 2')
            calibration2 = webcam_calibration(cap1)
            ret2, frame2 = cap2.read()
            ret, mtx2, dist2, newcamera_mtx2, roi2 = undistort_params(frame2, calibration2)
            numpy.save('cal2', [mtx2, dist2, newcamera_mtx2, roi2])
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            return

    mtx1, dist1, newcamera_mtx1, roi1 = numpy.load('cal1.npy')
    mtx2, dist2, newcamera_mtx2, roi2 = numpy.load('cal2.npy')

    while True:
        # Capture frame by frame.
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        #frame1 = undistort_frame(frame1, mtx1, dist1, newcamera_mtx1, roi1)
        #frame2 = undistort_frame(frame2, mtx2, dist2, newcamera_mtx2, roi2)

        #epipolar_geometry(frame1, frame2)

        # Gray images, which are needed for the main processes.
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=16, SADWindowSize=5)
        disparity = stereo.compute(gray2, gray1, disptype=cv2.CV_32F)
        norm_coeff = 255 / disparity.max()
        
        #plt.imshow(disparity, 'gray')
        #plt.show()
        #break

        #matplotlib.pyplot.imshow(disparity, 'gray')
        #matplotlib.pyplot.show()

        # Display input frames.
        cv2.imshow('Frame 1', frame1)
        cv2.imshow('Frame 2', frame2)
        cv2.imshow('Depth', disparity * norm_coeff / 255)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    
    cv2.destroyAllWindows()

        

if __name__ == '__main__':
    run()
