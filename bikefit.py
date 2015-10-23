# Bike fit initial source file
# Created by Noud Aldenhoven

import cv2
import numpy
import Image

import cv
import math

class Object:
    def __init__(self, camnr):
        self.camnr = camnr
        self.capture = cv.CaptureFromCAM(camnr)
        cv.NamedWindow("Object", 1)
        #cv.NamedWindow("HSV", 1)
        cv.NamedWindow("Red", 1)
        cv.NamedWindow("Output", 1)


    def run(self):
        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)

        # Initial images
        hsv_img = cv.CreateImage(cv.GetSize(cv.QueryFrame(self.capture)),8,3)
        threshold_img1 = cv.CreateImage(cv.GetSize(hsv_img),8,1)
        threshold_img1a = cv.CreateImage(cv.GetSize(hsv_img),8,1)
        threshold_img2 = cv.CreateImage(cv.GetSize(hsv_img),8,1)

        # Write video
        #width, height = cv.GetSize(hsv_img)
        #out = cv2.VideoWriter('output.avi', -1, 20.0, (width, height))

        i = 0

        writer = cv.CreateVideoWriter("angle_tracking.avi",cv.CV_FOURCC('M','J','P','G'),30,cv.GetSize(hsv_img),1)

        circles = []

        while True:
            # Capture image from cam camnr
            img = cv.QueryFrame(self.capture)
            output = img

            # Object image
            cv.ShowImage("Object", img)
            

            # HSV image
            cv.CvtColor(img, hsv_img, cv.CV_BGR2HSV)
            #cv.ShowImage("HSV", hsv_img)


            # Red Threshold
            #cv.InRangeS(hsv_img, (165,145,100), (250,210,160), threshold_img1)
            #cv.InRangeS(hsv_img, (0,145,100), (10,210,160), threshold_img1a)
            #cv.Add(threshold_img1, threshold_img1a, threshold_img1)            
            #cv.InRangeS(hsv_img, (160, 100, 100), (179, 255, 255), threshold_img1)
            #cv.InRangeS(hsv_img, (0, 200, 200), (10, 255, 255), threshold_img1)
            #cv.ShowImage("Red", threshold_img1)
            #cv.InRangeS(hsv_img, (159, 135, 135), (179, 255, 255), threshold_img1)
            #cv.InRangeS(hsv_img, (0, 135, 135), (20, 255, 255), threshold_img1a)


            # This is yellow, not red!
            cv.InRangeS(hsv_img, cv.Scalar(20, 100, 100), cv.Scalar(30, 255, 255), threshold_img1)
            cv.Add(threshold_img1, threshold_img1a, threshold_img1)            
            

            img_size = cv.GetSize(img)
            gray_img = cv.CreateImage(img_size, cv.IPL_DEPTH_8U, 1)
            test = cv.CreateImage(img_size, 8, 3)

            img_draw = cv.CreateImage(img_size, 8, 3)
            #cv.Flip(img, img, 1)
            cv.Smooth(img, img, cv.CV_GAUSSIAN, 3, 0)
            cv.Erode(threshold_img1, threshold_img1, None, 2)
            cv.Dilate(threshold_img1, threshold_img1, None, 2)
            
            cv.ShowImage("Red", threshold_img1)

            storage = cv.CreateMemStorage(0)
            contour = cv.FindContours(threshold_img1, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            points = []

            while contour:
                bound_rect = cv.BoundingRect(list(contour))
                contour = contour.h_next()

                pt1 = (bound_rect[0], bound_rect[1])
                pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])

                points.append(((pt1[0] + pt2[0])/2, (pt1[1] + pt2[1])/2))

                cv.Rectangle(output, pt1, pt2, cv.CV_RGB(0, 255, 255), 1)

            for x, y in points:
                cv.Circle(output, (x,y), 1, (255, 255, 255), 5)
            
            if len(points) == 3:
                points.sort(key = lambda elm : elm[0])

                p1 = points[2]
                p2 = points[0] if points[0][1] > points[1][1] else points[1]
                p3 = points[0] if points[0][1] < points[1][1] else points[1]

                cv.Line(output, p1, p2, (255, 255, 255), 2)
                cv.Line(output, p2, p3, (255, 255, 255), 2)

                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3

                a = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
                c = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)

                angle = int(math.acos((b**2 + a**2 - c**2)/(2*a*b))*180/math.pi)

                cv.PutText(output, str(angle), (x2+20, y2-20), font, 255)

            #out.write(numpy.asarray(output[:,:]))
            cv.WriteFrame(writer, output)

            cv.ShowImage("Output", output)


            #threshold_img = cv.GetMat(threshold_img1)
            #moments = cv.Moments(threshold_img)
            #area = cv.GetCentralMoment(moments, 0, 0)

            #if area > 200000:
            #    x, y = int(cv.GetSpatialMoment(moments, 1, 0)/area), int(cv.GetSpatialMoment(moments, 0, 1)/area)
            #    cv.Circle(img, (x,y), 2, (0,255,0), 20)
            #cv.ShowImage("Output", img)

            ##image = numpy.asarray(img[:,:])
            #image = numpy.asarray(threshold_img1[:,:])
            ##image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #output = cv2.GaussianBlur(image, (9,9), 2, 2)
            #cv2.imshow("Output", output)
           
            
            # Detect circles
            #output = numpy.asarray(img[:,:])
            #gray = cv2.cvtColor(cv2.medianBlur(output, 5), cv2.COLOR_BGR2GRAY)

            #new_circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.3, 10)
            #if new_circles is not None:
            #    circles = numpy.round(new_circles[0, :]).astype("int")
            #if len(circles) != 0:
            #    print circles
            #    for (x, y, r) in circles:
            #        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            #        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            #cv2.imshow("Output", output)

            #circles = np.uint16(np.around(circles))
            #for i in circles[0,:]:
            #    # draw the outer circle
            #    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            #    # draw the center of the circle
            #    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)


            #img_circ = cv.CreateImage(cv.GetSize(cv.QueryFrame(self.capture)),8,3)
            ####gray = cv.cvtColor(img_circ, cv.COLOR_BGR2GRAY)
            #circles = cv.HoughCircles(img_circ, cv.CV_HOUGH_GRADIENT, 1.2, 100)
            #if circles is not None:
            #    circles = np.round(circles[0, :]).astype("int")
            #    for (x, y, r) in circles:
            #        cv.circle(img_circ, (x, y), r, (0, 255, 0), 4)
            #        cv.rectangle(img_circ, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            #cv.ShowImage("Output", cimg)
 

            # Listen for ESC or ENTER key
            key = cv.WaitKey(7) % 0x100
            if key in [10, 27]:
                break

        cv.DestroyAllWindows()


if __name__ == '__main__':
    mainobj = Object(0)
    mainobj.run()
