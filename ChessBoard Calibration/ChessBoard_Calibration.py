
import numpy as np
import cv2

def init_video():
    cv2.namedWindow("cam1")
    cv2.namedWindow("cam2")
    vc1 = cv2.VideoCapture(0)
    vc2 = cv2.VideoCapture(3)

    if vc1.isOpened():
        rval, frame1 = vc1.read()
        rval, frame2 = vc2.read()
    else:
        rval = False
        frame1 = None
        frame2 = None

    return vc1, vc2, rval, frame1, frame2

if __name__ == "__main__":
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane
    imgpoints2 = [] # 2d points in image plane

    numPicsTaken = 0
    num = 1
    vc1, vc2, rval, frame1, frame2 = init_video()
    stereo = cv2.StereoBM_create()

    stereo.setMinDisparity(4)
    
    stereo.setBlockSize(21)
    #stereo.setSpeckleRange(16)
    #stereo.setSpeckleWindowSize(45)
    while (rval and (numPicsTaken < 10)):
        rval, frame1 = vc1.read()
        rval, frame2 = vc2.read()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        cv2.imshow("cam1", frame1)
        cv2.imshow("cam2", frame2)
        #--Drawing and detecting code example--

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        elif key == ord('w'):
            num += 1
        elif key == ord('s'):
            num -= 1
        elif key == ord('c'):
            ret1, corners1 = cv2.findChessboardCorners(gray1, (7,6), None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (7,6), None)
            if ret1 and ret2 == True:
                numPicsTaken += 1
                objpoints.append(objp)
                cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
                cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                cv2.drawChessboardCorners(frame1, (7,6), corners1, ret1)
                cv2.drawChessboardCorners(frame2, (7,6), corners2, ret2)
                cv2.imshow('chessboard1', frame1)
                cv2.imshow('chessboard2', frame2)
                print(numPicsTaken)

    cv2.destroyAllWindows()

    if numPicsTaken >= 10:
        print('Calibrating Camera...')

        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1],None,None)
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)

        h1 = 0
        w1 = 0

        while rval:
            rval, frame1 = vc1.read()
            rval, frame2 = vc2.read()

            # undistort
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w1,h1), 1,(w1,h1))
            newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (w2,h2), 1,(w2,h2))
            dst1 = cv2.undistort(frame1, mtx1, dist1, None, newcameramtx1)
            dst2 = cv2.undistort(frame2, mtx2, dist2, None, newcameramtx2)
            
            #crop image
            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2
            dst1 = dst1[y1:y1+h1, x1:x1+w1]
            dst2 = dst2[y2:y2+h2, x2+x2+w2]
            cv2.imshow("cam1Calibrated", dst1)
            cv2.imshow("cam2Calibrated", dst2)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
            
        (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
            objpoints, imgpoints2, imgpoints1,
            mtx2, dst2,
            mtx1, dst1,
            (w1, h1), None, None, None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

        np.savez_compressed(outputFile, )