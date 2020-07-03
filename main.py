import os 
import cv2 as cv 
import numpy as np
from mtcnn import MTCNN
from utils import show_keypoints, getKeypointsObj, getListFromArray, getKeypointsFromDict


class Main:
    def __init__(self, arImagePath, live=True, selectOrdinate= False):
        self.ARIMAGE = cv.imread(arImagePath)
        self.WINDOWNAME = 'getcordinate'
        self.CAP = cv.VideoCapture(0)
        self.DETECTOR = MTCNN()
        self.LABELS = iter(['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'])
        self.hT, self.wT, self.cT = self.ARIMAGE.shape
        self.CLICKCOUNT = 0
        self.SELECTORDINATE = selectOrdinate
        if selectOrdinate:
            self.ARIMAGECOORDINATES = {}
        else:
            self.ARIMAGECOORDINATES = {'left_eye': (444, 440), 'right_eye': (584, 433), 'nose': (511, 516), 'mouth_left': (469, 576), 'mouth_right': (553, 573)}
    
    def getOrdinatesArImage(self):
        cv.imshow(self.WINDOWNAME, self.ARIMAGE)
        cv.setMouseCallback(self.WINDOWNAME, self.callback, param={'image': self.ARIMAGE.copy()})
        cv.waitKey(0)
        cv.destroyWindow(self.WINDOWNAME)
    
    def callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.CLICKCOUNT <= 4:
                image = param['image']
                image = cv.circle(image, (x, y), 3, (0, 255, 0), -1)
                cv.imshow(self.WINDOWNAME, image)
                self.ARIMAGECOORDINATES[next(self.LABELS)] = (x, y)
                print('Co-ordinates : {}'.format(self.ARIMAGECOORDINATES))
                print(self.CLICKCOUNT)
                self.CLICKCOUNT += 1
            else:
                print('Can"t add coordinates')
    
    def videoRecord(self):
        while True:
            sucess, self.frame = self.CAP.read()
            if sucess:
                self.frame = cv.resize(self.frame, (500, 500))
                cv.imshow('frame', self.frame)
                self.result = self.DETECTOR.detect_faces(self.frame)
                if self.result:
                    framKeypoints = getKeypointsFromDict(self.result)
                    arKeypoints = getKeypointsObj([values for _, values in self.ARIMAGECOORDINATES.items()])
                    dstPts = np.float32([kp.pt for kp in framKeypoints]).reshape(-1, 1, 2)
                    srcPts = np.float32([kp.pt for kp in arKeypoints]).reshape(-1, 1, 2)
                    matrix, _ = cv.findHomography(srcPts, dstPts, cv.RANSAC)
                    pts = np.float32([[0, 0], [0, self.hT], [self.wT, self.hT], [self.wT, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, matrix)
                    imgWrap = cv.warpPerspective(self.ARIMAGE, matrix, (self.frame.shape[1], self.frame.shape[0]))
                    cv.imshow('imgWrap', imgWrap)
                    _, mask = cv.threshold(cv.cvtColor(imgWrap, cv.COLOR_BGR2GRAY),10, 1, cv.THRESH_BINARY_INV)     
                    #Erode and dilate are used to delete the noise
                    mask = cv.erode(mask,(3,3))
                    mask = cv.dilate(mask,(3,3))         
                    #The two images are added using the mask
                    for c in range(0,3):
                        self.frame[:, :, c] = imgWrap[:,:,c]*(1-mask[:,:]) + self.frame[:,:,c]*mask[:,:]

                    cv.imshow('frame', self.frame)
                key = cv.waitKey(1)
                if key == 27:
                    break

    def run(self):
        if self.SELECTORDINATE:
            self.getOrdinatesArImage()
        self.videoRecord()
        cv.destroyAllWindows()

        
if __name__ == "__main__":
    BASE_DIR = os.path.join(os.getcwd(), 'images')

    imageName = input('Please Enter the imageName: ')
    choice = input('Yes/No for select co_ordinate:')
    if choice in ['y', 'yes', 'Yes', 'YES']:
        selectOrdinate = True
    else:
        selectOrdinate = False

    print(selectOrdinate)
    a = Main(os.path.join(BASE_DIR, imageName), selectOrdinate=selectOrdinate)
    a.run()
