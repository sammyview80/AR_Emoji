import cv2 as cv
import numpy as np
import random

def show_keypoints(image, dict_results, keypoint=True, labels=False):
    """
    image: image array ,
    dict_result : [
                        {
                            'box': [277, 90, 48, 63],
                            'keypoints':
                            {
                                'left_eye': (303, 131),
                                'right_eye': (313, 141),
                                'nose': (314, 114),
                                'mouth_left': (291, 117),
                                'mouth_right': (296, 143)
                            },
                            'confidence': 0.99851983785629272
                        }
                    ]
    returns:
    image with lables, points as list of keypoints objects if keypoint = True, otherwise, return list of keypoints.
    """
    try:
        keypoints = dict_results[0]['keypoints']
        points = []
        for key, value in keypoints.items():
            if labels:
                image = cv.circle(image, value, 3, (0, 255, 0), -1)
                bend_point = (value[0]+20, value[1]-20)
                putText_origin = (bend_point[0]+20, bend_point[1])
                image = cv.line(image, value, bend_point, (0,0,0), 2)
                image = cv.line(image, bend_point, putText_origin, (0,0,0), 2)
                image = cv.putText(image, key, putText_origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            if keypoint:
                keyPointObj = cv.KeyPoint(value[0], value[1], 3)
                points.append(keyPointObj)
            else:
                points.append(value)
        return image, points
    except IndexError:
        pass
def getKeypointsFromDict(result):
    try:
        keypoints = result[0]['keypoints']
        points = []
        for key, values in keypoints.items():
            keyPointObj = cv.KeyPoint(values[0], values[1], 3)
            points.append(keyPointObj)
        return points
    except IndexError:
        pass

def draw_matches(img1Wresult, img2Wresult, stackImg, hStack):
    """
    img1Wresult: tuple(image1, result1OfMTCNN),

    img2Wresult: tuple(image2, result2OfMTCNN),
    resultofMTCNN : [
                        {
                            'box': [277, 90, 48, 63],
                            'keypoints':
                            {
                                'left_eye': (303, 131),
                                'right_eye': (313, 141),
                                'nose': (314, 114),
                                'mouth_left': (291, 117),
                                'mouth_right': (296, 143)
                            },
                            'confidence': 0.99851983785629272
                        }
                    ]
    stackImg: stackHorizontally Image.
    hStack: shape of image.
    returns:
    image with lables, points as list of keypoints objects if keypoint = True, otherwise, return list of keypoints.
    """
    try:
        keypoints1 = img1Wresult[1][0]['keypoints']
        keypoints2 = img2Wresult[1][0]['keypoints']

        colors = [(0, 0, 0), (0, 255, 255), (0, 1, 200), (255, 55, 255), (0, 100, 100), (0, 255, 255), (1, 244, 200)]


        for (key1, value1) in keypoints1.items():
            value1 = (value1[0]+hStack, value1[1])
            cv.circle(stackImg, value1, 3, (0, 255, 0), -1)
            for (key2, value2) in keypoints2.items():
                cv.circle(stackImg, value2, 3, (0, 255, 0), -1)
                if key1 == key2:
                    color = colors[np.random.choice(range(0, len(colors)), 1)[0]]
                    cv.line(stackImg, value1, value2, color, 2)

    except IndexError:  
        pass

def getKeypointsObj(_list):
    kp = []
    for i in _list:
        keypointObj = cv.KeyPoint(i[0], i[1], 3)
        kp.append(keypointObj)
    return kp

def keypoint2list(keypoint):
    kp = []
    for kp in keypoint:
        (x, y) = kp.pt
        kp.append(x, y)
    return kp

def getListFromArray(np_array):
    kp = []
    points = np_array.squeeze()
    for point in points:
        kp.append(tuple(point))
    return kp

def getCenter(ARimage):
    center = ARimage.shape[1] /2
    return center, center