import cv2
import dlib
import numpy
import constants
import sys

predictor = dlib.shape_predictor(constants.predictor_path)
cascade = cv2.CascadeClassifier(constants.cascade_path)

def annotate_landmarks(im):
    im = im.copy()
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=2)
        cv2.circle(im, pos, 3, color=(0, 0, 0), thickness=3)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0, 0, 255),5) 
    return im

infn = sys.argv[1]
im = cv2.imread(infn)
flim = annotate_landmarks(im)
#cv2.imshow('Result',flim)
cv2.imwrite(infn.replace(".JPG", "") + ".faciallandmarks.jpg", flim)
cv2.waitKey(0)
cv2.destroyAllWindows()