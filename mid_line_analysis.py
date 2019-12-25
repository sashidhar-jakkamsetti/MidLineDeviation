import cv2
import dlib
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

PREDICTOR_PATH = "detector_architectures/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='detector_architectures/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=3, minSize=(50,50))
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

vidcap = cv2.VideoCapture('videos/JawDeviation.mp4')
success,image = vidcap.read()
count = 0
hor_dist = [0]
ver_dist = [0]
while success:
    try:
        landmarks = get_landmarks(image)
        verd = landmarks[30][0, 1] - landmarks[8][0, 1]
        hord = landmarks[30][0, 0] - landmarks[8][0, 0]
        ver_dist.append(verd)
        hor_dist.append(hord)
        print("frame: {}, opening: {}, deviation: {}".format(count, verd, hord))
    except:
        print('frame: {} failed to identify'.format(count))
        ver_dist.append(ver_dist[-1])
        hor_dist.append(hor_dist[-1])
    success,image = vidcap.read()    
    count += 1

# x = numpy.array([i for i in range(len(hor_dist))])
# y = numpy.array(hor_dist)
# f = interp1d(x, y, kind='quadratic')
# y_smooth = f(x)
plt.plot(hor_dist, 'C2', label='deviation', markevery=10)
#plt.plot(ver_dist, 'C1', label='opening')
plt.legend()
plt.show()