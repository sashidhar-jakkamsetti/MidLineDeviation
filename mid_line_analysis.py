import cv2
import dlib
import numpy
import math
import sys
import matplotlib.pyplot as plt
import constants

predictor = dlib.shape_predictor(constants.PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(constants.cascade_path)

def measure_scale(vid_file):
    vidcap = cv2.VideoCapture(vid_file)
    success, image = vidcap.read()
    scale = 0
    count = 0
    while success and count < constants.SCALE_AVG_NFRAMES:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = 0
        flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
        flags |= cv2.CALIB_CB_FAST_CHECK
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (9,7), flags)
        if ret == True:
            dist = math.hypot(corners[1][0, 0] - corners[0][0, 0], corners[1][0, 1] - corners[0][0, 1])
            scale += constants.REAL_SQAURE_LEN / dist
            count += 1
        success, image = vidcap.read()
    return scale / constants.SCALE_AVG_NFRAMES

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=3, minSize=(50,50))
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

vid_file = sys.argv[1]
scale = measure_scale(vid_file)

vidcap = cv2.VideoCapture(vid_file)
success,image = vidcap.read()
count = 0
hor_dist = []
ver_dist = []
prev_verd = 0
prev_hord = 0
first = True
while success:
    try:
        landmarks = get_landmarks(image)
        verd = (landmarks[8][0, 1] - landmarks[30][0, 1])*scale
        hord = (landmarks[8][0, 0] - landmarks[30][0, 0])*scale
        if first:
            first = False
            ver_dist.append(verd)
            hor_dist.append(hord)
        else:
            if abs(prev_verd - verd) > constants.OPENING_DELTA_CHECK or abs(prev_hord - hord) > constants.DEVIATION_DELTA_CHECK:
                ver_dist.append(ver_dist[-1])
                hor_dist.append(hor_dist[-1])
            else:
                ver_dist.append(verd)
                hor_dist.append(hord)
        prev_hord = hord
        prev_verd = verd
        print("frame: {}, opening: {}, deviation: {} \t PROCESSED -> opening: {}, deviation: {}".format(count, verd, hord, ver_dist[-1], hor_dist[-1]))
    except:
        print('frame: {} failed to identify'.format(count))
        ver_dist.append(ver_dist[-1])
        hor_dist.append(hor_dist[-1])
    success,image = vidcap.read()    
    count += 1

ver_dist = [x - ver_dist[0] for x in ver_dist]

plt.figure(0)
plt.plot(hor_dist, 'C2', label='deviation (in mm)')
plt.plot(ver_dist, 'C1', label='opening (in mm)')
plt.legend()
plt.savefig("{}opening-deviation.{}.png".format(constants.output_figures_path, vid_file.split("/")[1]))

plt.figure(1)
plt.plot(hor_dist, 'C2', label='deviation (in mm)')
plt.legend()
plt.savefig("{}deviation.{}.png".format(constants.output_figures_path, vid_file.split("/")[1]))
plt.show()