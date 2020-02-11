import cv2
import dlib
import numpy
import math
import sys
import os
import matplotlib.pyplot as plt
import constants

predictor = dlib.shape_predictor(constants.predictor_path)
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
        ret, corners = cv2.findChessboardCorners(gray, constants.CHESSBOARD_NCORNERS, flags)
        if ret == True:
            dist = math.hypot(corners[1][0, 0] - corners[0][0, 0], corners[1][0, 1] - corners[0][0, 1])
            scale += constants.REAL_SQAURE_LEN / dist
            count += 1
        success, image = vidcap.read()
    return scale / constants.SCALE_AVG_NFRAMES

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=3, minSize=(50,50))
    x,y,w,h = rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()]), rects[0]

def get_measurements(landmarks, facebox, scale, out_file):
    ver1 = (landmarks[36][0, 0] - facebox[0])*scale
    ver2 = (landmarks[39][0, 0] - landmarks[36][0, 0])*scale
    ver3 = (landmarks[42][0, 0] - landmarks[39][0, 0])*scale
    ver4 = (landmarks[45][0, 0] - landmarks[42][0, 0])*scale
    ver5 = (facebox[0] + facebox[2] - landmarks[45][0, 0])*scale
    hor1 = (int((landmarks[19][0, 1] + landmarks[24][0, 1])/2) - facebox[1])*scale
    hor2 = (landmarks[33][0, 1] - int((landmarks[19][0, 1] + landmarks[24][0, 1])/2))*scale
    hor3 = (int((landmarks[50][0, 1] + landmarks[52][0, 1])/2) - landmarks[33][0, 1])*scale
    hor4 = (facebox[1] + facebox[3] - int((landmarks[50][0, 1] + landmarks[52][0, 1])/2))*scale
    print_str = "vertical segments: {} \t {} \t {} \t {} \t {}\nhorizontal segments: {} \t {} \t {} \t {}"\
                    .format(ver1, ver2, ver3, ver4, ver5, hor1, hor2, hor3, hor4)
    out_file.write(print_str + "\n")
    if constants.verbose: print(print_str)


def run(filename):
    scale = measure_scale(filename)
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    hor_dist = []
    ver_dist = []
    prev_verd = 0
    prev_hord = 0
    first = True
    vid_file_name = filename.replace('\\','/').split('/')[-1]
    out_logfile = open("{}{}.log.txt".format(constants.output_log_path, vid_file_name), "w")
    out_measurefile = open("{}{}.measurements.txt".format(constants.output_measurements_path, vid_file_name), "w")

    while success:
        try:
            landmarks, facebox = get_landmarks(image)
            verd = (landmarks[8][0, 1] - landmarks[30][0, 1])*scale
            hord = (landmarks[8][0, 0] - landmarks[30][0, 0])*scale
            if first:
                first = False
                ver_dist.append(verd)
                hor_dist.append(hord)
                get_measurements(landmarks, facebox, scale, out_measurefile)
            else:
                if abs(prev_verd - verd) > constants.OPENING_DELTA_CHECK or abs(prev_hord - hord) > constants.DEVIATION_DELTA_CHECK:
                    ver_dist.append(ver_dist[-1])
                    hor_dist.append(hor_dist[-1])
                else:
                    ver_dist.append(verd)
                    hor_dist.append(hord)
            prev_hord = hord
            prev_verd = verd
            print_str = "frame: {}, opening: {}, deviation: {} \t PROCESSED -> opening: {}, deviation: {}".format(count, verd, hord, ver_dist[-1], hor_dist[-1])
            out_logfile.write(print_str + "\n")
            if constants.verbose: print(print_str)
        except:
            print_str = "frame: {} failed to identify".format(count)
            out_logfile.write(print_str + "\n")
            if constants.verbose: print(print_str)
            ver_dist.append(ver_dist[-1])
            hor_dist.append(hor_dist[-1])
        success,image = vidcap.read()    
        count += 1

    out_logfile.close()
    out_measurefile.close()
    ver_dist = [x - ver_dist[0] for x in ver_dist]
    plt.figure()
    plt.plot(hor_dist, 'C2', label='deviation (in mm)')
    plt.plot(ver_dist, 'C1', label='opening (in mm)')
    plt.legend()
    plt.savefig("{}opening-deviation.{}.png".format(constants.output_figures_path, vid_file_name))
    plt.figure()
    plt.plot(hor_dist, 'C2', label='deviation (in mm)')
    plt.legend()
    plt.savefig("{}deviation.{}.png".format(constants.output_figures_path, vid_file_name))


vid_file = sys.argv[1]
if vid_file.endswith('/') or vid_file.endswith('\\'):
    for filename in os.listdir(vid_file):
        print(vid_file + filename)
        run(vid_file + filename)
else:
    run(vid_file)
    plt.show()