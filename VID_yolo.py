#!/usr/bin/python
#-*-coding:utf-8 -*-
# Import the necessary packages

import numpy as np
import cv2
import os
import time
import progressbar
import pandas
import sys
import argparse
import Utils_Video
import tensorflow as tf

# Import DET Alg package
sys.path.insert(0, 'YOLO')
import YOLO_small_tf
from timer import Timer

# DET Alg Params

# yolo.disp_console = (True or False, default = True)
# yolo.imshow = (True or False, default = True)
# yolo.tofile_img = (output image filename)
# yolo.tofile_txt = (output txt filename)
# yolo.filewrite_img = (True or False, default = False)
# yolo.filewrite_txt = (True of False, default = False)
# yolo.detect_from_file(filename)
# yolo.detect_from_cvmat(cvmat)

########## SETTING PARAMETERS

def still_image_YOLO_DET(frames_list, frames_name, folder_path_det_frames,folder_path_det_result):
    print("Starting DET Phase")
    if not os.path.exists(folder_path_det_frames):
        os.makedirs(folder_path_det_frames)
        print("Created Folder: %s"%folder_path_det_frames)
    if not os.path.exists(folder_path_det_result):
        os.makedirs(folder_path_det_result)
        print("Created Folder: %s"%folder_path_det_result)
    yolo = YOLO_small_tf.YOLO_TF()
    det_frames_list=[]
    det_result_list=[]
    print("%d Frames to DET"%len(frames_list))
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    timer = Timer()

    for i in progress(range(0,len(frames_list))):
        # det_frame_name = frames_name[i]
        #print frames_name[i]
        timer.tic()
        det_frame_name = frames_name[i].replace('.jpg','_det.jpg')
        det_frame_name = folder_path_det_frames + det_frame_name
        det_frames_list.append(det_frame_name)
        
        det_result_name= frames_name[i].replace('.jpg','.txt')
        det_result_name = folder_path_det_result + det_result_name
        det_result_list.append(det_result_name)
        
        yolo.tofile_txt = det_result_name
        yolo.filewrite_txt = True
        yolo.disp_console = False
        yolo.filewrite_img = True
        yolo.tofile_img = det_frame_name
        yolo.detect_from_cvmat(frames_list[i][1])
        timer.toc()
    print ('Detection took {:.3f}s ').format(timer.total_time)
    return det_frames_list,det_result_list


def print_YOLO_DET_result(det_results_list,folder_path_summary_result, file_path_summary_result ):
    results_list=[]
    if not os.path.exists(folder_path_summary_result):
        os.makedirs(folder_path_summary_result)
        print("Created Folder: %s"%folder_path_summary_result)
    print("Starting Loading Results ")
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    names=['class_name', 'x1','y1','x2','y2','score']
    df = pandas.DataFrame(columns=names)
    a = len(df)
    mean=0.0
    with open(file_path_summary_result, "w") as out_file:
        for i in progress(range(0,len(det_results_list))):
        #df.append(pandas.read_csv(det_results_list[i], sep=',',names=names, encoding="utf8"))
        #results_list.append(pandas.read_csv(det_results_list[i], sep=',',names=names, encoding="utf8"))
            for line in open(det_results_list[i], "r"):
                df.loc[i] =tuple(line.strip().split(','))
                mean=mean+float(df.loc[i].score)
                out_file.write(str(tuple(line.strip().split(',')))+ os.linesep)
    print("Finished Loading Results ")
    print("Computing Final Mean Reasults..")
    print "Class: " + str(df.class_name.max())
    print "Max Value: " + str(df.score.max())
    print "Min Value: " + str(df.score.min())
    #print "Avg Value: " + str(mean/len(df))
    print "Avg Value: " + str(mean / len(det_results_list))
    return

######### MAIN ###############

def main():
    '''
    Parse command line arguments and execute the code

    '''
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--det_frames_folder', default='det_frames/', type=str)
    parser.add_argument('--det_result_folder', default='det_results/', type=str)
    parser.add_argument('--result_folder', default='summary_result/', type=str)
    parser.add_argument('--summary_file', default='results.txt', type=str)
    parser.add_argument('--output_name', default='output.mp4', type=str)
    parser.add_argument('--perc', default=100, type=int)
    #parser.add_argument('--path_video', dest='path_video', required=True, type=str)
    parser.add_argument('--path_video', dest='path_video', type=str)
    args = parser.parse_args()
    args.path_video = './DAVIS-2017-TrainVal.mp4'


    #with tf.device('/gpu:0'):
    frame_list, frames = Utils_Video.extract_frames(args.path_video, args.perc)
    det_frame_list,det_result_list=still_image_YOLO_DET(frame_list, frames, args.det_frames_folder,args.det_result_folder)
    Utils_Video.make_video_from_list(args.output_name, det_frame_list)
    print_YOLO_DET_result(det_result_list,args.result_folder, args.summary_file)

    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")


if __name__ == '__main__':
    main()



