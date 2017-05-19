import cv2
import sys
import time

from global_Var import *

from scripts.butil.seq_config import *
from GeomLib.rectDefine import *

from tracker_cf_Staple import *

videoInfo_list = []

def Staple_Full_OTB():
    stapleMatlab = staple_MatlabEngine()
    for idx in range(len(videoInfo_list)):
        subSeqs, subAnno = get_sub_seqs(videoInfo_list[idx], 20.0, EVALTYPE)
        subS = subSeqs[0]

        tic = time.clock()
        stapleMatlab.run_Staple_Matlab(subS, RESULTPATH, SAVE_IMAGE)

        #res = stapleMatlab.run_Staple_MatlabFull(subS, RESULTPATH, SAVE_IMAGE)
        duration = time.clock() - tic
        print(videoName_list[idx], duration)

        for frame in range(0, len(subS.s_frames)):
            img = cv2.imread(subS.s_frames[frame])
            rect = rectBase(res['res'][frame], "Geom")
            cv2.rectangle(img, rect.GetLeftTop_Int_(), rect.GetRightBottom_Int_(), (255, 0, 0), 2)
            cv2.imshow('frame', img)
            cv2.waitKey(100)
        cv2.destroyAllWindows()

def Staple_Part_OTB():
    stapleMatlab = staple_MatlabEngine()
    for idx in range(len(videoInfo_list)):
        subSeqs, subAnno = get_sub_seqs(videoInfo_list[idx], 20.0, EVALTYPE)
        subS = subSeqs[0]

        tic = time.clock()

        stapleMatlab.run_Staple_Matlab_readOTBParam(subS, RESULTPATH, SAVE_IMAGE)
        stapleMatlab.run_Staple_Matlab_init()

        res = stapleMatlab.run_Staple_Matlab_tracking()

        for frame in range(0, len(subS.s_frames)):
            img = cv2.imread(subS.s_frames[frame])

            res = stapleMatlab.run_Staple_Matlab_tracking()

            rect = rectBase(res, "Geom")
            cv2.rectangle(img, rect.GetLeftTop_Int_(), rect.GetRightBottom_Int_(), (255, 0, 0), 2)
            cv2.imshow('frame', img)
            cv2.waitKey(100)

        duration = time.clock() - tic
        print(videoName_list[idx], duration)
        cv2.destroyAllWindows()

def Staple_Part_VOT():
    stapleMatlab = staple_MatlabEngine()
    for idx in range(len(videoInfo_list)):

        tic = time.clock()

        stapleMatlab.run_Staple_Matlab_readVOTParam(dataPathVOT, videoInfo_list[idx],1)
        stapleMatlab.run_Staple_Matlab_normParam()
        stapleMatlab.run_Staple_Matlab_init()

        stapleMatlab.run_Staple_Matlab_updating()

        for frame in range(0, len(stapleMatlab.params['img_files'])):
            img = cv2.imread(stapleMatlab.params['img_path'] + stapleMatlab.params['img_files'][frame])

            res = stapleMatlab.run_Staple_Matlab_tracking()

            rect = rectBase(res, "Geom")
            cv2.rectangle(img, rect.GetLeftTop_Int_(), rect.GetRightBottom_Int_(), (255, 0, 0), 2)
            cv2.imshow('frame', img)
            cv2.waitKey(100)

        duration = time.clock() - tic
        print(videoName_list[idx], duration)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if e_dataSet == "OTB":
        videoName_list = get_seq_names(LOADSEQS_OTB)
        videoInfo_list = load_seq_configs(videoName_list)
    elif e_dataSet == "VOT":
        fid_seq_list = open(dataPathVOT + "list.txt")
        line = fid_seq_list.readline()
        while line:
            videoInfo_list.append(line)
            line = fid_seq_list.readline()

    Staple_Part_VOT()
