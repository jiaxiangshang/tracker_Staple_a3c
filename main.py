import cv2
import sys

from global_Var import *

import sys
sys.path.append(otbPath)
sys.path.append(otbPath + "scripts\\butil")
sys.path.append(libPath + "DataIOLib")
print(sys.path)
from seq_config import *

from tracker_cf_Staple import *

videoName_list = butil.get_seq_names(LOADSEQS_OTB)
videoInfo_list = butil.load_seq_configs(videoName_list)

for idx in range(len(videoName_list)):
    subSeqs, subAnno = butil.get_sub_seqs(videoInfo_list[idx], 20.0, EVALTYPE)
    subS = subSeqs[0]

    videoPath = dataPath + videoName_list[idx] + "/img/"
    video = loadVideo_byName(videoName_list[idx])
    video_gt = loadVideoGt(videoName_list[idx])

    stapleMatlab = staple_MatlabEngine();
    tic = time.clock()
    res,heatmap = staple_MatlabEngine.run_Staple_MatlabFull(subS, RESULTPATH, SAVE_IMAGE)
    duration = time.clock() - tic
    print(videoName_list[idx], duration)

    for frame in range(0, len(video)):
        img = video[frame]
        rect = rectBase(res[frame], "Geom")
        cv2.rectangle(img, rect.GetLeftTop_Int_(), rect.GetRightBottom_Int_(), (255, 0, 0), 2)
        cv2.imshow('frame', img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
