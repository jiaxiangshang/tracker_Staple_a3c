import os

proPath = "D:\\Program\\Tracking\\Project\\"
cwdPath = proPath + "tracking_Staple_cvpr16\\"
otbPath = proPath + "tracker_benchmark_OTB_Python\\"
libPath = proPath + "tracking_Lib\\"

import sys
sys.path.append(otbPath)
sys.path.append(libPath)
print("global_Var", sys.path)

e_dataSet = "VOT"

############################################################ OTB
# dataIO
dataPath = "../../Data/OTB_Python/"
# dataIO OTB
gtFileName = "groundtruth_rect.txt"

LOADSEQS_OTB = "TB50"
RESULTPATH = "./"
SAVE_IMAGE = False
EVALTYPE = "OPE"

############################################################ OVOT
dataPathVOT = "../../Data/VOT_2015/";