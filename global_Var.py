import os

proPath = "D:\\Program\\Tracking\\Project\\"
cwdPath = proPath + "tracking_Staple_cvpr16\\"
otbPath = proPath + "tracker_benchmark_OTB_Python\\"
libPath = proPath + "tracking_Lib\\"
############################################################ OTB
# dataIO
dataPath = "../../Data/OTB_Python/"
# dataIO OTB
gtFileName = "groundtruth_rect.txt"

LOADSEQS_OTB = "TB50"
RESULTPATH = "./"
SAVE_IMAGE = False
EVALTYPE = "OPE"
