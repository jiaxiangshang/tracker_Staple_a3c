import os

from global_Var import *

import sys
sys.path.append(otbPath)
sys.path.append(otbPath + "scripts\\butil")
print(sys.path)
from config import *
from seq_config import *


def tracker_cf_Staple_RL(seq, rp, bSaveImage):
    result,heatmap = run_Staple_RL(seq, rp, bSaveImage)
    return result['res'],heatmap

def run_Staple(seq, rp, bSaveImage):
    cwd_curr = os.getcwd()
    os.chdir(cwdPath)
    print(cwdPath)
    global m
    if m == None:
        print 'Starting matlab engine...'
        m = matlab.engine.start_matlab()

    print(m.genpath('./staple', nargout=1))

    m.addpath(m.genpath('./staple', nargout=1), nargout=0)
    seq.init_rect = matlab.double(seq.init_rect)
    m.workspace['subS'] = seq.__dict__
    m.workspace['rp'] = os.path.abspath(rp)
    m.workspace['bSaveImage'] = bSaveImage
    func = 'run_Staple(subS, rp, bSaveImage);'
    ###################################################################### Debug1 : Reference to non-existent field 'videoPlayer'.
    res = m.eval(func, nargout=1)
    res['res'] = scripts.butil.matlab_double_to_py_float(res['res'])
    # m.quit()

    os.chdir(cwd_curr)
    return res

def run_Staple_RL(seq, rp, bSaveImage):
    cwd_curr = os.getcwd()
    os.chdir(cwdPath)

    global m
    if m == None:
        print 'Starting matlab engine...'
        m = matlab.engine.start_matlab()
    m.addpath(m.genpath('./staple', nargout=1), nargout=0)
    seq.init_rect = matlab.double(seq.init_rect)
    m.workspace['subS'] = seq.__dict__
    m.workspace['rp'] = os.path.abspath(rp)
    m.workspace['bSaveImage'] = bSaveImage
    func = 'ExpRL_run_Staple(subS, rp, bSaveImage);'
    ###################################################################### Debug1 : Reference to non-existent field 'videoPlayer'.
    ###################################################################### Change1 : Change Output'.
    res,heatmap = m.eval(func, nargout=2)
    res['res'] = scripts.butil.matlab_double_to_py_float(res['res'])
    # m.quit()

    os.chdir(cwd_curr)
    return res,heatmap