import os
import math
import numpy as np
import cv2

from global_Var import *

from config import *
from scripts.butil.seq_config import *

def getSubwindow(im, pos, model_sz, scaled_sz):
    # solve the nest ndarray - because of using matlab.engine
    if len(pos) == 1:
        pos = pos[0]
        model_sz = model_sz[0]
        scaled_sz = scaled_sz[0]

    sz = scaled_sz
    sz[sz < 2] = 2

    xs = np.round(pos[1] + np.asarray( range(0,int(sz[1])) ) - sz[1] / 2)
    ys = np.round(pos[0] + np.asarray( range(0,int(sz[0])) ) - sz[0] / 2)

    xs[xs < 1] = 1
    ys[ys < 1] = 1
    xs[xs > im.shape[1]] = im.shape[1]
    ys[ys > im.shape[0]] = im.shape[0]

    xs_int = xs.astype(int)
    ys_int = ys.astype(int)

    im_patch_original = np.floor(im[ys_int[0]:ys_int[-1]+1, xs_int[0]:xs_int[-1]+1, :])

    im_patch = cv2.resize(im_patch_original, tuple(model_sz.astype(int).tolist()) )

    return im_patch

class staple_MatlabEngine():
    def __init__(self):
        # Matlab Engine and Path
        cwd_curr = os.getcwd()
        os.chdir(cwdPath)
        # print(cwdPath)
        print 'Starting matlab engine...'
        self.m = matlab.engine.start_matlab()
        self.m.addpath(self.m.genpath('./staple', nargout=1), nargout=0)
        self.m.addpath(self.m.genpath('./staple/staple_A3C', nargout=1), nargout=0)
        # print(m.genpath('./staple', nargout=1))
        os.chdir(cwd_curr)

    def run_Staple_Matlab(self):
        self.seq.init_rect = matlab.double(self.seq.init_rect)
        self.m.workspace['subS'] = self.seq.__dict__
        self.m.workspace['rp'] = os.path.abspath(self.rp)
        self.m.workspace['bSaveImage'] = self.bSaveImage
        func = 'run_Staple(subS, rp, bSaveImage);'
        ###################################################################### Debug1 : Reference to non-existent field 'videoPlayer'.
        res = self.m.eval(func, nargout=1)
        res['res'] = scripts.butil.matlab_double_to_py_float(res['res'])
        # m.quit()
        return res

    def run_Staple_Matlab_readOTBParam(self, seq, rp, bSaveImage):
        self.seq.init_rect = matlab.double(seq.init_rect)
        self.m.workspace['subS'] = seq.__dict__
        self.m.workspace['rp'] = os.path.abspath(rp)
        self.m.workspace['bSaveImage'] = bSaveImage
        func = 'read_OTBParam(subS, rp, bSaveImage);'
        self.params, self.im, self.bg_area, self.fg_area, self.area_resize_factor = self.m.eval(func, nargout=5)

    def run_Staple_Matlab_readVOTParam(self, seq_path, sequence, start_frame):
        self.m.workspace['seq_path'] = seq_path
        self.m.workspace['sequence'] = sequence
        self.m.workspace['start_frame'] = start_frame
        func = 'read_VOTParam(seq_path, sequence, start_frame);'
        self.params, self.im, self.bg_area, self.fg_area, self.area_resize_factor = self.m.eval(func, nargout=5)

    def run_Staple_Matlab_normParam(self):
        # Param list to numpy
        self.params['init_pos'] = np.asarray(self.params['init_pos'])
        self.params['target_sz'] = np.asarray(self.params['target_sz'])
        self.params['norm_bg_area'] = np.asarray(self.params['norm_bg_area'])
        self.params['cf_response_size'] = np.asarray(self.params['cf_response_size'])
        self.params['norm_target_sz'] = np.asarray(self.params['norm_target_sz'])
        self.params['norm_delta_area'] = np.asarray(self.params['norm_delta_area'])
        self.params['norm_pwp_search_area'] = np.asarray(self.params['norm_pwp_search_area'])

        self.im = np.asarray(self.im)
        self.bg_area = np.asarray(self.bg_area)
        self.fg_area = np.asarray(self.fg_area)

    def run_Staple_Matlab_init(self):
        self.frame = 0

        # OTB DATA
        self.num_frames = len(self.params['img_files'])
        self.pos = self.params['init_pos']
        self.target_sz = self.params['target_sz']

        self.m.workspace['im'] = matlab.double(self.im.tolist())
        self.m.workspace['pos'] = matlab.double(self.pos.tolist())
        self.m.workspace['norm_bg_area'] = matlab.double(self.params['norm_bg_area'].tolist())
        self.m.workspace['bg_area'] = matlab.double(self.bg_area.tolist())
        func  = 'getSubwindow(im, pos, norm_bg_area, bg_area);'
        patch_padded =  self.m.eval(func, nargout=1)

        self.new_pwp_model = True
        self.m.workspace['new_pwp_model'] = self.new_pwp_model
        self.m.workspace['patch_padded'] = patch_padded
        self.m.workspace['bg_area'] = matlab.double(self.bg_area.tolist())
        self.m.workspace['fg_area'] = matlab.double(self.fg_area.tolist())
        self.m.workspace['target_sz'] = matlab.double(self.target_sz.tolist())
        self.m.workspace['norm_bg_area'] = matlab.double(self.params['norm_bg_area'].tolist())
        self.m.workspace['n_bins'] = self.params['n_bins']
        self.m.workspace['grayscale_sequence'] = self.params['grayscale_sequence']
        func  = 'updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, norm_bg_area, n_bins, grayscale_sequence);'
        self.bg_hist, self.fg_hist =  self.m.eval(func, nargout=2)
        self.new_pwp_model = False

        # IF isToolboxAvailable('Signal Processing Toolbox')
        self.m.workspace['cf_response_size1'] = self.params['cf_response_size'].tolist()[0][0]
        self.m.workspace['cf_response_size2'] = self.params['cf_response_size'].tolist()[0][1]
        self.hann_window = self.m.eval("single(hann(cf_response_size1) * hann(cf_response_size2)')", nargout=1)

        # output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
        output_sigma = math.sqrt(self.params['norm_target_sz'].prod()) * self.params['output_sigma_factor'] / self.params['hog_cell_size']

        self.m.workspace['cf_response_size'] = matlab.double(self.params['cf_response_size'].tolist())
        self.m.workspace['output_sigma'] = output_sigma
        func = 'gaussianResponse(cf_response_size, output_sigma);'
        y = self.m.eval(func, nargout=1)
        y = np.asarray(y)

        self.yf = np.fft.fft2(y)

        if self.params['scale_adaptation'] == True :
            self.scale_factor = 1
            self.base_target_sz = self.target_sz
            scale_sigma = math.sqrt(self.params['num_scales']) * self.params['scale_sigma_factor']

            # (1:p.num_scales) - ceil(p.num_scales/2);
            ss = np.asarray( range(1, int(self.params['num_scales']+1)) ) - math.ceil(self.params['num_scales']/2)
            ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)

            self.m.workspace['ys'] = matlab.double(ys.tolist())
            func = 'single(fft(ys));'
            self.ysf = self.m.eval(func, nargout=1)

            self.m.workspace['num_scales'] = self.params['num_scales']
            if self.params['num_scales'] % 2 == 0 :
                func = 'single(hann(num_scales+1));'
                self.scale_window = self.m.eval(func, nargout=1)
                self.scale_window = scale_window[2:end]
            else :
                func = 'single(hann(num_scales));'
                self.scale_window = self.m.eval(func, nargout=1)

            # scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
            self.scale_factors = self.params['scale_step'] ** (math.ceil(self.params['num_scales'] / 2) - np.asarray(range(1, int(self.params['num_scales']) + 1)))

            if self.params['scale_model_factor'] ** 2 * self.params['norm_target_sz'].prod() > self.params['scale_model_max_area']:
                self.params['scale_model_factor'] = math.sqrt(self.params['scale_model_max_area'] / self.params['norm_target_sz'].prod())


            self.scale_model_sz = np.floor(self.params['norm_target_sz'] * self.params['scale_model_factor'])
            # find maximum and minimum scales
            self.min_scale_factor = self.params['scale_step'] ** np.ceil(np.log(np.max(5 / self.bg_area)) / math.log(self.params['scale_step']))
            self.max_scale_factor = self.params['scale_step'] ** np.floor(np.log(np.min([self.im.shape[0], self.im.shape[1]] / self.target_sz)) / math.log(self.params['scale_step']))

        self.hf_den_List = []
        self.hf_num_List = []

        return "Test Finish"

    def run_Staple_Matlab_tracking(self):
        self.im = cv2.imread(self.params['img_path'] + self.params['img_files'][self.frame])

        self.m.workspace['im'] = matlab.double(self.im.tolist())
        self.m.workspace['pos'] = matlab.double(self.pos.tolist())
        self.m.workspace['norm_bg_area'] = matlab.double(self.params['norm_bg_area'].tolist())
        self.m.workspace['bg_area'] = matlab.double(self.bg_area.tolist())
        func  = 'getSubwindow(im, pos, norm_bg_area, bg_area);'
        im_patch_cf =  self.m.eval(func, nargout=1)

        pwp_search_area = np.round(self.params['norm_pwp_search_area'] / self.area_resize_factor)

        self.m.workspace['im'] = matlab.double(self.im.tolist())
        self.m.workspace['pos'] = matlab.double(self.pos.tolist())
        self.m.workspace['norm_pwp_search_area'] = matlab.double(self.params['norm_pwp_search_area'].tolist())
        self.m.workspace['pwp_search_area'] = matlab.double(pwp_search_area.tolist())
        func  = 'getSubwindow(im, pos, norm_pwp_search_area, pwp_search_area);'
        im_patch_pwp =  self.m.eval(func, nargout=1)

        # compute feature map, of cf_response_size
        self.m.workspace['im_patch_cf'] =  im_patch_cf
        self.m.workspace['feature_type'] = self.params['feature_type']
        self.m.workspace['cf_response_size'] = matlab.double(self.params['cf_response_size'].tolist())
        self.m.workspace['hog_cell_size'] = self.params['hog_cell_size']
        func  = 'getFeatureMap(im_patch_cf, feature_type, cf_response_size, hog_cell_size);'
        xt = self.m.eval(func, nargout=1)

        self.m.workspace['cf_response_size'] = self.hann_window
        self.m.workspace['xt'] = xt
        func  = 'bsxfun(@times, hann_window, xt);'
        xt_windowed = self.m.eval(func, nargout=1)

        self.m.workspace['xt_windowed'] = xt_windowed
        func  = 'fft2(xt_windowed);'
        xtf = self.m.eval(func, nargout=1)


        ################ NetWork Here !!!
        response_cf_List = []
        for i in range(0, len(self.hf_den_List)):
            if self.params['den_per_channel'] == True:
                hf = self.hf_num_List[i] / (self.hf_den_List[i] + self.params['lambda'])
            else:
                self.m.workspace['hf_den_List'] = self.hf_den_List[i]
                self.m.workspace['hf_num_List'] = self.hf_num_List[i]
                self.m.workspace['lambda'] = self.params['lambda']
                func = 'bsxfun(@rdivide, hf_num_List, sum(hf_den_List, 3)+lambda );'
                hf = self.m.eval(func, nargout=1)

            self.m.workspace['hf'] = hf
            self.m.workspace['xtf'] = xtf
            func = 'ensure_real(ifft2(sum(conj(hf).* xtf, 3)));'
            response_cf = self.m.eval(func, nargout=1)

            self.m.workspace['response_cf'] = response_cf
            self.m.workspace['norm_delta_area'] = matlab.double(self.params['norm_bg_area'].tolist())
            self.m.workspace['hog_cell_size'] = self.params['hog_cell_size']
            func = 'cropFilterResponse(response_cf, floor_odd(norm_delta_area / hog_cell_size));'
            response_cf = self.m.eval(func, nargout=1)


            if self.params['hog_cell_size'] > 1:
                self.m.workspace['response_cf'] = matlab.single(response_cf)
                self.m.workspace['norm_delta_area'] = matlab.double(self.params['norm_delta_area'].tolist())
                func = 'mexResize(response_cf, norm_delta_area, \'auto\');'
                response_cf = self.m.eval(func, nargout=1)

            ################ NetWork Input !!!
            response_cf_List.append(response_cf)

            self.m.workspace['im_patch_pwp'] = im_patch_pwp
            self.m.workspace['bg_hist'] = self.bg_hist
            self.m.workspace['fg_hist'] = self.fg_hist
            self.m.workspace['n_bins'] = self.params['n_bins']
            self.m.workspace['grayscale_sequence'] = self.params['grayscale_sequence']
            func = 'getColourMap(im_patch_pwp, bg_hist, fg_hist, n_bins, grayscale_sequence);'
            likelihood_map = self.m.eval(func, nargout=1)
            # (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)


            likelihood_map = np.asarray(likelihood_map)
            likelihood_map[np.isnan(likelihood_map)] = 0

            self.m.workspace['likelihood_map'] = matlab.double(likelihood_map.tolist())
            self.m.workspace['norm_target_sz'] = matlab.double(self.params['norm_target_sz'].tolist())
            func = 'getCenterLikelihood(likelihood_map, norm_target_sz);'
            response_pwp = self.m.eval(func, nargout=1)

            self.m.workspace['response_cf'] = response_cf
            self.m.workspace['response_pwp'] = response_pwp
            self.m.workspace['merge_factor'] = self.params['merge_factor']
            self.m.workspace['merge_method'] = self.params['merge_method']
            func = 'mergeResponses(response_cf, response_pwp, merge_factor, merge_method);'
            response = self.m.eval(func, nargout=1)

            index_tmp = response.index(max(response))
            row = index_tmp[0]
            col = index_tmp[1]
            center = (1+self.params['norm_delta_area']) / 2
            self.pos = self.pos + (np.asarray([row, col]) - center) / self.area_resize_factor
            rect_position = [self.pos[::-1] - self.target_sz[::-1]/2, self.target_sz[::-1]]

            if self.params['scale_adaptation'] == True:
                self.m.workspace['im'] = matlab.double(self.im.tolist())
                self.m.workspace['pos'] = matlab.double(self.pos.tolist())
                self.m.workspace['base_target_sz'] = matlab.double(self.base_target_sz.tolist())
                self.m.workspace['scale_factor_scale_factors'] = matlab.double(
                    self.scale_factor * self.scale_factors.tolist())
                self.m.workspace['scale_window'] = self.scale_window
                self.m.workspace['scale_model_sz'] = matlab.double(self.scale_model_sz.tolist())
                self.m.workspace['hog_scale_cell_size'] = self.params['hog_scale_cell_size']
                func = 'getScaleSubwindow(im, pos, base_target_sz, scale_factor_scale_factors, scale_window, scale_model_sz, hog_scale_cell_size);'
                im_patch_scale = self.m.eval(func, nargout=1)

                self.m.workspace['im_patch_scale'] = im_patch_scale
                func = 'fft(im_patch_scale, [], 2);'
                xsf = self.m.eval(func, nargout=1)

                self.m.workspace['sf_num'] = self.sf_num
                self.m.workspace['xsf'] = xsf
                self.m.workspace['sf_den'] = self.sf_den
                self.m.workspace['lambda'] = self.params['lambda']
                func = 'real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda) ));'
                scale_response = self.m.eval(func, nargout=1)

                self.m.workspace['scale_response'] = scale_response
                func = 'ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));'
                recovered_scale = self.m.eval(func, nargout=1)

                self.m.workspace['scale_factor'] = self.scale_response
                self.m.workspace['scale_factors'] = self.scale_factors
                self.m.workspace['recovered_scale'] = recovered_scale
                func = 'scale_factor * scale_factors(recovered_scale);'
                self.scale_factor = self.m.eval(func, nargout=1)

                if self.scale_factor < self.min_scale_factor :
                    self.scale_factor = self.min_scale_factor
                elif self.cale_factor > self.max_scale_factor:
                    self.scale_factor = self.max_scale_factor

                self.target_sz = np.round(self.base_target_sz * self.scale_factor)
                avg_dim = np.sum(self.target_sz)/2
                self.bg_area = np.round(self.target_sz + avg_dim)

                if self.bg_area[1] > self.im.shape[1]:
                    self.bg_area[1] = self.im.shape[1]-1
                if self.bg_area[0] > self.im.shape[0]:
                    self.bg_area[0] = self.im.shape[0]-1

                self.bg_area = self.bg_area - (self.bg_area - self.target_sz) % 2
                self.fg_area = np.round(self.target_sz - avg_dim * self.params['inner_padding'])
                self.fg_area = self.fg_area + (self.bg_area - self.target_sz) % 2

                self.area_resize_factor = np.sqrt(self.params['fixed_areaself'] / self.bg_area.prod())

        return "Finish"

    def run_Staple_Matlab_updating(self):
        self.m.workspace['im'] = matlab.double(self.im.tolist())
        self.m.workspace['pos'] = matlab.double(self.pos.tolist())
        self.m.workspace['norm_bg_area'] = matlab.double(self.params['norm_bg_area'].tolist())
        self.m.workspace['bg_area'] = matlab.double(self.bg_area.tolist())
        func  = 'getSubwindow(im, pos, norm_bg_area, bg_area);'
        im_patch_bg = self.m.eval(func, nargout=1)
        im_patch_bg = np.asarray(im_patch_bg)
        im_patch_bg = np.round(im_patch_bg)
        im_patch_bg = im_patch_bg.astype(int)
        #im_patch_bg = getSubwindow(self.im, self.pos, self.params['norm_bg_area'], self.bg_area)

        # compute feature map, of cf_response_size
        self.m.workspace['im_patch_bg'] =  matlab.uint8(im_patch_bg.tolist())
        self.m.workspace['feature_type'] = self.params['feature_type']
        self.m.workspace['cf_response_size'] = matlab.double(self.params['cf_response_size'].tolist())
        self.m.workspace['hog_cell_size'] = self.params['hog_cell_size']
        func  = 'getFeatureMap(im_patch_bg, feature_type, cf_response_size, hog_cell_size);'
        xt = self.m.eval(func, nargout=1)

        # apply Hann window
        self.m.workspace['hann_window'] = self.hann_window
        self.m.workspace['xt'] = xt
        func  = 'bsxfun(@times, hann_window, xt);'
        xt = self.m.eval(func, nargout=1)
        # TODO DEBUG
        # compute FFT

        self.m.workspace['xt'] = xt
        func  = 'fft2(xt);'
        xtf = self.m.eval(func, nargout=1)

        self.m.workspace['yf'] = matlab.single(self.yf.tolist(), None, True)
        self.m.workspace['xtf'] = xtf
        self.m.workspace['cf_response_size'] = matlab.double(self.params['cf_response_size'].tolist())
        func = 'bsxfun(@times, conj(yf), xtf) / prod(cf_response_size);'
        new_hf_num = self.m.eval(func, nargout=1)
        func = '(conj(xtf).* xtf) / prod(cf_response_size);'
        new_hf_den = self.m.eval(func, nargout=1)

        if self.frame == 0:
            self.hf_den_List.append(new_hf_num)
            self.hf_num_List.append(new_hf_den)
        else:
            for i in len(hf_den_List):
                self.hf_den_List[i] = (1 - self.params['learning_rate_cf']) * self.hf_den_List[i] + self.params['learning_rate_cf'] * new_hf_num
                self.hf_num_List[i] = (1 - self.params['learning_rate_cf']) * self.hf_num_List[i] + self.params['learning_rate_cf'] * new_hf_den

            self.m.workspace['new_pwp_model'] = self.new_pwp_model
            self.m.workspace['patch_padded'] = patch_padded
            self.m.workspace['bg_area'] = matlab.double(self.bg_area.tolist())
            self.m.workspace['fg_area'] = matlab.double(self.fg_area.tolist())
            self.m.workspace['target_sz'] = matlab.double(self.target_sz.tolist())
            self.m.workspace['norm_bg_area'] = matlab.double(self.params['norm_bg_area'].tolist())
            self.m.workspace['n_bins'] = self.params['n_bins']
            self.m.workspace['grayscale_sequence'] = self.params['grayscale_sequence']
            func  = 'updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, norm_bg_area, n_bins, grayscale_sequence);'
            self.bg_hist, self.fg_hist =  self.m.eval(func, nargout=2)

            if frame% 50 == 0:
                self.hf_den_List.append(new_hf_num)
                self.hf_num_List.append(new_hf_den)

        if self.params['scale_adaptation'] == True:
            self.m.workspace['im'] = matlab.double(self.im.tolist())
            self.m.workspace['pos'] = matlab.double(self.pos.tolist())
            self.m.workspace['base_target_sz'] = matlab.double(self.base_target_sz.tolist())
            self.m.workspace['scale_factor_scale_factors'] = matlab.double(self.scale_factor * self.scale_factors.tolist())
            self.m.workspace['scale_window'] = self.scale_window
            self.m.workspace['scale_model_sz'] = matlab.double(self.scale_model_sz.tolist())
            self.m.workspace['hog_scale_cell_size'] = self.params['hog_scale_cell_size']
            func = 'getScaleSubwindow(im, pos, base_target_sz, scale_factor_scale_factors, scale_window, scale_model_sz, hog_scale_cell_size);'
            im_patch_scale = self.m.eval(func, nargout=1)

            self.m.workspace['im_patch_scale'] = im_patch_scale
            func = 'fft(im_patch_scale, [], 2);'
            xsf = self.m.eval(func, nargout=1)

            self.m.workspace['xsf'] = xsf
            self.m.workspace['ysf'] = self.ysf
            func = 'bsxfun(@times, ysf, conj(xsf));'
            new_sf_num = self.m.eval(func, nargout=1)

            self.m.workspace['xsf'] = xsf
            func = 'sum(xsf .* conj(xsf), 1);'
            new_sf_den = self.m.eval(func, nargout=1)

            if self.frame == 0:
                self.sf_den = new_sf_den
                self.sf_num = new_sf_num
            else:
                self.sf_den = (1 - self.params['learning_rate_cf']) * self.sf_den + self.params['learning_rate_cf'] * self.sf_den
                self.sf_num = (1 - self.params['learning_rate_cf']) * self.sf_num + self.params['learning_rate_cf'] * self.sf_num


        #if self.frame == 0:
            #rect_position = [self.pos[::-1] - self.target_sz[::-1] / 2, self.target_sz[::-1]]
        #rect_position_padded = [self.pos[::-1] - self.bg_area[::-1] / 2, self.bg_area[::-1]]