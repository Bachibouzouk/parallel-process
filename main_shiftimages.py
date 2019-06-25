import os
import glob
import numpy as np
import skued
import scipy.io as spio
from run_parallel_processes import start_parallel_analysis
import sys
sys.path.append(r'PATHTOFOLDERINWHICHYOUHAVE_fhi_fed_utils')
import fhi_fed_utils as utils

PATH_CFG = 'W://HEED//data//TemCam//2019//20190502//meas2//python_analysis//config1.cfg'
dict_path, dict_numerics = utils.read_cfg(PATH_CFG)

PATH = dict_path['path']

partial = dict_numerics['partial']
if partial == 1:
    FILE_LIST = glob.glob(PATH + '*.tif')[:dict_numerics['to_file']]
else:
    FILE_LIST = glob.glob(PATH + '*.tif')

NO_OF_FILES = len(FILE_LIST)

#Loads flatfield
PATH_FF =  dict_path['path_ff']
FFmat = spio.loadmat(PATH_FF)
FF = FFmat['FF']

#Loads initial peak positions
PATH_PEAKS = PATH + dict_path['path_peaks']
PPmat = spio.loadmat(PATH_PEAKS)
peakpos_all = np.transpose(PPmat['peakpos_all'])
no_peaks = np.shape(peakpos_all)[0]
center = PPmat['center']

#Loads laser background
PATH_BKG = dict_path['path_bkg']
LASER_BKG = np.array(skued.diffread(PATH + PATH_BKG), dtype = np.float64)


#Loads parameters
window_size_intensity = dict_numerics['window_size_intensity']
window_size_background_peaks= dict_numerics['window_size_background_peaks']
mask_size_zero_order = dict_numerics['mask_size_zero_order']
max_distance = dict_numerics['max_distance']

dummy = skued.diffread(FILE_LIST[1])

#Loads masks
path_mask = PATH + dict_path['path_mask']
mask_total = spio.loadmat(path_mask)['mask_total']
mask_zero_order = utils.mask_image(dummy.shape, [center], [mask_size_zero_order])*mask_total
masked_bragg = utils.mask_image(dummy.shape, peakpos_all, window_size_intensity*np.ones(no_peaks))*mask_zero_order
masked_total_counts = utils.mask_image(dummy.shape, [center], [max_distance], True)*mask_zero_order
masked_dyn_bg = utils.mask_image(dummy.shape, peakpos_all, window_size_background_peaks*np.ones(no_peaks))#*mask_zero_order

#Loads peak positions
PATH_PEAKPOS = dict_path['peak_pos']
#Checks if file exists
exists = os.path.isfile(PATH_PEAKPOS)

if dict_numerics['calculate_peak_evolution']:
    print('First need to generate peak position file...')
    PEAK_POS = utils.peakpos_evolution(FILE_LIST, mask_total, LASER_BKG, FF, peakpos_all, dict_numerics['lens_corr_repetitions'],dict_numerics['lens_corr_window_size'])
    np.savetxt(PATH_PEAKPOS,
               PEAK_POS, header='No peaks)')
    PEAK_POS = PEAK_POS.reshape((NO_OF_FILES,no_peaks, 2))
else:
    if exists:
        PEAK_POS = np.loadtxt(PATH_PEAKPOS)
        try:
            PEAK_POS = PEAK_POS.reshape((NO_OF_FILES,no_peaks, 2))
        except:
            print('Peak position file exists but size does not match number of files analyzed...')
            exit()

#Path for saving output file
PATH_OUTPUT = PATH + dict_path['path_output'] + '//output_mainA02_' + PATH_CFG.split('//')[-1].split('.')[0] + '.txt'

def dispatch_file_name(n_cpu, f_list, stack_file_name_func, queues):
    """Dispatch files names among the different queues waiting to be analysed."""
    for i in range(n_cpu):
        stack_file_name_func(f_list[i::n_cpu], queues[i])



def personal_analysis_func(filename, idx, mask_total_counts=masked_total_counts, laser_bkg = LASER_BKG, FF = FF, mask_total = mask_total, mask_dyn_bg = masked_dyn_bg, peak_pos = PEAK_POS, center = center, window_size_intensity = window_size_intensity):
    """Perform the analysis from the file name."""

    # don't forget to add some info about the file_name in res (i.e. its index in the file_list)
    idx_file = FILE_LIST.index(filename)

    image = np.array(skued.diffread(filename), dtype = np.int64)
    image = image*mask_total
    n = np.shape(image)[0]
    #Substract background and flatfield
    image = utils.remove_bgk(image, laser_bkg, FF)

    radius, intensity = utils.azimuthal_average(image, (center[1], center[0]), mask = np.invert(np.isnan(mask_dyn_bg)))
    distance_matrix = utils.centeredDistanceMatrix_centered(n, center[1] - 0.5*n, center[0] -0.5*n)
    bg_im = utils.rings_to_average(distance_matrix,intensity,len(intensity))


    image_bgs = image - bg_im

    Image_bgs = Image_BG - Bg_im;
    Image_mask = Image_bgs. * mask_corr_4;
    Image_mask(isnan(Image_mask)) = 0;
    Image_2_mask = Image_bgs - Image_mask;
    Image_2_mask(isnan(Image_2_mask)) = 0;

    % calculate
    min
    Image_ref = Image_1_mask;
    Image_sample = Image_2_mask;
    x0 = [x(1), x(2)];
    [x, fval, exitflag, output] = fminsearch( @ (x)
    Fun_diff(x, Image_ref, Image_sample), x0, optimset('TolX', 1e-8));
    % add
    image
    translate
    vector
    image_corr = imtranslate(Image_BG, [x(1), x(2)]);
    #Total counts
    totale = np.nansum(image*mask_total_counts)
    #Dynamic background
    radius, intensity = utils.azimuthal_average(image, (center[1], center[0]), mask = np.invert(np.isnan(mask_dyn_bg)))
    distance_matrix = utils.centeredDistanceMatrix_centered(n, center[1] - 0.5*n, center[0] -0.5*n)
    bg_im = utils.rings_to_average(distance_matrix,intensity,len(intensity))
    background_raw = np.nansum(bg_im)

    image_bgs = image - bg_im
    intensities_raw = []
    for idp, peak in enumerate(new_peakpos_all2):
        intensities_raw.append(utils.sum_peak_pixels(image_bgs, peak, window_size_intensity))

    list_output = [idx_file, totale, background_raw] + list(new_peakpos_all2[:,0]) + list(new_peakpos_all2[:,1]) + intensities_raw
    res = np.array(list_output)
    return res


def recombine_func(analysis_output, n):
    """Manage the outputs of the analysis performed on the file."""

    print('Analysed output ', n) #, ' ', analysis_output)
    with open(PATH_OUTPUT, 'a') as fp:
        for i in range(len(analysis_output)):
            if i == (len(analysis_output) -1):
                fp.write('{}\n'.format(analysis_output[-1]))
            else:
                fp.write('{}\t'.format(analysis_output[i]))
        
        #fp.write(str(analysis_output))


if __name__ == '__main__':
	start_parallel_analysis(
		FILE_LIST,
		task_split_func=dispatch_file_name,
		analyse_func=personal_analysis_func,
		recombine_func=recombine_func,
		num_cpu = 8
		)
