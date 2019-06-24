import os
import glob
import numpy as np
import scipy
import skued
import scipy.io as spio
from scipy.interpolate import interp1d
from run_parallel_processes import start_parallel_analysis
import sys
sys.path.append(r'PATHTOFOLDERINWHICHYOUHAVE_fhi_fed_utils')
from fhi_fed_utils import utils

PATH = 'W://HEED//data//TemCam//2019//20190413//meas2//'
FILE_LIST = glob.glob(PATH + '*.tif')
print(FILE_LIST)
NO_OF_FILES = len(glob.glob(PATH + '*.tif'))

MAT_DEFAULTS=np.ones((3,2))
TRUC_DEFAULT="dsakjd"

#Loads flatfield
path_FF=  'Z://HEED//data//TemCam//general_BG-FF//20180801_flatfield.mat'
FFmat = spio.loadmat(path_FF)
FF = FFmat['FF']

#Loads initial peak positions
path_peaks = PATH + r'//eval_new_script//peak_positions.mat'
PPmat = spio.loadmat(path_peaks)
peakpos_all = np.transpose(PPmat['peakpos_all'])[:60]
no_peaks = np.shape(peakpos_all)[0]
center = PPmat['center']

#Loads laser background
LASER_BKG = np.array(skued.diffread(PATH + '//laser_background//meas2_0003.tif'), dtype = np.float64)

#Loads peak positions
PEAK_POS = np.loadtxt('W:\\HEED\\data\\TemCam\\2019\\20190413\\meas2\\eval_new_script\\peak_pos_evolution.txt')
PEAK_POS = PEAK_POS.reshape((NO_OF_FILES,no_peaks, 2))
print(np.shape(PEAK_POS))

#Loads parameters
window_size_intensity = 20
window_size_background_peaks=25
mask_size_zero_order = 80
max_distance = 950
dummy = skued.diffread(FILE_LIST[1])
#dummy_bkg = remove_bgk(dummy, laser_bkg, FF)

#Loads masks
path_mask = PATH + r'//eval_new_script//mask.mat'
mask_total = spio.loadmat(path_mask)['mask_total']
mask_zero_order = mask_image(dummy.shape, [center], [mask_size_zero_order])*mask_total
masked_bragg = mask_image(dummy.shape, peakpos_all, window_size_intensity*np.ones(no_peaks))*mask_zero_order
masked_total_counts = mask_image(dummy.shape, [center], [max_distance], True)*mask_zero_order
masked_dyn_bg = mask_image(dummy.shape, peakpos_all, window_size_background_peaks*np.ones(no_peaks))#*mask_zero_order



def dispatch_file_name(n_cpu, f_list, stack_file_name_func, queues):
    """Dispatch files names among the different queues waiting to be analysed."""
    for i in range(n_cpu):
        print(f_list[i::n_cpu])
        stack_file_name_func(f_list[i::n_cpu], queues[i])


distance_matrix = centeredDistanceMatrix(np.shape(dummy)[0])


def personal_analysis_func(filename, idx, mask_total_counts=masked_total_counts, laser_bkg = LASER_BKG, FF = FF, mask_total = mask_total, mask_dyn_bg = masked_dyn_bg, center = center, window_size_intensity = window_size_intensity, distance_matrix = distance_matrix, truc=TRUC_DEFAULT):
    """Perform the analysis from the file name."""

    # don't forget to add some info about the file_name in res (i.e. its index in the file_list)
    idx_file = FILE_LIST.index(filename)
    #print(idx*MAT_DEFAULTS)
    #print(truc)
    #print('Start analysis fname {} on process {}'.format(filename, idx))
    image = np.array(skued.diffread(filename), dtype = np.int64)
    image = image*mask_total
    n = np.shape(image)[0]
    #Substract background and flatfield
    image = remove_bgk(image, laser_bkg, FF)
    #peak_pos_fname = 'peakpos{}.npy'.format(idx)
    #if os.path.exists(peak_pos_fname):
    #     peakpos_all = np.load(peak_pos_fname, allow_pickle=True)
    #else:
    #    path_peaks = PATH + r'//eval_new_script//peak_positions.mat'
    #   PPmat = spio.loadmat(path_peaks)
    #   peakpos_all = np.transpose(PPmat['peakpos_all'])[:60]
    #   #peakpos_all = np.load('data/peakpos.npy', allow_pickle=True)[0]


    try:
         #new_peakpos_all = refine_peakpos_arb_dim(peakpos_all, image, 0, 8)
         new_peakpos_all2 = PEAK_POS[idx_file,:,:]
    except:
        res = np.array([1,2])

    #np.save(peak_pos_fname, new_peakpos_all)

    #print('Finish analysis fname {} on process {}'.format(filename, idx))
    
    #Total counts
    totale = np.nansum(image*mask_total_counts)
    #Dynamic background
    radius, intensity = azimuthal_average(image, (center[1], center[0]), mask = np.invert(np.isnan(mask_dyn_bg)))
    distance_matrix = centeredDistanceMatrix_centered(n, center[1] - 0.5*n, center[0] -0.5*n)
    bg_im = rings_to_average(distance_matrix,intensity,len(intensity))
    background_raw = np.nansum(bg_im)

    image_bgs = image - bg_im
    intensities_raw = []
    for idp, peak in enumerate(new_peakpos_all2):
        intensities_raw.append(sum_peak_pixels(image_bgs, peak, window_size_intensity))

    list_output = [idx_file, totale, background_raw] + list(new_peakpos_all2[:,0]) + list(new_peakpos_all2[:,1]) + intensities_raw
    res = np.array(list_output)
    return res


def recombine_func(analysis_output, n):
    """Manage the outputs of the analysis performed on the file."""

    print('Analysed output ', n) #, ' ', analysis_output)
    with open(PATH + '//eval_new_script//output.txt', 'a') as fp:
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
