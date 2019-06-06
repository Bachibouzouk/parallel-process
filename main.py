import os
import glob
import numpy as np
import scipy
import skued
from run_parallel_processes import start_parallel_analysis, analyse_proc_template


def dispatch_file_name(n_cpu, f_list, stack_file_name_func, queues, analysis_ps):
    """Dispatch files names among the different queues waiting to be analysed."""
    for i in range(n_cpu):
        stack_file_name_func(f_list[i::n_cpu], queues[i])
        analysis_ps[i].start()


def refine_peakpos_arb_dim(peakpos_all, image, numrefine, window_size):

    new_peakpos_all = np.empty_like(peakpos_all)

    for idx, peak in enumerate(peakpos_all):

        lbx = int(int(peak[1]) - window_size)
        ubx = int(int(peak[1]) + window_size)
        lby = int(int(peak[0]) - window_size)
        uby = int(int(peak[0]) + window_size)

        im_p = image[lby:uby, lbx:ubx]
        cy, cx = scipy.ndimage.measurements.center_of_mass(np.power(im_p, 2))

        new_peak = new_peakpos_all[idx, :]
        new_peak[1] = cx + peak[1] - window_size - 1
        new_peak[0] = cy + peak[0] - window_size - 1

        counter = 0
        while counter < numrefine:
            lbx = int(int(new_peak[1] - window_size))
            ubx = int(int(new_peak[1] + window_size))
            lby = int(int(new_peak[0] - window_size))
            uby = int(int(new_peak[0] + window_size))

            im_p = image[lby:uby, lbx:ubx]
            cy, cx = scipy.ndimage.measurements.center_of_mass(np.power(im_p, 2))

            counter = counter + 1

            new_peak[1] = cx + new_peak[1] - window_size - 1
            new_peak[0] = cy + new_peak[0] - window_size - 1

        new_peakpos_all[idx, 0] = new_peak[0]
        new_peakpos_all[idx, 1] = new_peak[1]

    return new_peakpos_all


def personal_analysis_func(filename, idx):
    """Perform the analysis from the file name."""

    #print('Start analysis fname {} on process {}'.format(filename, idx))
    d = skued.diffread(filename)

    peak_pos_fname = 'data/peakpos{}.npy'.format(idx)
    if os.path.exists(peak_pos_fname):
        peakpos_all = np.load(peak_pos_fname, allow_pickle=True)
    else:
        peakpos_all = np.load('data/peakpos.npy', allow_pickle=True)[0]

    res = np.array([np.nansum(d)])
    # try:
    #     res = refine_peakpos_arb_dim(peakpos_all, d, 5, 8)
    # except:
    #     print('fucking nans')
    #     res = np.array([1,2])

    np.save(peak_pos_fname, res)

    #print('Finish analysis fname {} on process {}'.format(filename, idx))
    return res


def personal_analysis_proc(queue_to_analyse, idx, queue_analyzed):
    analyse_proc_template(queue_to_analyse, idx, queue_analyzed, personal_analysis_func)


def recombine_func(analysis_output, n):
    """Manage the outputs of the analysis performed on the file."""
    print('Analysed output ',n,' ', analysis_output)
    with open('test.txt', 'a') as fp:
        fp.write('{}\n'.format(n))


file_list = glob.glob('data/*.tif')
print(file_list)

start_parallel_analysis(
    file_list,
    #task_split_func=dispatch_file_name,
    # analyse_func=personal_analysis_func,
    recombine_func=recombine_func,
    analysis_p=personal_analysis_proc,
    num_cpu=None
)
