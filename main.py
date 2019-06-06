from run_parallel_processes import start_parallel_analysis


def dispatch_file_name(n_cpu, f_list, stack_file_name_func, queues):
    """Dispatch files names among the different queues waiting to be analysed."""
    for i in range(n_cpu):
        stack_file_name_func(f_list[i::n_cpu], queues[i])


def analysis_func(fname, idx):
    """Perform the analysis from the file name."""
    pass


def recombine_func(analysis_output):
    """Manage the outputs of the analysis performed on the file."""
    pass

file_list = []

start_parallel_analysis(
        file_list,
        task_split_func=dispatch_file_name,
        analyse_func=analysis_func,
        recombine_func=recombine_func,
        num_cpu=None
)
