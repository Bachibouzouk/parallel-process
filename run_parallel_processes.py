import time
import multiprocessing as mp


def stack_file_name(file_names, queue_to_analyse):
    """Load file names in a queue to be analysed."""
    lstart = time.time()
    print('Loading..  {} files in the queue to be analysed'.format(len(file_names)))
    for fname in file_names:
        queue_to_analyse.put(fname)
    print(
        'Loaded {} files in the queue to be analysed {}s'.format(
            len(file_names),
            time.time() - lstart)
    )


def analyse_proc(queue_to_analyse, idx, queue_analyzed, analyse_func):
    """Execute the analyse_func on image files
        fetches the files from from the queue and put the result of the analysis
        on anothe queue
    """
    pr = mp.current_process()
    rstart = time.time()
    print(
        'Start {} from loop {}, queue empty={}'.format(pr.pid, idx, str(queue_to_analyse.empty())))
    while not queue_to_analyse.empty():
        # get filename from the queue
        fname = queue_to_analyse.get()
        # performs the analysis
        res = analyse_func(fname, idx)
        # put the result in another queue
        queue_analyzed.put(res)

    print('Finish {} from loop {} in {}s'.format(pr.pid, idx, time.time() - rstart))


def recombine_proc(queue_analyzed, recombine_func, n_entries):
    """Process responsible to recombine the data"""
    pr = mp.current_process()
    print(
        'Start {} from recombine process, queue empty={}'.format(
            pr.pid, str(queue_analyzed.empty())
        )
    )
    wstart = time.time()

    n_loop = 0
    while n_loop < n_entries:
        # get the next available result from the list
        res = queue_analyzed.get(timeout=10)
        # pass it to the function with assembles the outputs of the analyse_func together
        recombine_func(res, n_loop)
        n_loop = n_loop + 1
    print('{} writer done in {}s'.format(pr.pid, time.time() - wstart))


def start_parallel_analysis(
        file_list,
        task_split_func,
        analyse_func,
        recombine_func,
        analysis_proc=analyse_proc,
        recombine_proc=recombine_proc,
        num_cpu=None
):
    """
    Manages the parallelization of the analysis task performed by analyse_func
    :param file_list: user_defined
    :param task_split_func: user_defined
    :param analyse_func: user_defined
    :param recombine_func: user_defined
    :param analysis_p:
    :param recombine_p:
    :param num_cpu:
    :return:
    """
    if num_cpu is None:
        num_cpu = mp.cpu_count()

    queue_analyzed = mp.Queue()

    # Queues for loading entries to be analyzed
    to_analyse_queues = []
    # Processes which run analysis function on files from file_list
    analysis_ps = []
    for i in range(num_cpu):
        to_analyse_queue = mp.Queue()
        to_analyse_queues.append(to_analyse_queue)
        analysis_p = mp.Process(
            target=analysis_proc,
            args=(to_analyse_queue, i, queue_analyzed, analyse_func)
        )
        analysis_p.daemon = True
        analysis_ps.append(analysis_p)

    _start = time.time()

    # load the file names in the queues, waiting for the analyse_func to process them
    task_split_func(num_cpu, file_list, stack_file_name, to_analyse_queues)

    # start the analysis processes
    for i in range(num_cpu):
        analysis_ps[i].start()

    n_entries = len(file_list)
    # start the recombining process which manages the outputs from the analyse_func
    time.sleep(1)
    recombine_p = mp.Process(
        target=recombine_proc,
        args=(queue_analyzed, recombine_func, n_entries)
    )
    recombine_p.daemon = True

    time.sleep(2)
    recombine_p.start()
    recombine_p.join()

    print("Total process took {} seconds".format(time.time() - _start))
