import sys
import traceback
import logging
from multiprocessing import cpu_count, Pool
from pathlib import Path
import yaml

LOG = logging.getLogger()
logging.basicConfig()

CPU_COUNT = cpu_count()


def log_traceback():
    # Get current system exception
    ex_type, ex_value, ex_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append(
            "File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                trace[0], trace[1], trace[2], trace[3]))

    LOG.error(f"Exception type : {ex_type.__name__}")
    LOG.error(f"Exception message : {ex_value}")
    LOG.error(f"Stack trace : {stack_trace}")


def mprun(mp_func, inputs, n_workers=CPU_COUNT, ):
    pool = Pool(n_workers)

    results = []
    try:
        results = pool.map(mp_func, inputs)
    except:
        log_traceback()
    finally:
        pool.close()
        pool.join()

    return results


def read_yaml_file(file):
    with Path(file).open('r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)
    

def write_yaml_file(obj, file):
    with Path(file).open('w') as file:
        return yaml.dump(obj, file)