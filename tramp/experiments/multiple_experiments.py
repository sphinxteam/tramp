import pandas as pd
import numpy as np
import itertools
import logging
logger = logging.getLogger(__name__)


def log_on_progress(i, total):
    logger.info(f"experiment {i}/{total}")


def as_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return list(x)
    else:
        return [x]


def get_experiments_from_kwargs(**kwargs):
    kwargs_coerced = {key: as_list(val) for key, val in kwargs.items()}
    experiments = [
        {key: value for key, value in zip(kwargs_coerced.keys(), record_values)}
        for record_values in itertools.product(*kwargs_coerced.values())
    ]
    return experiments


def run_experiments(run, on_progress=None, **kwargs):
    on_progress = on_progress or log_on_progress
    experiments = get_experiments_from_kwargs(**kwargs)
    n_experiments = len(experiments)
    records = []
    # iterate over experiments
    for idx, experiment in enumerate(experiments):
        record = experiment.copy()
        try:
            results = run(**experiment)
            if isinstance(results, dict):
                results = [results]
            for result in results:
                result.update(record)
            records += results
        except Exception as e:
            logger.error(f"Experiment {experiment} failed\n{e}")
        on_progress(idx + 1, n_experiments)
    df = pd.DataFrame(records)
    return df


def simple_run_experiments(run, **kwargs):
    "Same as run_experiments but raises error and no `on_progress` callback"
    experiments = get_experiments_from_kwargs(**kwargs)
    n_experiments = len(experiments)
    records = []
    # iterate over experiments
    for idx, experiment in enumerate(experiments):
        record = experiment.copy()
        results = run(**experiment)
        if isinstance(results, dict):
            results = [results]
        for result in results:
            result.update(record)
        records += results
    df = pd.DataFrame(records)
    return df


def save_experiments(run, csv_file, on_progress=None, **kwargs):
    df = run_experiments(run, on_progress, **kwargs)
    df.to_csv(csv_file, index=False)
