import logging
import itertools
import numpy as np
import pandas as pd


def log_on_progress(i, total):
    logging.info(f"experiment {i}/{total}")


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
            result = run(**experiment)
            record.update(result)
            records.append(record)
        except Exception as e:
            logging.error(f"Experiment {experiment} failed\n{e}")
        on_progress(idx + 1, n_experiments)
    df = pd.DataFrame(records)
    return df


def save_experiments(run, csv_file, on_progress=None, **kwargs):
    df = run_experiments(run, on_progress, **kwargs)
    df.to_csv(csv_file, index=False)
