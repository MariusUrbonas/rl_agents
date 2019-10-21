import numpy as np
from utils import load_experiments

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def interpolate(x, y, total_len, samples=None):
    if samples is None:
        samples = total_len
    xvals = np.linspace(0, total_len, samples)
    yvals = np.interp(xvals, x, y)
    return xvals, yvals
    

def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys
    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == 'TIMESTEPS':
        x_var = np.cumsum(timesteps.ep_timesteps.values)
        y_var = timesteps.ep_reward.values
    elif xaxis == 'EPISODES':
        x_var = np.arange(len(timesteps))
        y_var = timesteps.ep_reward.values
    elif xaxis == 'WALLTIME':
        x_var = timesteps.ep_time.values / 3600.
        y_var = timesteps.ep_reward.values
    else:
        raise NotImplementedError
    return x_var, y_var


def ts2xinterpy(datas, total_len, xaxis='TIMESTEPS', smoothing_window=10):
    ys = []
    for data in datas:
        x, y = ts2xy(data, xaxis)
        y = moving_average(y, window=smoothing_window)
        # Truncate x
        x = x[len(x) - len(y):]
        x, y = interpolate(x, y, total_len)
        ys.append(y)
    return x, np.array(ys)


def logdir2xinterpy(log_dir, total_len,  xaxis='TIMESTEPS', smoothing_window=10):
    exps = load_experiments(log_dir)
    return ts2xinterpy(exps, total_len=total_len, xaxis=xaxis, smoothing_window=smoothing_window)