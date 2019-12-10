import numpy as np


def filter_to_conv_weights(f, real=True, axes=None):
    """Compute convolution weights associated from filter.

    Parameters
    ----------
    - f : array
        filter
    - real : bool, default True
        If filter and weights are supposed to be real
    - axes : tuple or None
        Axes over which the filter operates. Default `None` is over all axes.

    Returns
    -------
    - w : array of same shape as f
        Associated convolution weights.
        w is equal the time reversed f along specified axes.
        For the fft along the specified axes we have w_fft = conjugate(f_fft)
    """
    f_fft = np.fft.fftn(f, axes=axes)
    w_fft = np.conjugate(f_fft)
    w = np.fft.ifftn(w, axes=axes)
    if real:
        w = np.real(w)
    return w


def first_derivative_filter(N):
    "Forward first derivative filter"
    f = np.zeros(N)
    f[0] = -1
    f[1] = 1
    return f


def second_derivative_filter(N):
    "Second derivative filter"
    f = np.zeros(N)
    f[0] = -2
    f[1] = f[-1] = 1
    return f


def gaussian_filter(sigma, N):
    "Scaled gaussian filter"
    freq = np.fft.fftfreq(N)
    coef = 2 * (np.pi * sigma)**2
    y_fft = np.exp(- coef * freq**2)
    y = np.fft.ifft(y_fft)
    y = np.real(y)
    return y


def first_derivative_along_axis(axis, shape):
    "Filter for forward derivative along axis"
    f = np.zeros(shape)
    swaped = np.swapaxes(f, -1, axis)
    assert swaped.shape[-1] == shape[axis]
    d = len(shape)
    zero = (0,) * (d-1)
    swaped[zero] = first_derivative_filter(swaped.shape[-1])
    f = np.swapaxes(swaped, -1, axis)
    return f


def second_derivative_along_axis(axis, shape):
    "Filter for second derivative along axis"
    f = np.zeros(shape)
    swaped = np.swapaxes(f, -1, axis)
    assert swaped.shape[-1] == shape[axis]
    d = len(shape)
    zero = (0,) * (d-1)
    swaped[zero] = second_derivative_filter(swaped.shape[-1])
    f = np.swapaxes(swaped, -1, axis)
    return f


def differential_filter(shape, D1, D2=None):
    "Filter D = D1 . dx + D2 . dx dx"
    d = len(shape)
    D2 = D2 or np.zeros(d)
    D = sum(
        D1[axis]*first_derivative_along_axis(axis, shape) for axis in range(d)
    ) + sum(
        D2[axis]*second_derivative_along_axis(axis, shape) for axis in range(d)
    )
    return D


def laplacian_filter(shape):
    "Laplacian filter"
    d = len(shape)
    laplacian = sum(
        second_derivative_along_axis(axis, shape) for axis in range(d)
    )
    return laplacian


def gradient_filters(shape):
    """Gradient filters

    Parameters
    ----------
    - shape : shape of input array
        d = len(shape) is the number of dimensions

    Returns
    -------
    - gradient : array of shape (d,) + shape
        gradient[i] = derivative filter along direction i for i in range(d)
    """
    d = len(shape)
    gradient = np.zeros((d,) + shape)
    for axis in range(d):
        gradient[axis] = first_derivative_along_axis(axis, shape)
    return gradient
