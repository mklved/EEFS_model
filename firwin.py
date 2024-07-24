import numpy as np
import scipy.signal as signal

def pad_boundary(x: np.ndarray, padwidth: int) -> np.ndarray:
    """
    Parameters
    ---------
    x: np.ndarray
        input
        shape: (width), (batches, width) or (batches, channels, width)
    padwidth: int
        padding width
    Returns
        output
        shape: (width + padwidth * 2), (batches, width + padwidth * 2) or (batches, channels, width + padwidth * 2)
    """

    shape = list(x.shape)
    width = shape[-1]
    x = np.reshape(x, (-1, width))

    assert padwidth > 1, "invalid padwidth"
    assert width > padwidth, "invalid width"

    w = 2**(-np.linspace(0, 2, padwidth + 1, endpoint=True))

    def padding(x, w):
        idx = np.arange(x.shape[-1])
        sw = np.sum(w)
        swx = np.sum(w * idx)
        swy = np.sum(w * x, axis=1, keepdims=True)
        swxy = np.sum(w * idx * x, axis=1, keepdims=True)
        swx2 = np.sum(w * idx * idx)
        c = sw * swx2 - swx * swx
        slope = (sw * swxy - swx * swy) / c
        intercept = (swx2 * swy - swx * swxy) / c

        rx = idx * slope + intercept
        ry = (idx[:-1] - x.shape[-1]) * slope + intercept

        dx = (x - rx)[:, 1:][:, ::-1]
        y = ry - dx * w[:-1][::-1]

        return y, ry

    x0, x1 = x[:, : padwidth + 1], x[:, ::-1][:, : padwidth + 1]
    y0, _ = padding(x0, w)
    y1, _ = padding(x1, w)

    y = np.concatenate([y0, x, y1[:, ::-1]], axis=1)

    shape[-1] = y.shape[-1]
    y = np.reshape(y, shape)

    return y

class FIRFilter():
    @staticmethod
    def filtering(x : np.ndarray, firwin : np.ndarray) -> np.ndarray:
        assert (len(firwin) % 2) == 1 and len(firwin) >= 3, "firwin must be odd length"

        padwidth = len(firwin) // 2
        x_pad = pad_boundary(x, padwidth)
        x_pad = np.reshape(x_pad, (-1, x_pad.shape[-1]))

        y = signal.lfilter(firwin, 1, x_pad, axis=-1)[:, len(firwin)-1:]

        y = np.reshape(y, x.shape)

        return y

class LowPass():
    def __init__(self, sample_hz : float, cutoff_hz : float, numtaps : int = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, cutoff_hz / nyquist_hz, pass_zero=True)
        
        self.__sample_hz = sample_hz
        self.__cutoff_hz = cutoff_hz

    def __call__(self, x : np.ndarray) -> np.ndarray:
        y = FIRFilter.filtering(x, self.__firwin)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def cutoff_hz(self) -> float:
        return self.__cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class HighPass():
    def __init__(self, sample_hz : float, cutoff_hz : float, numtaps : int = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, cutoff_hz / nyquist_hz, pass_zero=False)
        
        self.__sample_hz = sample_hz
        self.__cutoff_hz = cutoff_hz

    def __call__(self, x : np.ndarray) -> np.ndarray:
        y = FIRFilter.filtering(x, self.__firwin)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def cutoff_hz(self) -> float:
        return self.__cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class BandPass():
    def __init__(self, sample_hz : float, lower_cutoff_hz : float, higher_cutoff_hz : float, numtaps : int = 255) -> None:
        nyquist_hz = sample_hz / 2
        
        if lower_cutoff_hz <= 0:
            if lower_cutoff_hz < 0:
                import warnings
                warnings.warn("Negative value cufoff freq was specified for the BandPass. This is ignored.")

            self.__firwin = signal.firwin(numtaps, higher_cutoff_hz / nyquist_hz, pass_zero=True)
        elif higher_cutoff_hz >= nyquist_hz:
            if higher_cutoff_hz > nyquist_hz:
                import warnings
                warnings.warn("Cufoff freq above the nyquist freq was specified for the BandPass. This is ignored.")

            self.__firwin = signal.firwin(numtaps, lower_cutoff_hz / nyquist_hz, pass_zero=False)
        else:
            self.__firwin = signal.firwin(numtaps, [lower_cutoff_hz / nyquist_hz, higher_cutoff_hz / nyquist_hz], pass_zero=False)
        
        self.__sample_hz = sample_hz
        self.__lower_cutoff_hz = lower_cutoff_hz
        self.__higher_cutoff_hz = higher_cutoff_hz

    def __call__(self, x : np.ndarray) -> np.ndarray:
        y = FIRFilter.filtering(x, self.__firwin)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def lower_cutoff_hz(self) -> float:
        return self.__lower_cutoff_hz

    @property
    def higher_cutoff_hz(self) -> float:
        return self.__higher_cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin

class BandCut():
    def __init__(self, sample_hz : float, lower_cutoff_hz : float, higher_cutoff_hz : float, numtaps : int = 255) -> None:
        nyquist_hz = sample_hz / 2
        self.__firwin = signal.firwin(numtaps, [lower_cutoff_hz / nyquist_hz, higher_cutoff_hz / nyquist_hz], pass_zero=True)
        
        self.__sample_hz = sample_hz
        self.__lower_cutoff_hz = lower_cutoff_hz
        self.__higher_cutoff_hz = higher_cutoff_hz

    def __call__(self, x : np.ndarray) -> np.ndarray:
        y = FIRFilter.filtering(x, self.__firwin)

        return y

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def lower_cutoff_hz(self) -> float:
        return self.__lower_cutoff_hz

    @property
    def higher_cutoff_hz(self) -> float:
        return self.__higher_cutoff_hz

    @property
    def firwin(self) -> np.ndarray:
        return self.__firwin
    
class FIRChannels():
    def __init__(self, sample_hz : float, channels: int = 8, numtaps: int = 255) -> None:
        assert channels >= 2, "imvalid channels"
        assert (numtaps % 2) == 1 and numtaps >= 3, "firwin must be odd length"
        
        filters = []
        for i in range(channels - 1):
            filters.append(LowPass(sample_hz, np.ldexp(sample_hz, i - channels), numtaps))

        self.__weights = [filters[0].firwin]
        self.__cutoff_hz = [(0, filters[0].cutoff_hz)]
        for i in range(1, channels - 1):
            self.__weights.append(filters[i].firwin - filters[i - 1].firwin)
            self.__cutoff_hz.append((filters[i - 1].cutoff_hz, filters[i].cutoff_hz))

        delta = np.zeros(numtaps, dtype = filters[-1].firwin.dtype)
        delta[numtaps // 2] = 1

        self.__weights.append(delta - filters[-1].firwin)
        self.__cutoff_hz.append((filters[-1].cutoff_hz, np.inf))

        self.__weights = np.stack(self.__weights, axis = 0)
        self.__numtaps = numtaps

        self.__sample_hz = sample_hz
        
    def __call__(self, x : np.ndarray) -> np.ndarray:
        """
        Parameters
        ---------
        x: np.ndarray
            input
            shape: (width), (batches, width) or (batches, channels, width)
        Returns
        ---------
        y: np.ndarray
            input
            shape: (filters, width), (batches, filters, width) or (batches, channels, filters, width)
        """

        padwidth = self.__numtaps // 2
        x_pad = pad_boundary(x, padwidth)
        x_pad = np.reshape(x_pad, (-1, x_pad.shape[-1]))
        
        ys = []

        for firwin in self.__weights:
            y = signal.lfilter(firwin, 1, x_pad, axis=-1)[:, len(firwin)-1:]
            y = np.reshape(y, x.shape)
            ys.append(y)
            
        ys = np.stack(ys, axis=-2)

        return ys

    @property
    def weight(self) -> np.ndarray:
        return self.__weights.copy()

    @property
    def numtaps(self) -> int:
        return self.__numtaps

    @property
    def channels(self) -> int:
        return len(self.__weights)

    @property
    def sample_hz(self) -> float:
        return self.__sample_hz

    @property
    def cutoff_hz(self) -> int:
        return self.__cutoff_hz.copy()