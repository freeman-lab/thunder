"""
Class and standalone app for Fourier analysis
"""

import argparse
from numpy import mean, fix, sqrt, pi, array, angle
from numpy.fft import fft
from thunder.timeseries.base import TimeSeriesBase
from thunder.utils import ThunderContext, save


class Fourier(TimeSeriesBase):
    """Class for computing fourier transform"""

    def __init__(self, freq):
        self.freq = freq

    def get(self, y):
        """Compute fourier amplitude (coherence) and phase"""

        y = y - mean(y)
        nframes = len(y)
        ft = fft(y)
        ft = ft[0:int(fix(nframes/2))]
        amp_ft = 2*abs(ft)/nframes
        amp = amp_ft[self.freq]
        amp_sum = sqrt(sum(amp_ft**2))
        co = amp / amp_sum
        ph = -(pi/2) - angle(ft[self.freq])
        if ph < 0:
            ph += pi * 2
        return array([co, ph])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute a fourier transform on each time series")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("freq", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="fourier")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    out = Fourier(freq=args.freq).calc(data)

    outputdir = args.outputdir + "-fourier"
    save(out, outputdir, "fourier", "matlab")
