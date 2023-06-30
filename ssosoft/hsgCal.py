import numpy as np
import astropy.io.fits as fits
import glob
import configparser
import os
import re
import tqdm
from astropy.time import Time, TimeDelta
from datetime import datetime

class hsgCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    data from the Dunn Solar Telescope's Horizontal Steerable Grating spectrograph.

    --------------------------------------------------------------------------------

    Use this software package to process/reduce data from the HSG
    instrument at the Dunn Solar Telescope. This module is designed
    to perform first-order calibrations on HSG data from the DST.
    This module will gain-correct the provided data, then perfom
    wavelength calibration, spectral tilt correction.
    SEAN: Reassess after writing, see if FFT-based de-fringing function is required.

    --------------------------------------------------------------------------------

    Parameters:
    -----------
    instrument : str
        A string containing the instrument name.
        Currently accepted values are HSG_****,
        where **** denotes the spectral window of the raster.
    configFile : str
        Path to the configuration file.

    Example:
    --------

    To process a standard HSG dataset, with configFile 'config.ini' in the current directory, do
        import ssosoft
        h = ssosoft.hsgCal('HSG_8542', 'config.ini')
        h.hsg_run_calibration()

    """

    def __init__(self, instrument, configFile):
        """
        Parameters:
        -----------
        instrument : str
            A string containing the instrument name.
            Accepted values are HSG_****, where ****
            denotes the spectral window of the HSG data
        configFile : str
            Path to the configuration file.
        """

        self.avgDark = None
        self.avgSolarFlat = None
        self.avgLampFlat = None
        self.gainTable = None
        self.dataBase = ""
        self.dataList = []
        self.darkList = []
        self.sflatList = []
        self.lflatList = []
        self.configFile = configFile
        self.spectralWindow = instrument.split("_")[1]
        self.obsDate = ""
        self.obsTime = ""
        self.expTimems = ""
        self.workBase = ""
        self.imageShape = ()

        def hsg_average_image_from_list(self, fileList):
            """
            Computes an average image from a list of image files
            Parameters:
            -----------
            fileList : list
                A list of paths to images that are to be averaged

            Returns:
            --------
            avgIm : numpy.ndarray
                2-Dimensional numpy array with float64 dtype.
            """
            avgIm = np.zeros(self.imageShape)
            fnum = 0
            for file in fileList:
                with fits.open(file) as hdu:
                    fnum += len(hdu[1:])
                    for ext in hdu[1:]:
                        avgIm += ext.data
            return avgIm / fnum
