import astropy.io.fits as fits
from astropy.time import Time
from astropy.time import TimeDelta
from datetime import datetime
import configparser
import glob
import logging, logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import warnings

from sunpy.coordinates import frames
import astropy.units as u
from astropy.coordinates import SkyCoord


def _find_nearest(array, value):
    """Finds index of closest value in array"""
    return np.abs(array-value).argmin()


class rosaZylaCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    ROSA and Zyla data from the Dunn Solar Telescope.

    -----------------------------------------------------------------

    Use this software package to process/reduce data from the ROSA
    or Zyla instruments at the Dunn Solar Telescope. This package
    is designed to be used with the included wrapper for the
    Kiepenheuer-Institut Speckle Interferometry Package (KISIP).
    KISIP is not freely available. For more information, please
    see Woeger & Luehe, 2008, SPIE, 7019E doi:10.1117/12.788062.

    1) Install the softwate using the included distutils script.

    2) Set the necessary instrument parameters in a configuration
    file (use the included rosazyla_sampleConfig.ini as a template).

    3) Open a Python terminal and do `import ssosoft`.

    4) Start a new instance of the calibration class by doing
    `r=ssosoft.rosaZylaCal('<instrument>','<path to config file>')`.

    5) Run the standard calibration method
    `r.rosa_zyla_run_calibration()`.

    6) Use the kisipWrapper class to despeckle images using KISIP.

    -----------------------------------------------------------------

    Parameters
    ----------
    instrument : str
        A string containing the instrument name.
        Accepted values are ROSA_3500, ROSA_4170,
        ROSA_CAK, ROSA_GBAND, and ZYLA.
    configFile : str
        Path to the configuration file.

    -----------------------------------------------------------------

    See Also
    --------

    kisipWrapper : a Python wrapper for KISIP.

    -----------------------------------------------------------------

    Example
    -------

    To process a standard Zyla dataset, with configFile 'config.ini'
    in the current directory, do

        import ssosoft
        r=ssosoft.rosaZylaCal('zyla', 'config.ini')
        r.rosa_zyla_run_calibration()

    The necessary files for speckle analysis with KISIP are now
    available.

    -----------------------------------------------------------------
    """

    from ssosoft import ssosoftConfig

    def __init__(self, instrument, configFile):
        """
        Parameters
        ----------
        instrument : str
            A string containing the instrument name.
            Accepted values are ROSA_3500, ROSA_4170,
            ROSA_CAK, ROSA_GBAND, and ZYLA.
        configFile : str
            Path to the configuration file.

        """

        try:
            assert (instrument.upper() in ['ZYLA', 'ROSA_3500',
                                           'ROSA_4170', 'ROSA_CAK', 'ROSA_GBAND', 'ROSA_RED', "ROSA_6561"]
                    ), ('Allowed values for <instrument>: '
                        'ZYLA, ROSA_3500, ROSA_4170, '
                        'ROSA_CAK', 'ROSA_GBAND', 'ROSA_RED', "ROSA_6561"
                        )
        except Exception as err:
            print("Exception {0}".format(err))
            raise
        try:
            f = open(configFile, mode='r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.avgDark = None
        self.avgFlat = None
        self.batchList = []
        self.burstNumber = 0
        self.configFile = configFile
        self.darkBase = ""
        self.darkList = [""]
        self.dataBase = ""
        self.dataList = [""]
        self.dataShape = None
        self.darkFilePattern = ""
        self.dataFilePattern = ""
        self.flatFilePattern = ""
        self.flatBase = ""
        self.flatList = [""]
        self.gain = None
        self.imageShape = None
        self.instrument = instrument.upper()
        self.logFile = ""
        self.noise = None
        self.noiseFile = ""
        self.obsDate = ""
        self.obsTime = ""
        self.expTimems = ""

        self.preSpeckleBase = ""
        self.workBase = ""
        self.hdrBase = ""

        self.plateScaleX = 0
        self.plateScaleY = 0
        self.DCSSLog = ""
        self.DCSSTimes = np.array([])
        self.DCSSLat = np.array([])
        self.DCSSLon = np.array([])
        self.DCSSSee = np.array([])
        self.DCSSLlvl = np.array([])
        self.DCSSGdran = np.array([])
        self.DCSSSdim = np.array([])

    def rosa_zyla_average_image_from_list(self, fileList):
        """
        Computes an average image from a list of image files.

        Parameters
        ----------
        fileList : list
            A list of file paths to the images to be averaged.

        Returns
        -------
        numpy.ndarray
            2-Dimensional with dtype np.float32.
        """

        def rosa_zyla_print_average_image_progress():
            if not fNum % 100:
                self.logger.info("Progress: "
                                 "{:0.1%}.".format(fNum / numImg)
                                 )

        self.logger.info("Computing average image from {0} files "
                         "in directory: {1}".format(
            len(fileList), os.path.dirname(fileList[0])
        )
        )
        avgIm = np.zeros(self.imageShape, dtype=np.float32)
        fNum = 0
        numImg = len(fileList)
        for file in fileList:
            if 'ZYLA' in self.instrument:
                if fNum == 0:
                    numImg = len(fileList)
                avgIm = avgIm + self.rosa_zyla_read_binary_image(file)
                fNum += 1
                rosa_zyla_print_average_image_progress()
            if 'ROSA' in self.instrument:
                with fits.open(file) as hdu:
                    if fNum == 0:
                        numImg = len(fileList) * len(hdu[1:])
                    for ext in hdu[1:]:
                        avgIm = avgIm + ext.data
                        fNum += 1
                        rosa_zyla_print_average_image_progress()
        avgIm = avgIm / fNum

        self.logger.info("Images averaged/images predicted: "
                         "{0}/{1}".format(fNum, numImg)
                         )
        try:
            assert fNum == numImg
        except AssertionError as err:
            self.logger.warning("Number of images averaged does not match the "
                                "number predicted. If instrument is ROSA, then "
                                "this might be OK. Otherwise, something went wrong."
                                )
            self.logger.warning("This calibration will continue to run.")

        self.logger.info("Average complete, directory: "
                         "{0}".format(os.path.dirname(fileList[0]))
                         )
        return avgIm

    def rosa_zyla_check_dark_data_flat_shapes(self):
        """
        Checks dark, data, and flat frame sizes are all the same.
        NOT YET IMPLEMENTED.
        """
        pass

    def rosa_zyla_compute_gain(self):
        """
        Computes the gain table.
        """
        self.logger.info("Computing gain table.")
        darkSubtracted = self.avgFlat - self.avgDark
        try:
            self.gain = np.median(darkSubtracted) / (darkSubtracted)
        except Exception as err:
            self.error("Error computing gain table: "
                       "{0}".format(err)
                       )

        self.logger.info("Gain table computed.")

    def rosa_zyla_compute_noise_file(self):
        """
        Computes the noise file needed by KISIP. Optional.
        """
        self.logger.info("Computing noise cube: shape: "
                         "{0}".format(self.imageShape + (self.burstNumber,))
                         )
        self.noise = np.zeros((self.burstNumber,) + self.imageShape,
                              dtype=np.float32
                              )
        i = 0
        if "ROSA" in self.instrument:
            with fits.open(self.flatList[0]) as hdu:
                if len(hdu[1:]) >= self.burstNumber:
                    while i < self.burstNumber:
                        self.noise[i, :, :] = hdu[1:][i].data - self.avgDark
                        i += 1
                else:
                    for hduExt in hdu[1:]:
                        self.noise[i, :, :] = hduExt.data - self.avgDark
                        i += 1
                    while i < self.burstNumber:
                        self.noise[i, :, :] = np.nanmean(self.noise[:i, :, :], axis=0)
        else:
            for flat in self.flatList[0:self.burstNumber]:
                self.noise[i, :, :] = (self.rosa_zyla_read_binary_image(flat)
                                       - self.avgDark)
                i += 1
        self.noise = np.multiply(self.noise, self.gain)

        self.logger.info("Noise cube complete. "
                         "Saving to noise file: "
                         "{0}".format(
            os.path.join(self.preSpeckleBase, self.noiseFile)
        )
        )
        self.rosa_zyla_save_binary_image_cube(
            self.noise,
            os.path.join(self.preSpeckleBase, self.noiseFile)
        )
        self.logger.info("Saved noise file: "
                         "{0}".format(os.path.join(self.preSpeckleBase, self.noiseFile))
                         )

    def rosa_zyla_configure_run(self):
        """
        Configures the rosaZylaCal instance according to the contents of
        configFile.
        """

        def rosa_zyla_assert_base_dirs(baseDir):
            assert (os.path.isdir(baseDir)), (
                "Directory does not exist: {0}".format(baseDir)
            )

        config = configparser.ConfigParser()
        config.read(self.configFile)

        self.burstNumber = int(config[self.instrument]['burstNumber'])
        self.burstFileForm = config[self.instrument]['burstFileForm']
        self.darkBase = config[self.instrument]['darkBase']
        self.dataBase = config[self.instrument]['dataBase']
        self.darkFilePattern = config[self.instrument]['darkFilePattern']
        self.dataFilePattern = config[self.instrument]['dataFilePattern']
        self.flatFilePattern = config[self.instrument]['flatFilePattern']
        self.flatBase = config[self.instrument]['flatBase']
        self.noiseFile = config[self.instrument]['noiseFile']
        self.obsDate = config[self.instrument]['obsDate']
        self.obsTime = config[self.instrument]['obsTime']
        self.expTimems = config[self.instrument]['expTimems']
        self.speckledFileForm = config[self.instrument]['speckledFileForm']
        self.workBase = config[self.instrument]['workBase']
        self.plateScaleX = config[self.instrument]['kisipArcsecPerPixX']
        self.plateScaleY = config[self.instrument]['kisipArcsecPerPixY']
        if "SHARED" in list(config.keys()):
            self.DCSSLog = config['SHARED']['DCSSLog']

        self.preSpeckleBase = os.path.join(self.workBase, 'preSpeckle')
        self.speckleBase = os.path.join(self.workBase, 'speckle')
        self.postSpeckleBase = os.path.join(self.workBase, 'postSpeckle')
        self.hdrBase = os.path.join(self.workBase, 'hdrs')
        self.darkFile = os.path.join(self.workBase, '{0}_dark.fits'.format(self.instrument))
        self.flatFile = os.path.join(self.workBase, '{0}_flat.fits'.format(self.instrument))
        self.gainFile = os.path.join(self.workBase, '{0}_gain.fits'.format(self.instrument))
        self.noiseFileFits = os.path.join(self.workBase, '{0}_noise.fits'.format(self.instrument))

        ## Directories preSpeckleBase, speckleBase, and postSpeckle
        ## must exist or be created in order to continue.
        for dirBase in [self.preSpeckleBase, self.speckleBase, self.postSpeckleBase, self.hdrBase]:
            if not os.path.isdir(dirBase):
                print("{0}: os.mkdir: attempting to create directory:"
                      "{1}".format(__name__, dirBase)
                      )
                try:
                    os.mkdir(dirBase)
                except Exception as err:
                    print("An exception was raised: {0}".format(err))
                    raise

        ## Set-up logging.
        self.logFile = '{0}{1}'.format(
            os.path.join(self.workBase,
                         '{0}_{1}'.format(self.obsTime, self.instrument.lower())
                         ),
            '.log'
        )
        logging.config.fileConfig(self.configFile,
                                  defaults={'logfilename': self.logFile}
                                  )
        self.logger = logging.getLogger('{0}Log'.format(self.instrument.lower()))

        ## Print an intro message.
        self.logger.info("This is SSOsoft version {0}".format(self.ssosoftConfig.__version__))
        self.logger.info("Contact {0} to report bugs, make suggestions, "
                         "or contribute.".format(self.ssosoftConfig.__email__))
        self.logger.info("Now configuring this {0} data calibration run.".format(self.instrument))

        ## darkBase, dataBase, and flatBase directories must exist.
        try:
            rosa_zyla_assert_base_dirs(self.darkBase)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using dark directory: {0}".format(self.darkBase))

        try:
            rosa_zyla_assert_base_dirs(self.dataBase)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using data directory: {0}".format(self.dataBase))

        try:
            rosa_zyla_assert_base_dirs(self.flatBase)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using flat directory: {0}".format(self.flatBase))

    def rosa_zyla_detect_rosa_dims(self, header):
        """
        Detects data and image dimensions in ROSA FITS image file headers.

        Parameters
        ----------
        header : astropy.io.fits.header
            A FITS header conforming to the FITS specification.
        """
        try:
            self.dataShape = (header['NAXIS2'], header['NAXIS1'])
            self.imageShape = (header['NAXIS2'], header['NAXIS1'])
        except Exception as err:
            self.logger.critical("Could not read from FITS header: "
                                 "{0}".format(err)
                                 )
        self.logger.info("Auto-detected data dimensions "
                         "(rows, cols): {0}".format(self.dataShape))
        self.logger.info("Auto-detected image dimensions "
                         "(rows, cols): {0}".format(self.imageShape))

    def rosa_zyla_detect_zyla_dims(self, imageData):
        """
        Detects the data and image dimensions in Zyla unformatted
        binary image files.

        Parameters
        ----------
        imageData : numpy.ndarray
            A one-dimensional Numpy array containing image data.
        """

        # Detects data then usable image dimensions.
        # Assumes all overscan regions within the raw
        # image has a zero value. If no overscan,
        # this function will fail. Will also fail if
        # dead pixels are present.
        def rosa_zyla_detect_overscan():
            # Borrowed from a Stackoverflow article
            # titled "Finding the consecutive zeros
            # in a numpy array." Author unknown.
            # Create an array that is 1 where imageData is 0,
            # and pad each end with an extra 0.
            # LATER: check for identical indices which would
            #	be indicative of non-contiguous dead
            #	pixels.
            iszero = np.concatenate(([0],
                                     np.equal(imageData, 0).view(np.int8),
                                     [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0]
            self.logger.info("Zeros boundary detected at: "
                             "{0}".format(np.unique(ranges))
                             )
            return ranges

        # Detects image columns by looking for the
        # last overscan pixel in the first row. Rows
        # are computed from the quotient of the
        # number of pixels and the number of columns.
        # Get data dimensions.
        # LATER: add function to check for overscan.
        #	Could be done by examining the number
        # 	of results in ovrScn.
        self.logger.info("Attempting to detect overscan and data shape.")
        ovrScn = rosa_zyla_detect_overscan()
        datDim = (np.uint16(imageData.size / ovrScn[1]),
                  ovrScn[1])
        # Detects usable image columns by using
        # the first overscan index. Finds first
        # overscan row by looking for the first
        # occurance where overscan indices are
        # not separated by dx1 or dx2.
        # Get image dimensions.
        self.logger.info("Attempting to detect image shape.")
        dx1 = ovrScn[1] - ovrScn[0]
        dx2 = ovrScn[2] - ovrScn[1]
        DeltaX = np.abs(np.diff(ovrScn))
        endRow = (np.where(
            np.logical_and(
                DeltaX != dx1,
                DeltaX != dx2
            )
        )[0])[0] / 2 + 1
        imgDim = (np.uint16(endRow), ovrScn[0])
        self.dataShape = datDim
        self.imageShape = imgDim
        self.logger.info("Auto-detected data dimensions "
                         "(rows, cols): {0}".format(self.dataShape))
        self.logger.info("Auto-detected image dimensions "
                         "(rows, cols): {0}".format(self.imageShape))

    def rosa_zyla_display_image(self, im):
        """
        Displays image data.

        Parameters
        ----------
        im : numpy.ndarray or array-like
            A 2-dimensional array containing image data.
        """
        plt.imshow(im, origin='upper',
                   interpolation='none',
                   cmap='hot'
                   )
        plt.show()

    def rosa_zyla_get_cal_images(self):
        """
        Reads average dark, average flat, and gain files and store as class
        attributes if exist. If these files do not exist, compute the average
        dark, average flat, and gain images and store as class attributes.
        """
        if os.path.exists(self.darkFile):
            self.logger.info("Average dark file found: {0}".format(self.darkFile))
            self.logger.info("Reading average dark.")
            with fits.open(self.darkFile) as hdu:
                self.avgDark = hdu[0].data
        else:
            self.avgDark = self.rosa_zyla_average_image_from_list(
                self.darkList
            )
        if os.path.exists(self.flatFile):
            self.logger.info("Average flat file found: {0}".format(self.flatFile))
            self.logger.info("Reading average flat.")
            with fits.open(self.flatFile) as hdu:
                self.avgFlat = hdu[0].data
        else:
            self.avgFlat = self.rosa_zyla_average_image_from_list(
                self.flatList
            )
        if os.path.exists(self.gainFile):
            self.logger.info("Gain file found: {0}".format(self.gainFile))
            self.logger.info("Reading gain file.")
            with fits.open(self.gainFile) as hdu:
                self.gain = hdu[0].data
        else:
            self.rosa_zyla_compute_gain()

    def rosa_zyla_get_file_lists(self):
        """
        Construct darkList, dataList, and flatList attributes, which
        are lists of respective file types.
        """

        def rosa_zyla_assert_file_list(fList):
            assert (len(fList) != 0), "List contains no matches."

        self.logger.info("Searching for darks, flats, and data files.")
        self.logger.info("Searching for dark image files: {0}".format(self.darkBase))
        self.darkList = glob.glob(
            os.path.join(self.darkBase, self.darkFilePattern)
        )
        try:
            rosa_zyla_assert_file_list(self.darkList)
        except AssertionError as err:
            self.logger.critical("Error: darkList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in darkList: {0}".format(len(self.darkList)))

        self.logger.info("Searching for data image files: {0}".format(self.dataBase))
        self.dataList = glob.glob(
            os.path.join(self.dataBase, self.dataFilePattern)
        )
        try:
            rosa_zyla_assert_file_list(self.dataList)
        except AssertionError as err:
            self.logger.critical("Error: dataList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in dataList: {0}".format(len(self.dataList)))

        self.logger.info("Searching for flat image files: {0}".format(self.flatBase))
        self.flatList = glob.glob(
            os.path.join(self.flatBase, self.flatFilePattern)
        )
        try:
            rosa_zyla_assert_file_list(self.flatList)
        except AssertionError as err:
            self.logger.critical("Error: flatList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in flatList: {0}".format(len(self.flatList)))

    def rosa_zyla_get_data_image_shapes(self, file):
        """
        The main data and image shape detection method.

        Parameters
        ----------
        file : str
            Path to image file.
        """
        self.logger.info("Detecting image and data dimensions in "
                         "file: {0}".format(file)
                         )
        if 'ZYLA' in self.instrument:
            try:
                with open(file, mode='rb') as imageFile:
                    imageData = np.fromfile(imageFile,
                                            dtype=np.uint16
                                            )
            except Exception as err:
                self.logger.critical("Could not get image or data "
                                     "shapes: {0}".format(err)
                                     )
                raise
            self.rosa_zyla_detect_zyla_dims(imageData)
        if 'ROSA' in self.instrument:
            try:
                with fits.open(file) as hdu:
                    header = hdu[1].header
            except Exception as err:
                self.logger.critical("Could not get image or data "
                                     "shapes: {0}".format(err)
                                     )
                raise
            self.rosa_zyla_detect_rosa_dims(header)

    def rosa_zyla_order_files(self):
        """
        Orders sequentially numbered file names in numerical order.
        Contains a special provision for ordering Zyla files, which
        begin with the least-significant digit.
        """

        def rosa_zyla_order_file_list(fList):
            if 'ZYLA' in self.instrument:
                orderList = [''] * len(fList)  ## List length same as fList
                ptrn = '[0-9]+'  # Match any digit one or more times.
                p = re.compile(ptrn)
                flag = 0
                for f in fList:
                    head, tail = os.path.split(f)
                    match = p.match(tail)
                    if match:
                        digitNew = match.group()[::-1]
                        # Added 2025-02-10 to skip any missing files and warn user
                        if int(digitNew) >= len(orderList):
                            flag += 1
                            continue
                        orderList[int(digitNew)] = f
                    else:
                        self.logger.error(
                            "Unexpected filename format: "
                            "{0}".format(f)
                        )
                if flag != 0:
                    for f in orderList:
                        if f == "":
                            flag += 1
                    orderList = [f for f in orderList if f != ""]
                    warnings.warn("WARNING: Zyla transfer error, {0} files missing".format(flag))
            if 'ROSA' in self.instrument:
                fList.sort()
                orderList = fList
            try:
                assert (all(orderList)), "List could not be ordered."
            except AssertionError as err:
                self.logger.critical("Error: {0}".format(err))
                raise
            else:
                return orderList

        self.logger.info("Sorting darkList.")
        self.darkList = rosa_zyla_order_file_list(self.darkList)
        self.logger.info("Sorting flatList.")
        self.flatList = rosa_zyla_order_file_list(self.flatList)
        self.logger.info("Sorting dataList.")
        self.dataList = rosa_zyla_order_file_list(self.dataList)

    def rosa_zyla_read_binary_image(self, file, dataShape=None, imageShape=None, dtype=np.uint16):
        """
        Reads an unformatted binary file. Slices the image as
        s[i] ~ 0:imageShape[i].

        Parameters
        ----------
        file : str
            Path to binary image file.
        dataShape : tuple
            Shape of the image or cube.
        imageShape : tuple
            Shape of sub-image or region of interest.
        dtype : Numpy numerical data type.
            Default is numpy.uint16.

        Returns
        -------
        numpy.ndarray : np.float32, shape imageShape.
        """
        if dataShape is None:
            dataShape = self.dataShape

        if imageShape is None:
            imageShape = self.imageShape

        try:
            with open(file, mode='rb') as imageFile:
                imageData = np.fromfile(imageFile,
                                        dtype=dtype
                                        )
        except Exception as err:
            self.logger.critical("Could not open/read binary image file: "
                                 "{0}".format(err)
                                 )
            raise

        im = imageData.reshape((dataShape))
        ## Generate a tuple of slice objects, s[i]~ 0:imageShape[i]
        s = tuple()
        for t in imageShape:
            s = s + np.index_exp[0:t]
        im = im[s]
        return np.float32(im)

    def rosa_zyla_run_calibration(self, saveBursts=True):
        """
        The main calibration method for standard ROSA or Zyla data.

        Parameters
        ----------

        saveBursts : bool
            Default True. Set to True to save burst cubes.
            Set to False to skip saving burst cubes.
        """
        self.rosa_zyla_configure_run()
        self.logger.info("Starting standard {0} calibration.".format(self.instrument)
                         )
        self.rosa_zyla_get_file_lists()
        self.rosa_zyla_order_files()
        self.rosa_zyla_get_data_image_shapes(self.flatList[0])
        self.rosa_zyla_get_cal_images()
        self.rosa_zyla_save_cal_images()
        if saveBursts:
            self.rosa_zyla_save_bursts()
        else:
            self.logger.info("SaveBursts set to {0}. "
                             "Skipping the save bursts step.".format(saveBursts)
                             )
        self.logger.info("Finished standard {0} calibration.".format(self.instrument)
                         )

    def rosa_zyla_save_binary_image_cube(self, data, file):
        """
        Saves binary images cubes formatted for KISIP.

        Parameters
        ----------
        data : numpy.ndarray dtype np.float32
            Image data to be saved.
        file : str
            Named path to save the image cube.
        """
        try:
            with open(file, mode='wb') as f:
                data.tofile(f)
        except Exception as err:
            self.logger.critical("Could not save binary file: {0}".format(err))
            raise

    def zyla_time(self, file_number):

        year = int(self.obsDate[0:4])
        month = int(self.obsDate[4:6])
        day = int(self.obsDate[6:8])
        hour = int(self.obsTime[0:2])
        minute = int(self.obsTime[2:4])
        second = int(self.obsTime[4:6])  # + ()
        t = Time(datetime(year, month, day, hour, minute, second))
        dt = TimeDelta(0.001 * int(self.expTimems) * int(self.burstNumber) * file_number, format='sec')
        return (t + dt)

    def parse_dcss(self):
        """
        Parser for DCSS logs to set up time, scintillation, light level, and pointing arrays.
        Modified 2024-06-17 to add ability to read ICC logs as well.
        These are largly similar, but DATE and TIME are seperate lines, VTT is used in place of DST,
        and there are no equal signs.
        """
        # No DCSS Log Set
        if self.DCSSLog == "":
            return
        with open(self.DCSSLog, "r") as f:
            dcss_lines = f.readlines()

        lat = []
        lon = []
        scin = []
        timestamp = []
        llvl = []
        gdran = []
        srad = []
        for line in dcss_lines:
            if "DST_TIME" in line:
                timestamp.append(
                    np.datetime64(
                        line.split("=")[1].split("/")[0].strip().replace("\'","")
                    )
                )
            # Reconstructing ICC Datetimes, which are kept in "DATE" and "TIME" lines
            # DCSS has WP_TIME, RM1_TIME, PT4_TIME, etc... we want to make sure we're not
            # getting these lines.  Since ICC logs don't have the underscore character
            # (except in CTRK TRACK_ERROR), we can filter by "_".
            # This is a blatant hack, but it works.
            elif ("TIME" in line) & ("_" not in line):
                timestamp.append(line.split(" ")[-1].replace("\n", ""))
            elif "DATE" in line:
                date = line.replace("\n", "").split(" ")[-1].split("/")
                year = "20" + date[-1]
                month = date[0]
                day = date[1]
                fdate = "-".join([year, month, day])
                timestamp[-1] = fdate + "T" + timestamp[-1]

            if "DST_SLAT" in line:
                lat.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SLAT" in line:
                lat.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_SLNG" in line:
                lon.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SLNG" in line:
                lon.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_SEE" in line:
                scin.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SEE" in line:
                scin.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_LLVL" in line:
                llvl.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT LLVL" in line:
                llvl.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )
            if "DST_GDRN" in line:
                gdran.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT GDRN" in line:
                gdran.append(
                    float(
                        line.split(" ")[-1].replace("\n" ,"")
                    )
                )
            if "DST_SDIM" in line:
                srad.append(
                    float(
                        line.split("=")[1].split("/")[0].strip()
                    )
                )
            elif "VTT SDIM" in line:
                srad.append(
                    float(
                        line.split(" ")[-1].replace("\n", "")
                    )
                )

        self.DCSSLat = np.array(lat)
        self.DCSSLon = np.array(lon)
        self.DCSSTimes = np.array(timestamp, dtype='datetime64[ms]')
        self.DCSSSee = np.array(scin)
        self.DCSSLlvl = np.array(llvl)
        self.DCSSSdim = np.array(srad)
        self.DCSSGdran = np.array(gdran)
        return


    def rosa_zyla_save_bursts(self):
        """
        Main method to save burst cubes formatted for KISIP.
        """

        def rosa_zyla_flatfield_correction(data):
            corrected = self.gain * (data - self.avgDark)
            return corrected

        def rosa_zyla_print_progress_save_bursts():
            self.logger.info("Progress: {:0.2%} "
                             "with file: {:s}".format(
                burst / lastBurst,
                burstFile
            )
            )

        burstShape = (self.burstNumber,) + self.imageShape
        self.logger.info("Preparing burst files, saving in directory: "
                         "{0}".format(self.preSpeckleBase)
                         )
        self.logger.info("Number of files to be read: "
                         "{0}".format(len(self.dataList))
                         )
        self.logger.info("Flat-fielding and saving data to burst files "
                         "with burst number: {0}: shape: {1}".format(
            self.burstNumber, burstShape
        )
        )
        burstCube = np.zeros(burstShape, dtype=np.float32)
        burst = 0
        batch = -1
        self.parse_dcss()
        if 'ZYLA' in self.instrument:
            lastBurst = len(self.dataList) // self.burstNumber
            lastFile = lastBurst * self.burstNumber
            i = 0
            for file in self.dataList[:lastFile]:
                data = self.rosa_zyla_read_binary_image(file)
                burstCube[i, :, :] = rosa_zyla_flatfield_correction(data)
                burstThsnds = burst // 1000
                burstHndrds = burst % 1000
                if i == 0:
                    startdate = self.zyla_time(1000 * burstThsnds + burstHndrds).fits
                i += 1
                if i == self.burstNumber:
                    burstFile = os.path.join(
                        self.preSpeckleBase,
                        (self.burstFileForm).format(
                            self.obsDate,
                            self.obsTime,
                            burstThsnds,
                            burstHndrds
                        )
                    )
                    hdrFile = os.path.join(
                        self.hdrBase,
                        (self.burstFileForm).format(
                            self.obsDate,
                            self.obsTime,
                            burstThsnds,
                            burstHndrds
                        )
                    )
                    text_file = open(hdrFile + '.txt', "w")
                    text_file.write("STARTOBS=" + startdate + "\n")
                    text_file.write('ENDOBS=' + self.zyla_time(1000 * burstThsnds + burstHndrds).fits + "\n")
                    text_file.write('EXPOSURE=' + self.expTimems + "\n")
                    if self.DCSSLog != "":
                        zyla_idx = _find_nearest(
                            self.DCSSTimes,
                            np.datetime64(self.zyla_time(1000 * burstThsnds + burstHndrds).fits)
                        )
                        slon = self.DCSSLon[zyla_idx]
                        slat = self.DCSSLat[zyla_idx]
                        # Remember, Guider head is permanently rotated to 13.3 degrees
                        rotan = self.DCSSGdran[zyla_idx] - 13.3
                        solrad = self.DCSSSdim[zyla_idx]
                        llvl = self.DCSSLlvl[zyla_idx]
                        scin = self.DCSSSee[zyla_idx]
                        telescope_pointing = SkyCoord(
                            slon * u.degree,
                            slat * u.degree,
                            frame=frames.HeliographicStonyhurst,
                            observer='earth',
                            obstime=self.DCSSTimes[zyla_idx].astype(str)
                        ).transform_to(frames.Helioprojective)
                        text_file.write("CRVAL1  ="+str(telescope_pointing.Tx.value) + "\n")
                        text_file.write("CRVAL2  ="+str(telescope_pointing.Ty.value) + "\n")
                        text_file.write("CTYPE1  =HPLN-TAN\n")
                        text_file.write("CTYPE2  =HPLT-TAN\n")
                        text_file.write("CUNIT1  =arcsec\n")
                        text_file.write("CUNIT2  =arcsec\n")
                        text_file.write("CDELT1  ="+str(self.plateScaleX)+"\n")
                        text_file.write("CDELT2  ="+str(self.plateScaleY)+"\n")
                        text_file.write("CRPIX1  ="+str(self.imageShape[0]/2)+"\n")
                        text_file.write("CRPIX2  ="+str(self.imageShape[1]/2)+"\n")
                        text_file.write("CROTA2  ="+str(rotan)+'\n')
                        text_file.write("SCINT   ="+str(scin)+'\n')
                        text_file.write("LLVL    ="+str(llvl)+'\n')
                        text_file.write("RSUN_ARC="+str(solrad))


                    text_file.close()
                    self.rosa_zyla_save_binary_image_cube(
                        burstCube,
                        burstFile
                    )
                    i = 0
                    burst += 1
                    rosa_zyla_print_progress_save_bursts()
                    if burstThsnds != batch:
                        batch = burstThsnds
                        (self.batchList).append(batch)
                    burstCube = np.zeros(burstShape,
                                         dtype=np.float32
                                         )
        if 'ROSA' in self.instrument:
            i = 0
            header_index = 0
            for file in self.dataList:
                with fits.open(file) as hdu:
                    for hduExt in hdu[1:]:
                        if (burst == 0) and (i == 0):
                            lastBurst = len(self.dataList) * len(hdu[1:]) // self.burstNumber
                        burstCube[i, :, :] = rosa_zyla_flatfield_correction(hduExt.data)
                        if i == 0:
                            startdate = hduExt.header['DATE']
                        i += 1
                        header_index += 1
                        if i == self.burstNumber:
                            burstThsnds = burst // 1000
                            burstHndrds = burst % 1000
                            burstFile = os.path.join(
                                self.preSpeckleBase,
                                (self.burstFileForm).format(
                                    self.obsDate,
                                    self.obsTime,
                                    burstThsnds,
                                    burstHndrds
                                )
                            )
                            hdrFile = os.path.join(
                                self.hdrBase,
                                (self.burstFileForm).format(
                                    self.obsDate,
                                    self.obsTime,
                                    burstThsnds,
                                    burstHndrds
                                )
                            )
                            text_file = open(hdrFile + '.txt', "w")
                            # here I should embed a line writting txt file with header
                            # this could probably make another module
                            if header_index >= 257:
                                header_index = header_index - 256
                            text_file.write(repr(hdu[header_index].header) + "\n")
                            text_file.write("STARTOBS=" + startdate + "\n")
                            text_file.write("ENDOBS=" + hdu[header_index].header['DATE'] + '\n')
                            # text_file.write("\n")
                            # text_file.write(repr(hdu[0].header))
                            if self.DCSSLog != "":
                                rosa_idx = _find_nearest(
                                    self.DCSSTimes,
                                    np.datetime64(hdu[header_index].header['DATE'])
                                )
                                slon = self.DCSSLon[rosa_idx]
                                slat = self.DCSSLat[rosa_idx]
                                # Remember, Guider head is permanently rotated to 13.3 degrees
                                rotan = self.DCSSGdran[rosa_idx] - 13.3
                                solrad = self.DCSSSdim[rosa_idx]
                                llvl = self.DCSSLlvl[rosa_idx]
                                scin = self.DCSSSee[rosa_idx]
                                telescope_pointing = SkyCoord(
                                    slon * u.degree,
                                    slat * u.degree,
                                    frame=frames.HeliographicStonyhurst,
                                    observer='earth',
                                    obstime=self.DCSSTimes[rosa_idx].astype(str)
                                ).transform_to(frames.Helioprojective)
                                text_file.write("CRVAL1  =" + str(telescope_pointing.Tx.value) + "\n")
                                text_file.write("CRVAL2  =" + str(telescope_pointing.Ty.value) + "\n")
                                text_file.write("CTYPE1  =HPLN-TAN\n")
                                text_file.write("CTYPE2  =HPLT-TAN\n")
                                text_file.write("CUNIT1  =arcsec\n")
                                text_file.write("CUNIT2  =arcsec\n")
                                text_file.write("CDELT1  =" + str(self.plateScaleX) + "\n")
                                text_file.write("CDELT2  =" + str(self.plateScaleY) + "\n")
                                text_file.write("CRPIX1  =" + str(self.imageShape[0] / 2) + "\n")
                                text_file.write("CRPIX2  =" + str(self.imageShape[1] / 2) + "\n")
                                text_file.write("CROTA2  =" + str(rotan)+'\n')
                                text_file.write("SCINT   =" + str(scin)+'\n')
                                text_file.write("LLVL    =" + str(llvl)+'\n')
                                text_file.write("RSUN_REF=" + str(solrad))
                            # print(burstFile)
                            # keywords = list(hdu[burstThsnds].header.keys())
                            # for L in range (len(keywords)):
                            # text_file.writelines(keywords[L])
                            #								text_file.writelines(hdu[burstThsnds].header[L])
                            text_file.close()
                            # the end of my alternations related to a header export
                            self.rosa_zyla_save_binary_image_cube(
                                burstCube,
                                burstFile
                            )
                            i = 0
                            burst += 1
                            rosa_zyla_print_progress_save_bursts()
                            if burstThsnds != batch:
                                batch = burstThsnds
                                (self.batchList).append(batch)
                            burstCube = np.zeros(burstShape,
                                                 dtype=np.float32
                                                 )

        self.logger.info("Burst files complete: {0}".format(self.preSpeckleBase))

    def rosa_zyla_save_cal_images(self):
        """
        Saves average dark, average flat, gain, and noise images
        in FITS format.
        """
        if os.path.exists(self.darkFile):
            self.logger.info("Dark file already exists: {}".format(self.darkFile))
        else:
            self.logger.info("Saving average dark: "
                             "{0}".format(
                os.path.join(
                    self.workBase,
                    self.darkFile
                )
            )
            )
            self.rosa_zyla_save_fits_image(self.avgDark,
                                           os.path.join(
                                               self.workBase,
                                               self.darkFile
                                           )
                                           )
        if os.path.exists(self.flatFile):
            self.logger.info("Flat file already exists: {0}".format(self.flatFile))
        else:
            self.logger.info("Saving average flat: "
                             "{0}".format(
                os.path.join(
                    self.workBase,
                    self.flatFile
                )
            )
            )
            self.rosa_zyla_save_fits_image(self.avgFlat,
                                           os.path.join(
                                               self.workBase,
                                               self.flatFile
                                           )
                                           )
        if os.path.exists(self.gainFile):
            self.logger.info("Gain file already exists: {0}".format(self.gainFile))
        else:
            self.logger.info("Saving gain: "
                             "{0}".format(
                os.path.join(
                    self.workBase,
                    self.gainFile
                )
            )
            )
            self.rosa_zyla_save_fits_image(self.gain,
                                           os.path.join(
                                               self.workBase,
                                               self.gainFile
                                           )
                                           )

        if os.path.exists(self.noiseFile):
            self.logger.info("Noise KISIP file already exists: {0}".format(self.noiseFile))
        else:
            self.logger.info(
                "Saving KISIP noise: {0}".format(
                    os.path.join(
                        self.workBase,
                        self.noiseFile
                    )
                )
            )
            self.rosa_zyla_compute_noise_file()

        if os.path.exists(self.noiseFileFits):
            self.logger.info("Noise FITS file already exists: {0}".format(self.noiseFileFits))
        else:
            self.logger.info("Saving noise: "
                             "{0}".format(
                os.path.join(
                    self.workBase,
                    self.noiseFileFits
                )
            )
            )
            self.rosa_zyla_save_fits_image(self.noise,
                                           os.path.join(
                                               self.workBase,
                                               self.noiseFileFits
                                           )
                                           )

    def rosa_zyla_save_despeckled_as_fits(self):
        """
        Saves despeckled (processed unformatted binary) KISIP images as
        FITS images.
        """
        self.logger.info("Saving despeckled binary image files to FITS.")
        self.logger.info("Searching for files: "
                         "{0}".format(
            os.path.join(self.speckleBase,
                         self.speckledFileForm.format(
                             self.obsDate, self.obsTime, 0, 0
                         )[:-7] + '*.final'
                         )
        )
        )
        fList = glob.glob(os.path.join(self.speckleBase,
                                       self.speckledFileForm.format(
                                           self.obsDate, self.obsTime, 0, 0
                                       )[:-7] + '*.final'
                                       )
                          )
        fListOrder = []
        for b in range(len(fList)):
            proxy1 = fList[b].split('speckle.batch.')
            proxy2 = proxy1[1].split('.')
            fListOrder.append(proxy2[0] + proxy2[1])
        fListOrder = [int(s) for s in fListOrder]
        header_prefix = self.preSpeckleBase + '/' + self.obsDate + '_' + self.obsTime
        headerList = glob.glob(header_prefix + '*.txt')
        headerListOrder = []
        for b in range(len(headerList)):
            proxy1 = headerList[b].split('raw.batch.')
            proxy2 = proxy1[1].split('.')
            headerListOrder.append(proxy2[0] + proxy2[1])
        headerListOrder = [int(s) for s in headerListOrder]

        try:
            assert (len(fList) != 0)
        except Exception as err:
            self.logger.critical("CRITICAL: no files found: {0}".format(err))
            raise
        else:
            self.logger.info("Found {0} files.".format(len(fList)))
        for i in range(len(fList)):
            im = self.rosa_zyla_read_binary_image(fList[i],
                                                  imageShape=self.imageShape,
                                                  dataShape=self.imageShape,
                                                  dtype=np.float32
                                                  )
            fName = os.path.basename(fList[i])
            my_file_index = headerListOrder.index(fListOrder[i])
            headerFile = open(headerList[my_file_index], "r")
            self.rosa_zyla_save_fits_image(im, os.path.join(
                self.postSpeckleBase,
                fName + '.fits'
            ), headerFile.readlines()
                                           )
            headerFile.close()
        self.logger.info("Finished saving despeckled images as FITS "
                         "in directory: {0}".format(self.postSpeckleBase))

    def rosa_zyla_save_fits_image(self, image, file, header='', clobber=True):
        """
        Saves 2-dimensional image data to a FITS file.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional image data to save.
        file : str
            Path to file to save to.
        clobber : bool
            Overwrite existing file if True, otherwise do not overwrite.
        """
        # for i in range(len(header)): header[i].translate({ord("'"): None})
        # print(header)
        if 'ROSA' in self.instrument:
            hdu = fits.PrimaryHDU(image)
            if len(header) == 35:
                hdr = hdu.header
                indexHeader = [5, 6, 7, 8]
                for i in indexHeader:
                    lineBreak = header[i].split('=')
                    lineBreak2 = lineBreak[1].split('/')
                    #		print(i)
                    #		print(type(str(lineBreak2[0])))
                    hdr[lineBreak[0]] = int(lineBreak2[0])
                indexHeaderString = [9, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
                for i in indexHeaderString:
                    lineBreak = header[i].split('=')
                    lineBreak2 = lineBreak[1].split('/')
                    # print('s')
                    hdr[lineBreak[0]] = lineBreak2[0].replace('\n', ' ').replace("'", '').strip()
            #	print(hdr)
            # empty_primary = fits.PrimaryHDU(header = hdr)
            # try:
            #	hdr[lineBreak[0]]=lineBreak2[0]#.translate({ord("'"): None})
            # hdr = hdu.header
            hdul = fits.HDUList([hdu])
        if 'ZYLA' in self.instrument:
            hdu = fits.PrimaryHDU(image)
            if len(header) == 2:
                hdr = hdu.header
                lineBreak = header[0].split('=')
                hdr[lineBreak[0]] = lineBreak[1].replace("\n", ' ')
                lineBreak = header[1].split('=')
                hdr[lineBreak[0]] = lineBreak[1]
                hdr['comment'] = 'WARNING: Timestamps were reconstructed during the data reduction.'
                hdr['comment'] = 'Timestamp = start time + burst number * time exposure * file number'
            hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(file, overwrite=clobber)
        except Exception as err:
            self.logger.warning("Could not write FITS file: "
                                "{0}".format(file)
                                )
            self.logger.warning("FITS write warning: continuing, but "
                                "this could cause problems later."
                                )
