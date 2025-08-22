import configparser
import glob
import logging
import logging.config
import os
import re
import warnings
from datetime import datetime

import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from scipy.signal import find_peaks
from sunpy.coordinates import frames


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

    def __init__(self, instrument: str, config_file: str) -> None:
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
            assert (instrument.upper() in ["ZYLA", "ROSA_3500",
                                           "ROSA_4170", "ROSA_CAK",
                                           "ROSA_GBAND", "ROSA_RED",
                                           "ROSA_6561", "ROSA_NAD", "ROSA_CAM"]
                    ), ("Allowed values for <instrument>: "
                        "ZYLA", "ROSA_3500", "ROSA_4170",
                        "ROSA_CAK", "ROSA_GBAND", "ROSA_RED",
                        "ROSA_6561", "ROSA_NAD", "ROSA_CAM"
                        )
        except Exception as err:
            print("Exception {0}".format(err))
            raise
        try:
            f = open(config_file, mode="r")
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.config_file = config_file
        self.instrument = instrument.upper()
        self.logfile = ""

        self.avg_dark = np.array([])
        self.avg_flat = np.array([])
        self.avg_target = np.array([])
        self.avg_linegrid = np.array([])
        self.gain = np.array([])
        self.imageshape = (0, 0)
        self.noise = np.array([])

        self.dark_file = ""
        self.flat_file = ""
        self.gain_file = ""
        self.linegrid_file = ""
        self.target_file = ""
        self.noise_file = ""
        self.noise_file_fits = ""

        self.batch_list = []
        self.burst_number = 0

        self.dark_base = ""
        self.dark_list = [""]
        self.data_base = ""
        self.data_list = [""]
        self.data_shape = ()
        self.dark_file_pattern = ""
        self.data_file_pattern = ""
        self.flat_file_pattern = ""
        self.flat_base = ""
        self.flat_list = [""]
        self.target_base = ""
        self.target_list = []
        self.linegrid_base = ""
        self.linegrid_list = []

        self.obsdate = ""
        self.obstime = ""
        self.exptime_ms = ""

        self.burst_file_form = ""
        self.speckle_file_form = ""
        self.target_file_pattern = ""
        self.linegrid_file_pattern = ""

        self.speckle_base = ""
        self.postspeckle_base = ""
        self.prespeckle_base = ""
        self.workbase = ""
        self.hdrbase = ""

        self.plate_scale_x = 0
        self.plate_scale_y = 0
        self.plate_scale = (0, 0)
        self.dcss_log = ""
        self.dcss_times = np.array([])
        self.dcss_lat = np.array([])
        self.dcss_lon = np.array([])
        self.dcss_see = np.array([])
        self.dcss_llvl = np.array([])
        self.dcss_gdran = np.array([])
        self.dcss_sdim = np.array([])

        self.correct_plate_scale = False

        self.correct_time = False

        self.create_fiducial_maps = False

        self.progress = True

        self.dst_plate_scale = 206264.806 / 54864.

        return

    def rosa_zyla_run_calibration(self, save_bursts: bool=True) -> None:
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
        self.rosa_zyla_get_data_image_shapes(self.flat_list[0])
        self.rosa_zyla_get_cal_images()
        self.rosa_zyla_save_cal_images()
        self.parse_dcss()
        if self.correct_plate_scale:
            self.determine_grid_spacing()
        if save_bursts:
            self.rosa_zyla_save_bursts()
        else:
            self.logger.info("SaveBursts set to {0}. "
                             "Skipping the save bursts step.".format(save_bursts)
                             )
        self.logger.info("Finished standard {0} calibration.".format(self.instrument)
                         )
        return

    def rosa_zyla_average_image_from_list(self, filelist: list) -> np.ndarray:
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
            if not fnum % 100:
                self.logger.info("Progress: "
                                 "{:0.1%}.".format(fnum / num_img)
                                 )

        self.logger.info("Computing average image from {0} files "
                         "in directory: {1}".format(
            len(filelist), os.path.dirname(filelist[0])
        )
        )
        avg_im = np.zeros(self.imageshape, dtype=np.float32)
        fnum = 0
        num_img = len(filelist)
        for file in tqdm.tqdm(
            filelist,
            desc=f"Computing Average Image from {len(filelist)} files",
            disable=not self.progress
        ):
            if "ZYLA" in self.instrument:
                if fnum == 0:
                    num_img = len(filelist)
                avg_im = avg_im + self.rosa_zyla_read_binary_image(file)
                fnum += 1
                rosa_zyla_print_average_image_progress()
            if "ROSA" in self.instrument:
                with fits.open(file) as hdu:
                    if fnum == 0:
                        num_img = len(filelist) * len(hdu[1:])
                    for ext in hdu[1:]:
                        avg_im = avg_im + ext.data
                        fnum += 1
                        rosa_zyla_print_average_image_progress()
        avg_im = avg_im / fnum

        self.logger.info("Images averaged/images predicted: "
                         "{0}/{1}".format(fnum, num_img)
                         )
        try:
            assert fnum == num_img
        except AssertionError:
            self.logger.warning(
                "Number of images averaged does not match the "
                "number predicted. If instrument is ROSA, then "
                "this might be OK. Otherwise, something went wrong."
            )
            self.logger.warning("This calibration will continue to run.")

        self.logger.info("Average complete, directory: "
                         "{0}".format(os.path.dirname(filelist[0]))
                         )
        return avg_im

    def rosa_zyla_compute_gain(self) -> None:
        """
        Computes the gain table.
        """
        self.logger.info("Computing gain table.")
        dark_subtract = self.avg_flat - self.avg_dark
        try:
            self.gain = np.median(dark_subtract) / (dark_subtract)
        except Exception as err:
            self.logger.critical("Error computing gain table: {0}".format(err))

        self.logger.info("Gain table computed.")
        return

    def rosa_zyla_compute_noise_file(self) -> None:
        """
        Computes the noise file needed by KISIP. Optional.
        """
        self.logger.info(
            "Computing noise cube: shape: {0}".format(
                self.imageshape + (self.burst_number,)
            )
        )
        self.noise = np.zeros(
            (self.burst_number, *self.imageshape),
            dtype=np.float32
        )
        i = 0
        if "ROSA" in self.instrument:
            with fits.open(self.flat_list[0]) as hdu:
                if len(hdu[1:]) >= self.burst_number:
                    while i < self.burst_number:
                        self.noise[i, :, :] = hdu[1:][i].data - self.avg_dark
                        i += 1
                else:
                    for hdu_ext in hdu[1:]:
                        self.noise[i, :, :] = hdu_ext.data - self.avg_dark
                        i += 1
                    while i < self.burst_number:
                        self.noise[i, :, :] = np.nanmean(
                            self.noise[:i, :, :], axis=0
                        )
        else:
            for flat in self.flat_list[0:self.burst_number]:
                self.noise[i, :, :] = (
                    self.rosa_zyla_read_binary_image(flat) - self.avg_dark
                )
                i += 1
        self.noise = np.multiply(self.noise, self.gain)

        self.logger.info(
            "Noise cube complete. Saving to noise file: {0}".format(
                os.path.join(self.prespeckle_base, self.noise_file)
            )
        )
        self.rosa_zyla_save_binary_image_cube(
            self.noise,
            os.path.join(self.prespeckle_base, self.noise_file)
        )
        self.logger.info(
            "Saved noise file: {0}".format(
                os.path.join(self.prespeckle_base, self.noise_file)
            )
        )
        return

    def rosa_zyla_configure_run(self) -> None:
        """
        Configures the rosaZylaCal instance according to the contents of
        configFile.
        """

        def rosa_zyla_assert_base_dirs(base_dir):
            assert (os.path.isdir(base_dir)), (
                "Directory does not exist: {0}".format(base_dir)
            )

        config = configparser.ConfigParser()
        config.read(self.config_file)

        if "SHARED" in list(config.keys()):
            self.dcss_log = config["SHARED"]["DCSSLog"] if "dcsslog" \
                in config["SHARED"].keys() else self.dcss_log

        self.data_base = config[self.instrument]["dataBase"]
        # If these are undefined, assign them to the same directory as dataBase
        # Useful for ROSA where all files are in the same directory.
        self.dark_base = config[self.instrument]["darkBase"] if "darkbase" \
            in config[self.instrument].keys() else self.data_base
        self.flat_base = config[self.instrument]["flatBase"] if "flatbase" \
            in config[self.instrument].keys() else self.data_base
        self.target_base = config[self.instrument]["targetBase"] if \
            "targetbase" in config[self.instrument].keys() else self.data_base
        self.linegrid_base = config[self.instrument]["linegridBase"] if \
            "linegridbase" in config[self.instrument].keys() else self.data_base

        self.workbase = config[self.instrument]["workBase"]

        self.burst_number = int(config[self.instrument]["burstNumber"])
        self.burst_file_form = config[self.instrument]["burstFileForm"]

        self.obsdate = config[self.instrument]["obsDate"]
        self.obstime = config[self.instrument]["obsTime"]
        self.exptime_ms = config[self.instrument]["expTimems"]

        self.speckle_file_form = config[self.instrument]["speckledFileForm"]

        # Again, since all Zyla files are named *spool.dat,
        # having it multiply-defined for each obstype is redundant
        # Instead, if not all are given, pin the file pattern to dataFilePattern
        self.data_file_pattern = config[self.instrument]["dataFilePattern"]
        self.dark_file_pattern = config[self.instrument]["darkFilePattern"] if \
            "darkfilepattern" in config[self.instrument].keys() else self.data_file_pattern
        self.flat_file_pattern = config[self.instrument]["flatFilePattern"] if \
            "flatfilepattern" in config[self.instrument].keys() else self.data_file_pattern
        self.target_file_pattern = config[self.instrument]["targetFilePattern"] \
            if "targetfilepattern" in config[self.instrument].keys() else self.data_file_pattern
        self.linegrid_file_pattern = config[self.instrument]["linegridFilePattern"] \
            if "linegridfilepattern" in config[self.instrument].keys() else self.data_file_pattern

        self.noise_file = config[self.instrument]["noiseFile"]

        self.plate_scale_x = float(config[self.instrument]["kisipArcsecPerPixX"])
        self.plate_scale_y = float(config[self.instrument]["kisipArcsecPerPixY"])
        self.plate_scale = np.array([self.plate_scale_x, self.plate_scale_y])

        if "correctplatescale" in config[self.instrument].keys():
            if config[self.instrument]["correctPlateScale"].lower() == "true":
                self.correct_plate_scale = True

        if "createfiducialmaps" in config[self.instrument].keys():
            if config[self.instrument]["createFiducialMaps"].lower() == "true":
                self.create_fiducial_maps = True

        if "correcttime" in config[self.instrument].keys():
            if config[self.instrument]["correctTime"].lower() == "true":
                self.correct_time = True

        self.progress = config[self.instrument]["progress"] if "progress" \
            in config[self.instrument].keys() else self.progress
        if str(self.progress).lower() == "true":
            self.progress = True # passthrough to tqdm "disable" kwarg. If False, progress bar is spawned

        self.prespeckle_base = os.path.join(self.workbase, "preSpeckle")
        self.speckle_base = os.path.join(self.workbase, "speckle")
        self.postspeckle_base = os.path.join(self.workbase, "postSpeckle")
        self.hdrbase = os.path.join(self.workbase, "hdrs")
        self.dark_file = os.path.join(self.workbase, "{0}_dark.fits".format(self.instrument))
        self.flat_file = os.path.join(self.workbase, "{0}_flat.fits".format(self.instrument))
        self.gain_file = os.path.join(self.workbase, "{0}_gain.fits".format(self.instrument))
        self.noise_file_fits = os.path.join(self.workbase, "{0}_noise.fits".format(self.instrument))
        if self.create_fiducial_maps:
            self.linegrid_file = os.path.join(self.workbase, "{0}_linegrid.fits".format(self.instrument))
            self.target_file = os.path.join(self.workbase, "{0}_target.fits".format(self.instrument))

        ## Directories preSpeckleBase, speckleBase, and postSpeckle
        ## must exist or be created in order to continue.
        for dir_base in [self.prespeckle_base, self.speckle_base, self.postspeckle_base, self.hdrbase]:
            if not os.path.isdir(dir_base):
                print("{0}: os.mkdir: attempting to create directory:"
                      "{1}".format(__name__, dir_base)
                      )
                try:
                    os.mkdir(dir_base)
                except Exception as err:
                    print("An exception was raised: {0}".format(err))
                    raise

        ## Set-up logging.
        self.logfile = "{0}{1}".format(
            os.path.join(
                self.workbase,
                "{0}_{1}".format(self.obstime, self.instrument.lower())
            ),
            ".log"
        )
        logging.config.fileConfig(
            self.config_file,
            defaults={"logfilename": self.logfile}
        )
        self.logger = logging.getLogger("{0}Log".format(self.instrument.lower()))

        ## Print an intro message.
        self.logger.info("This is SSOsoft version {0}".format(self.ssosoftConfig.__version__))
        self.logger.info("Contact {0} to report bugs, make suggestions, "
                         "or contribute.".format(self.ssosoftConfig.__email__))
        self.logger.info("Now configuring this {0} data calibration run.".format(self.instrument))

        ## darkBase, dataBase, and flatBase directories must exist.
        try:
            rosa_zyla_assert_base_dirs(self.dark_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using dark directory: {0}".format(self.dark_base))

        try:
            rosa_zyla_assert_base_dirs(self.data_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using data directory: {0}".format(self.data_base))

        try:
            rosa_zyla_assert_base_dirs(self.flat_base)
        except AssertionError as err:
            self.logger.critical("Fatal: {0}".format(err))
            raise
        else:
            self.logger.info("Using flat directory: {0}".format(self.flat_base))

        return

    def rosa_zyla_detect_rosa_dims(self, header):
        """
        Detects data and image dimensions in ROSA FITS image file headers.

        Parameters
        ----------
        header : astropy.io.fits.header
            A FITS header conforming to the FITS specification.
        """
        try:
            self.data_shape = (header["NAXIS2"], header["NAXIS1"])
            self.imageshape = (header["NAXIS2"], header["NAXIS1"])
        except Exception as err:
            self.logger.critical("Could not read from FITS header: "
                                 "{0}".format(err)
                                 )
        self.logger.info("Auto-detected data dimensions "
                         "(rows, cols): {0}".format(self.data_shape))
        self.logger.info("Auto-detected image dimensions "
                         "(rows, cols): {0}".format(self.imageshape))

        return

    def rosa_zyla_detect_zyla_dims(self, image_date):
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
                                     np.equal(image_date, 0).view(np.int8),
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
        overscan = rosa_zyla_detect_overscan()
        datadims = (np.uint16(image_date.size / overscan[1]),
                  overscan[1])
        # Detects usable image columns by using
        # the first overscan index. Finds first
        # overscan row by looking for the first
        # occurance where overscan indices are
        # not separated by dx1 or dx2.
        # Get image dimensions.
        self.logger.info("Attempting to detect image shape.")
        dx1 = overscan[1] - overscan[0]
        dx2 = overscan[2] - overscan[1]
        dx = np.abs(np.diff(overscan))
        endrow = (np.where(
            np.logical_and(
                dx != dx1,
                dx != dx2
            )
        )[0])[0] / 2 + 1
        imgdims = (np.uint16(endrow), overscan[0])
        self.data_shape = datadims
        self.imageshape = imgdims
        self.logger.info("Auto-detected data dimensions "
                         "(rows, cols): {0}".format(self.data_shape))
        self.logger.info("Auto-detected image dimensions "
                         "(rows, cols): {0}".format(self.imageshape))

        return

    def rosa_zyla_display_image(self, im):
        """
        Displays image data.

        Parameters
        ----------
        im : numpy.ndarray or array-like
            A 2-dimensional array containing image data.
        """
        plt.imshow(im, origin="upper",
                   interpolation="none",
                   cmap="hot"
                   )
        plt.show()
        return

    def rosa_zyla_get_cal_images(self):
        """
        Reads average dark, average flat, and gain files and store as class
        attributes if exist. If these files do not exist, compute the average
        dark, average flat, and gain images and store as class attributes.
        """
        if os.path.exists(self.dark_file):
            self.logger.info("Average dark file found: {0}".format(self.dark_file))
            self.logger.info("Reading average dark.")
            with fits.open(self.dark_file) as hdu:
                self.avg_dark = hdu[0].data
        else:
            self.avg_dark = self.rosa_zyla_average_image_from_list(
                self.dark_list
            )
        if os.path.exists(self.flat_file):
            self.logger.info("Average flat file found: {0}".format(self.flat_file))
            self.logger.info("Reading average flat.")
            with fits.open(self.flat_file) as hdu:
                self.avg_flat = hdu[0].data
        else:
            self.avg_flat = self.rosa_zyla_average_image_from_list(
                self.flat_list
            )
        if os.path.exists(self.gain_file):
            self.logger.info("Gain file found: {0}".format(self.gain_file))
            self.logger.info("Reading gain file.")
            with fits.open(self.gain_file) as hdu:
                self.gain = hdu[0].data
        else:
            self.rosa_zyla_compute_gain()

        if self.create_fiducial_maps or self.correct_plate_scale:
            if os.path.exists(self.linegrid_file):
                self.logger.info("Average linegrid file found: {0}".format(self.linegrid_file))
                self.logger.info("Reading average linegrid.")
                with fits.open(self.linegrid_file) as hdu:
                    self.avg_linegrid = hdu[0].data
            else:
                # Make sure we're not doing a ton of averages
                if len(self.linegrid_list) > 5:
                    self.linegrid_list = self.linegrid_list[:5]
                self.avg_linegrid = self.rosa_zyla_average_image_from_list(self.linegrid_list)
                self.avg_linegrid = (self.avg_linegrid - self.avg_dark) * self.gain
            if self.create_fiducial_maps:
                if os.path.exists(self.target_file):
                    self.logger.info("Average target file found: {0}".format(self.target_file))
                else:
                    if len(self.target_list) > 5:
                        self.target_list = self.target_list[:5]
                    self.avg_target = self.rosa_zyla_average_image_from_list(self.target_list)
                    self.avg_target = (self.avg_target - self.avg_dark) * self.gain

        return

    def determine_grid_spacing(self) -> None:
        """Uses self.avg_linegrid to get a better estimate of X/Y plate scale"""
        xmedian_spacing = []
        # Take the middle 300 pixels for grid determination
        for i in range(self.avg_linegrid.shape[0]//2 - 150, self.avg_linegrid.shape[0]//2 + 150):
            profile = 1/self.avg_linegrid[i, :]
            profile /= profile.max()
            approx_pix_scale = self.dst_plate_scale / self.plate_scale_x
            peaks, _ = find_peaks(profile, height=0.75, distance=approx_pix_scale / 2)
            xmedian_spacing.append(np.nanmedian(peaks[1:] - peaks[:-1]))
        xpix_scale = self.dst_plate_scale / np.nanmedian(xmedian_spacing)
        if np.abs((xpix_scale - self.plate_scale_x) / self.plate_scale_x) < 0.1:
            # We're within 10% of the initial value,
            # so we'll update the plate scale with the new value
            self.plate_scale_x = xpix_scale
        else:
            self.logger.warning(
                "Plate scale [x] could not be updated! Original Value: {0:03d}. Estimated Value: {1:03d}".format(
                    self.plate_scale_x, xpix_scale
                )
            )

        ymedian_spacing = []
        for i in range(self.avg_linegrid.shape[1]//2 - 150, self.avg_linegrid.shape[1]//2 + 150):
            profile = 1/self.avg_linegrid[:, i]
            profile /= profile.max()
            approx_pix_scale = self.dst_plate_scale / self.plate_scale_y
            peaks, _ = find_peaks(profile, height=0.75, distance=approx_pix_scale / 2)
            ymedian_spacing.append(np.nanmedian(peaks[1:] - peaks[:-1]))
        ypix_scale = self.dst_plate_scale / np.nanmedian(ymedian_spacing)
        if np.abs((ypix_scale - self.plate_scale_y) / self.plate_scale_y) < 0.1:
            self.plate_scale_y = ypix_scale
        else:
            self.logger.warning(
                "Plate scale [y] could not be updated! Original Value: {0:03d}. Estimated Value: {1:03d}".format(
                    self.plate_scale_y, ypix_scale
                )
            )
        self.plate_scale = (self.plate_scale_x, self.plate_scale_y)
        plate_scale_filename = os.path.join(self.workbase, "plate_scale.txt")
        np.savetxt(plate_scale_filename, np.array(self.plate_scale))
        return

    def rosa_zyla_get_file_lists(self):
        """
        Construct darkList, dataList, and flatList attributes, which
        are lists of respective file types.
        """

        def rosa_zyla_assert_file_list(flist):
            assert (len(flist) != 0), "List contains no matches."

        self.logger.info("Searching for darks, flats, fiducial, and data files.")
        self.logger.info("Searching for dark image files: {0}".format(self.dark_base))
        self.dark_list = glob.glob(
            os.path.join(self.dark_base, self.dark_file_pattern)
        )
        try:
            rosa_zyla_assert_file_list(self.dark_list)
        except AssertionError as err:
            self.logger.critical("Error: darkList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in darkList: {0}".format(len(self.dark_list)))

        self.logger.info("Searching for data image files: {0}".format(self.data_base))
        self.data_list = glob.glob(
            os.path.join(self.data_base, self.data_file_pattern)
        )
        try:
            rosa_zyla_assert_file_list(self.data_list)
        except AssertionError as err:
            self.logger.critical("Error: dataList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in dataList: {0}".format(len(self.data_list)))

        self.logger.info("Searching for flat image files: {0}".format(self.flat_base))
        self.flat_list = glob.glob(
            os.path.join(self.flat_base, self.flat_file_pattern)
        )
        try:
            rosa_zyla_assert_file_list(self.flat_list)
        except AssertionError as err:
            self.logger.critical("Error: flatList: {0}".format(err))
            raise
        else:
            self.logger.info("Files in flatList: {0}".format(len(self.flat_list)))

        if self.correct_plate_scale or self.create_fiducial_maps:
            self.logger.info("Searching for AF resolution target image files: {0}".format(self.target_base))
            self.target_list = sorted(glob.glob(
                os.path.join(self.target_base, self.target_file_pattern)
            ))
            try:
                rosa_zyla_assert_file_list(self.target_list)
            except AssertionError as err:
                self.logger.critical("Error: targetList: {0}".format(err))
                raise
            else:
                self.logger.info("Files in targetList: {0}".format(len(self.target_list)))

            self.logger.info("Searching for line grid image files: {0}".format(self.linegrid_base))
            self.linegrid_list = sorted(glob.glob(
                os.path.join(self.linegrid_base, self.linegrid_file_pattern)
            ))
            try:
                rosa_zyla_assert_file_list(self.linegrid_list)
            except AssertionError as err:
                self.logger.critical("Error: lingridList: {0}".format(err))
                raise
            else:
                self.logger.info("Files in linegridList: {0}".format(len(self.linegrid_list)))

        return

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
        if "ZYLA" in self.instrument:
            try:
                with open(file, mode="rb") as image_file:
                    image_data = np.fromfile(image_file,
                                            dtype=np.uint16
                                            )
            except Exception as err:
                self.logger.critical("Could not get image or data "
                                     "shapes: {0}".format(err)
                                     )
                raise
            self.rosa_zyla_detect_zyla_dims(image_data)
        if "ROSA" in self.instrument:
            try:
                with fits.open(file) as hdu:
                    header = hdu[1].header
            except Exception as err:
                self.logger.critical("Could not get image or data "
                                     "shapes: {0}".format(err)
                                     )
                raise
            self.rosa_zyla_detect_rosa_dims(header)
        return

    def rosa_zyla_order_files(self):
        """
        Orders sequentially numbered file names in numerical order.
        Contains a special provision for ordering Zyla files, which
        begin with the least-significant digit.
        """

        def rosa_zyla_order_file_list(flist):
            if "ZYLA" in self.instrument:
                orderlist = [""] * len(flist)  ## List length same as fList
                ptrn = "[0-9]+"  # Match any digit one or more times.
                p = re.compile(ptrn)
                flag = 0
                for f in flist:
                    _, tail = os.path.split(f)
                    match = p.match(tail)
                    if match:
                        digitnew = match.group()[::-1]
                        # Added 2025-02-10 to skip any missing files and warn user
                        if int(digitnew) >= len(orderlist):
                            flag += 1
                            continue
                        orderlist[int(digitnew)] = f
                    else:
                        self.logger.error(
                            "Unexpected filename format: "
                            "{0}".format(f)
                        )
                if flag != 0:
                    for f in orderlist:
                        if f == "":
                            flag += 1
                    orderlist = [f for f in orderlist if f != ""]
                    warnings.warn("WARNING: Zyla transfer error, {0} files missing".format(flag))
            if "ROSA" in self.instrument:
                flist.sort()
                orderlist = flist
            try:
                assert (all(orderlist)), "List could not be ordered."
            except AssertionError as err:
                self.logger.critical("Error: {0}".format(err))
                raise
            else:
                return orderlist

        self.logger.info("Sorting darkList.")
        self.dark_list = rosa_zyla_order_file_list(self.dark_list)
        self.logger.info("Sorting flatList.")
        self.flat_list = rosa_zyla_order_file_list(self.flat_list)
        self.logger.info("Sorting dataList.")
        self.data_list = rosa_zyla_order_file_list(self.data_list)
        return

    def rosa_zyla_read_binary_image(self, file, data_shape=None, image_shape=None, dtype=np.uint16) -> np.ndarray:
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
        if data_shape is None:
            data_shape = self.data_shape

        if image_shape is None:
            image_shape = self.imageshape

        try:
            with open(file, mode="rb") as image_file:
                image_data = np.fromfile(image_file,
                                        dtype=dtype
                                        )
        except Exception as err:
            self.logger.critical("Could not open/read binary image file: "
                                 "{0}".format(err)
                                 )
            raise

        im = image_data.reshape((data_shape))
        ## Generate a tuple of slice objects, s[i]~ 0:imageShape[i]
        s = tuple()
        for t in image_shape:
            s = s + np.index_exp[0:t]
        im = im[s]
        return im.astype(np.float32)

    def rosa_zyla_save_binary_image_cube(self, data, file) -> None:
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
            with open(file, mode="wb") as f:
                data.tofile(f)
        except Exception as err:
            self.logger.critical("Could not save binary file: {0}".format(err))
            raise
        return

    def zyla_time(self, file_number) -> Time:

        year = int(self.obsdate[0:4])
        month = int(self.obsdate[4:6])
        day = int(self.obsdate[6:8])
        hour = int(self.obstime[0:2])
        minute = int(self.obstime[2:4])
        second = int(self.obstime[4:6])  # + ()
        t = Time(datetime(year, month, day, hour, minute, second))
        dt = TimeDelta(0.001 * int(self.exptime_ms) * int(self.burst_number) * file_number, format="sec")
        return (t + dt)

    def parse_dcss(self) -> None:
        """
        Parser for DCSS logs to set up time, scintillation, light level, and pointing arrays.
        Modified 2024-06-17 to add ability to read ICC logs as well.
        These are largly similar, but DATE and TIME are seperate lines, VTT is used in place of DST,
        and there are no equal signs.
        """
        # No DCSS Log Set
        if self.dcss_log == "":
            return
        with open(self.dcss_log, "r") as f:
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

        self.dcss_lat = np.array(lat)
        self.dcss_lon = np.array(lon)
        self.dcss_times = np.array(timestamp, dtype="datetime64[ms]")
        self.dcss_see = np.array(scin)
        self.dcss_llvl = np.array(llvl)
        self.dcss_sdim = np.array(srad)
        self.dcss_gdran = np.array(gdran)
        return

    def rosa_zyla_save_bursts(self) -> None:
        """
        Main method to save burst cubes formatted for KISIP.
        """
        burst_file = ""
        last_burst = 0
        def rosa_zyla_flatfield_correction(data):
            corrected = self.gain * (data - self.avg_dark)
            return corrected

        def rosa_zyla_print_progress_save_bursts():
            self.logger.info(
                "Progress: {:0.2%} with file: {:s}".format(burst / last_burst, burst_file)
            )

        burst_shape = (self.burst_number, *self.imageshape)

        self.logger.info("Preparing burst files, saving in directory: {0}".format(self.prespeckle_base))
        self.logger.info("Number of files to be read: {0}".format(len(self.data_list)))
        self.logger.info(
            "Flat-fielding and saving data to burst files with burst number: {0}: shape: {1}".format(
                self.burst_number, burst_shape
            )
        )

        burst_cube = np.zeros(burst_shape, dtype=np.float32)
        burst = 0
        batch = -1

        if "ZYLA" in self.instrument:
            last_burst = len(self.data_list) // self.burst_number
            last_file = last_burst * self.burst_number
            i = 0
            for file in tqdm.tqdm(self.data_list[:last_file], desc="Writing burst files", disable=not self.progress):
                data = self.rosa_zyla_read_binary_image(file)
                burst_cube[i, :, :] = rosa_zyla_flatfield_correction(data)
                burst_thousands = burst // 1000
                burst_hundreds = burst % 1000
                startdate = ""
                if i == 0:
                    startdate = self.zyla_time(1000 * burst_thousands + burst_hundreds).fits
                i += 1
                if i == self.burst_number:
                    burst_file = os.path.join(
                        self.prespeckle_base,
                        self.burst_file_form.format(self.obsdate, self.obstime, burst_thousands, burst_hundreds)
                    )
                    enddate = self.zyla_time(1000 * burst_thousands + burst_hundreds).fits
                    self.write_header(burst_thousands, burst_hundreds, startdate, enddate)

                    self.rosa_zyla_save_binary_image_cube(burst_cube, burst_file)
                    i = 0
                    burst += 1
                    rosa_zyla_print_progress_save_bursts()
                    if burst_thousands != batch:
                        batch = burst_thousands
                        self.batch_list.append(batch)
                    burst_cube = np.zeros(burst_shape, dtype=np.float32)
        if "ROSA" in self.instrument:
            i = 0
            header_index = 0
            for file in tqdm.tqdm(
                self.data_list, desc="Processing ROSA files to KISIP bursts.", disable=not self.progress
            ):
                with fits.open(file) as hdu:
                    for hdu_ext in hdu[1:]:
                        if (burst == 0) and (i == 0):
                            last_burst = len(self.data_list) * len(hdu[1:]) // self.burst_number
                        burst_cube[i, :, :] = rosa_zyla_flatfield_correction(hdu_ext.data) # type: ignore
                        if i == 0:
                            startdate = hdu_ext.header["DATE"]
                            if self.correct_time:
                                startdate = (np.datetime64(startdate) - np.timedelta64(1, "h")).astype(str)
                        i += 1
                        header_index += 1
                        if i == self.burst_number:
                            burst_thousands = burst // 1000
                            burst_hundreds = burst % 1000
                            burst_file = os.path.join(
                                self.prespeckle_base,
                                (self.burst_file_form).format(
                                    self.obsdate,
                                    self.obstime,
                                    burst_thousands,
                                    burst_hundreds
                                )
                            )
                            enddate = (
                                np.datetime64(hdu_ext.header["DATE"]) + np.timedelta64(int(self.exptime_ms), "ms")
                            ).astype(str)
                            if header_index >= 257:
                                header_index = header_index - 256

                            self.write_header(
                                burst_thousands, burst_hundreds,
                                startdate, enddate=enddate, additional_header=hdu[header_index].header
                            )

                            # the end of my alternations related to a header export
                            self.rosa_zyla_save_binary_image_cube(
                                burst_cube,
                                burst_file
                            )
                            i = 0
                            burst += 1
                            rosa_zyla_print_progress_save_bursts()
                            if burst_thousands != batch:
                                batch = burst_thousands
                                (self.batch_list).append(batch)
                            burst_cube = np.zeros(burst_shape,
                                                 dtype=np.float32
                                                 )

        self.logger.info("Burst files complete: {0}".format(self.prespeckle_base))
        return

    def write_header(
            self, burst_thousands: int, burst_hundreds: int,
            startdate: str, enddate: str, additional_header: fits.Header | None=None
        ) -> None:
        """Writes header information file to the workBase directory

        Parameters
        ----------
        burstThsnds : int
            Thousands place of the file burst
        burstHndrds : int
            Hundreds place of the file burst
        startdate : str
            Startdate (UT)
        enddate : str
            Enddate (UT)
        additional_header : fits.Header | None, optional
            For ROSA files, additional header information to write, by default None
        """
        header_filename = os.path.join(
            self.hdrbase,
            self.burst_file_form.format(self.obsdate, self.obstime, burst_thousands, burst_hundreds)
        )

        header_file = open(header_filename + ".txt", "w")

        if additional_header is not None:
            header_file.write(repr(additional_header) + "\n")

        header_file.write("STARTOBS={0}\n".format(startdate))
        header_file.write("ENDOBS={0}\n".format(enddate))
        header_file.write("EXPTIME={0}\n".format(self.exptime_ms))
        header_file.write("NSUMEXP={0}\n".format(self.burst_number))
        header_file.write("TEXPOSUR={0}\n".format(self.exptime_ms * self.burst_number))

        obs_td = (np.datetime64(enddate) - np.datetime64(startdate)) / 2
        avg_date = np.datetime64(startdate) + obs_td

        header_file.write("DATE-AVG={0}\n".format(str(avg_date)))

        if self.dcss_log != "":
            dcss_idx = _find_nearest(
                self.dcss_times,
                avg_date
            )
            slon = self.dcss_lon[dcss_idx]
            slat = self.dcss_lat[dcss_idx]
            # Remember, Guider head is permanently rotated to 13.3 degrees
            rotan = self.dcss_gdran[dcss_idx] - 13.3
            solrad = self.dcss_sdim[dcss_idx]
            llvl = self.dcss_llvl[dcss_idx]
            scin = self.dcss_see[dcss_idx]
            telescope_pointing = SkyCoord(
                slon * u.degree,
                slat * u.degree,
                frame=frames.HeliographicStonyhurst,
                observer="earth",
                obstime=self.dcss_times[dcss_idx].astype(str)
            ).transform_to(frames.Helioprojective)
            header_file.write("CRVAL1  ={0}\n".format(telescope_pointing.Tx.value))
            header_file.write("CRVAL2  ={0}\n".format(telescope_pointing.Ty.value))
            header_file.write("CTYPE1  =HPLN-TAN\n")
            header_file.write("CTYPE2  =HPLT-TAN\n")
            header_file.write("CUNIT1  =arcsec\n")
            header_file.write("CUNIT2  =arcsec\n")
            header_file.write("CDELT1  ={0}\n".format(self.plate_scale[0]))
            header_file.write("CDELT2  ={0}\n".format(self.plate_scale[1]))
            header_file.write("CRPIX1  ={0}\n".format(self.imageshape[0]/2))
            header_file.write("CRPIX2  ={0}\n".format(self.imageshape[1]/2))
            header_file.write("CROTA2  ={0}\n".format(rotan))
            header_file.write("SCINT   ={0}\n".format(scin))
            header_file.write("LLVL    ={0}\n".format(llvl))
            header_file.write("RSUN_ARC={0}\n".format(solrad))

        header_file.close()

        return

    def rosa_zyla_save_cal_images(self) -> None:
        """
        Saves average dark, average flat, gain, and noise images
        in FITS format.
        """
        if os.path.exists(self.dark_file):
            self.logger.info("Dark file already exists: {}".format(self.dark_file))
        else:
            self.logger.info(
                "Saving average dark: {0}".format(os.path.join(self.workbase, self.dark_file))
            )
            self.rosa_zyla_save_fits_image(
                self.avg_dark,
                os.path.join(self.workbase, self.dark_file)
            )
        if os.path.exists(self.flat_file):
            self.logger.info("Flat file already exists: {0}".format(self.flat_file))
        else:
            self.logger.info(
                "Saving average flat: {0}".format(os.path.join(self.workbase, self.flat_file))
            )
            self.rosa_zyla_save_fits_image(
                self.avg_flat,
                os.path.join(self.workbase, self.flat_file)
            )
        if os.path.exists(self.gain_file):
            self.logger.info("Gain file already exists: {0}".format(self.gain_file))
        else:
            self.logger.info(
                "Saving gain: {0}".format(os.path.join(self.workbase, self.gain_file))
            )
            self.rosa_zyla_save_fits_image(
                self.gain,
                os.path.join(self.workbase, self.gain_file)
            )

        if os.path.exists(self.noise_file):
            self.logger.info("Noise KISIP file already exists: {0}".format(self.noise_file))
        else:
            self.logger.info(
                "Saving KISIP noise: {0}".format(os.path.join(self.workbase, self.noise_file))
            )
            self.rosa_zyla_compute_noise_file()

        if os.path.exists(self.noise_file_fits):
            self.logger.info("Noise FITS file already exists: {0}".format(self.noise_file_fits))
        else:
            self.logger.info("Saving noise: {0}".format(os.path.join(self.workbase, self.noise_file_fits)))
            self.rosa_zyla_save_fits_image(
                self.noise,
                os.path.join(self.workbase, self.noise_file_fits)
            )

        if self.avg_target is not None:
            if os.path.exists(self.target_file):
                self.logger.info("Air Force Resolution Target file already exists: {0}".format(self.target_file))
            else:
                self.logger.info(
                    "Saving AF Resolution Target: {0}".format(os.path.join(self.workbase, self.target_file))
                )
                self.rosa_zyla_save_fits_image(
                    self.avg_target,
                    os.path.join(self.workbase, self.target_file)
                )
        if self.avg_linegrid is not None:
            if os.path.exists(self.linegrid_file):
                self.logger.info("Line grid file already exists: {0}".format(self.linegrid_file))
            else:
                self.logger.info("Saving line grid: {0}".format(os.path.join(self.workbase, self.linegrid_file)))
                self.rosa_zyla_save_fits_image(
                    self.avg_linegrid,
                    os.path.join(self.workbase, self.linegrid_file)
                )

        return

    def rosa_zyla_save_despeckled_as_fits(self) -> None:
        """
        Saves despeckled (processed unformatted binary) KISIP images as
        FITS images.
        """
        self.logger.info("Saving despeckled binary image files to FITS.")
        self.logger.info(
            "Searching for files: {0}".format(
                os.path.join(
                    self.speckle_base,
                    self.speckle_file_form.format(self.obsdate, self.obstime, 0, 0)[:-7] + "*.final"
                )
            )
        )
        flist = glob.glob(os.path.join(
            self.speckle_base,
            self.speckle_file_form.format(self.obsdate, self.obstime, 0, 0)[:-7] + "*.final"
        ))
        flist_order = []
        for b in tqdm.tqdm(range(len(flist)), desc="Writing despeckled files", disable=not self.progress):
            proxy1 = flist[b].split("speckle.batch.")
            proxy2 = proxy1[1].split(".")
            flist_order.append(proxy2[0] + proxy2[1])
        flist_order = [int(s) for s in flist_order]
        header_prefix = self.prespeckle_base + "/" + self.obsdate + "_" + self.obstime
        header_list = glob.glob(header_prefix + "*.txt")
        ordered_header_list = []
        for b in range(len(header_list)):
            proxy1 = header_list[b].split("raw.batch.")
            proxy2 = proxy1[1].split(".")
            ordered_header_list.append(proxy2[0] + proxy2[1])
        ordered_header_list = [int(s) for s in ordered_header_list]

        try:
            assert (len(flist) != 0)
        except Exception as err:
            self.logger.critical("CRITICAL: no files found: {0}".format(err))
            raise
        else:
            self.logger.info("Found {0} files.".format(len(flist)))
        for i in range(len(flist)):
            im = self.rosa_zyla_read_binary_image(
                flist[i],
                image_shape=self.imageshape,
                data_shape=self.imageshape,
                dtype=np.float32
            )
            fname = os.path.basename(flist[i])
            my_file_index = ordered_header_list.index(flist_order[i])
            header_file = open(header_list[my_file_index], "r")
            self.rosa_zyla_save_fits_image(
                im,
                os.path.join(self.postspeckle_base, fname + ".fits"),
                header_file.readlines()
            )
            header_file.close()
        self.logger.info("Finished saving despeckled images as FITS in directory: {0}".format(self.postspeckle_base))
        return

    def rosa_zyla_save_fits_image(
            self,
            image: np.ndarray, file: str, header: dict | None=None,
            clobber: bool=True
        ) -> None:
        """
        Saves 2-dimensional image data to a FITS file.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional image data to save.
        file : str
            Path to file to save to.
        header : dict or None, optional
            If given, writes all key/value pairs in the dictionary to the header, default None
        clobber : bool
            Overwrite existing file if True, otherwise do not overwrite, default True.
        """
        hdu = fits.PrimaryHDU(image)
        if header is not None:
            for key in header.keys():
                hdu.header[key] = header[key]
        hdul = fits.HDUList([hdu])
        try:
            hdul.writeto(file, overwrite=clobber)
        except Exception:
            self.logger.warning("Could not write FITS file: {0}".format(file))
            self.logger.warning("FITS write warning: continuing, but this could cause problems later.")
        return
