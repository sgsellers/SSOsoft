import configparser
import glob
import os
import warnings

import astropy.io.fits as fits
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scinteg
import scipy.interpolate as scinterp
import scipy.io as scio
import scipy.ndimage as scind
import scipy.optimize as scopt
import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from . import spectraTools as spex
from .polarimetryTools import v2qu_crosstalk


class SpinorCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    SPINOR (SPectropolarimetry of INfrared and Optical Regions) data
    from the Dunn Solar Telescope. Note that at present moment, the
    spinorCal package is only compatible with the new SPINOR modulator
    installed by R. Casini and S. Sellers in September 2024.

    Data procured before this date should instead use the legacy IDL
    pipeline, developed by Christian Beck (spinor_soft).

    -------------------------------------------------------------------

    Use this software to process/reduce data from the SPINOR instrument at DST.
    To perform reductions using this package:
    1.) Install this package, along with the rest of SSOSoft.
    2.) Set the necessary instrument parameters in a configuration file.
        (Use the included spinor_sampleConfig.ini as a template)
    3.) Open a Python/iPython terminal and "from ssosoft.spectral import spinorCal"
    4.) Start an instance of the calibration class by using
        's=spinorCal('<CAMERA>', '<CONFIGFILE>')'
    5.) Run the standard calibration method using
        's.spinor_run_calibration()'

    Parameters
    ----------
    camera : str
        A string containing the camera name.
        Common values will be
            -SARNOFF1
            -SARNOFF2
            -FLIR1
            -FLIR2
            -SISJI
    config_file : str
        Path to the configuration file

    To my knowledge, a complete calibration should perform the following steps:
        1.) Create average solar flat, lamp flat, and dark files
        2.) Use the solar flat to determine the location of the two beams,
            as well as hairlines, and determine where to clip the beams
        3.) Flip the upper beam to match the lower, align the beams vertically,
            and determine the relative rotation of both beams.
        4.) Create a gaintable for each beam using deskew to iteratively detrend
            spectral profiles from the dark/lamp-corrected solar flat.
        5.) Read in polarization calibration measurements, and:
            a.) Correct for dark, lamp gain, and solar gain
            b.) Demodulate eight retarder rotation states into Stokes-IQUV
            c.) Determine positions of cal polarizer, cal retarder, telescope alt/az,
                values for light level for entire cal sequence.
            d.) Normalize by light level
            e.) Extrapolate telescope matrix to obs values.
            f.) Fit optical train Mueller matrix
        6.) Read in data files, and:
            a.) Correct dark, lamp gain, solar gain
            b.) Combine dual beams
            c.) Demodulate eight retarder states into Stokes-IQUV
            d.) Apply calibration Mueller matrix
            e.) Correct I->QUV crosstalk.
            f.) Deskew line profiles from flat fields
            g.) Tag hairline positions
            h.) Attempt prefilter correction
            i.) Attempt V->QU and QU->V crosstalk corrections
            j.) Attempt spectral fringe correction
            k.) Wavelength calibration via comparison to FTS atlas.
        7.) Create overview maps of moment analysis and polarization analysis
            quicklook parameters.
        8.) Write corrected raster and quicklook maps into FITS files
            a.) EXT 0: Header only
            b.) EXT 1: Stokes-I
            c.) EXT 2: Stokes-Q
            d.) EXT 3: Stokes-U
            e.) EXT 4: Stokes-V
            f.) EXT 5: lambda-coordinate
            g.) EXT 6: Metadata (alt, az, scin, llvl, etc.)
        9.) Stretch goal: Correct slit-jaw data, and create slit-jaw data cubes with
            slit position and hairlines marked for ease of alignment.
    """

    def __init__(self, camera: str, config_file: str) -> None:
        """

        Parameters
        ----------
        camera : str
            String containing camera name
        config_file : str
            Path to configuration file
        """

        try:
            f = open(config_file, 'r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.config_file = config_file
        self.camera = camera.upper()

        self.solar_dark = None
        self.solar_flat = None
        self.lamp_gain = None
        self.combined_gain_table = None
        self.combined_coarse_gain_table = None
        self.polcal_vecs = None
        self.t_matrix = None
        self.flip_wave = False

        # Locations
        self.indir = ""
        self.final_dir = ""
        self.reduced_file_pattern = None
        self.parameter_map_pattern = None
        self.polcal_file = None
        self.solar_flat_file_list = []
        self.solar_flat_file = None
        self.lamp_flat_file = None
        self.science_file_list = []
        # Need to re-combine longer map series that were split by SPINOR FITS daemons
        self.science_map_file_list = []
        self.science_files = None
        self.t_matrix_file = None

        self.line_grid_file = None
        self.target_file = None

        # For saving the reduced calibration files:
        self.solar_gain_reduced = None  # we'll include dark currents in these files
        self.lamp_gain_reduced = ""
        self.tx_matrix_reduced = ""

        # Setting up variables to be filled:
        self.beam_edges = None
        self.slit_edges = None
        self.hairlines = None
        self.beam1_xshift = None
        self.beam1_yshift = None
        self.spinor_line_cores = None
        self.fts_line_cores = None
        self.flip_wave_idx = 1
        self.analysis_ranges = None

        # Polcal-specific variables
        self.polcal_processing = True
        self.calcurves = None
        self.txmat = None
        self.input_stokes = None
        self.txchi = None
        self.txmat00 = None
        self.txmatinv = None

        # Some default vaules
        self.nhair = 4
        self.beam_threshold = 0.5
        self.hairline_width = 3
        self.grating_rules = 308.57  # lpmm
        self.blaze_angle = 52
        self.n_subslits = 10
        self.verbose = False
        self.v2q = True
        self.v2u = True
        self.u2v = True
        self.i2quv_residual = False
        self.despike = False
        self.despike_footprint = (1, 5, 1)
        self.plot = False
        self.save_figs = False
        self.crosstalk_continuum = None
        self.manual_hairline_selection = False
        self.manual_alignment_selection = False

        # Can be pulled from header:
        self.grating_angle = None
        self.slit_width = None

        # Must be set in config file, or surmised from config file
        self.pixel_size = None  # 16 um for Sarnoffs, 25 um for Flirs
        self.central_wavelength = None  # Should identify a spectral line, i.e., 6302, 8542
        self.spectral_order = None  # To-do, solve the most likely spectral order from the grating info. Function exists.

        # Default polarization modulation from new modulator (2024-09)
        # I: ++++++++
        # Q: --++--++
        # U: +--++--+
        # V: +----+++
        self.pol_demod = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, -1, -1, -1, 1, 1, 1]
        ], dtype=int)

        self.site_latitude = 32.786

        # Expressed as a fraction of mean(I). For polcals
        self.ilimit = [0.5, 1.5]
        # It's about a quarter-wave plate.
        self.cal_retardance = 90

        # Basic SPINOR Feed Optics
        self.slit_camera_lens = 780  # mm, f.l. of usual HSG feed lens
        self.telescope_plate_scale = 3.76  # asec/mm
        self.dst_collimator = 1559  # mm, f.l., of DST Port 4 Collimator mirror
        self.spectrograph_collimator = 3040  # mm, f.l., of the SPINOR/HSG post-slit collimator
        self.camera_lens = 1700  # mm, f.l., of the SPINOR final camera lenses

        return

    @staticmethod
    def spinor_assert_file_list(flist: list) -> None:
        assert (len(flist) != 0), "List contains no matches."

    def spinor_run_calibration(self) -> None:
        """Main SPINOR calibration module"""

        self.spinor_configure_run()
        if self.verbose:
            print("Found {0} science map files in base directory:\n{1}\nReduced files will be saved to:\n{2}".format(
                len(self.science_file_list), self.indir, self.final_dir
            ))
        for index in range(len(self.science_map_file_list)):
            if self.verbose:
                for subindex in range(len(self.science_map_file_list[index])):
                    if subindex == 0:
                        print(
                            "Proceeding with calibration of: {0}".format(
                                os.path.split(self.science_map_file_list[index][subindex])[1]
                            )
                        )
                    else:
                        print(
                            "                                {0}".format(
                                os.path.split(self.science_map_file_list[index][subindex])[1]
                            )
                        )
                print("Using Solar Flat File: {0}".format(os.path.split(self.solar_flat_file_list[index])[1]))
                if self.lamp_flat_file is not None:
                    print("Using Lamp Flat File: {0}".format(os.path.split(self.lamp_flat_file)[1]))
                else:
                    print("No Lamp Flat File")
                print("Using Polcal File: {0}".format(os.path.split(self.polcal_file)[1]))
                if self.plot:
                    print("Plotting is currently ON.")
                    if self.save_figs:
                        print("Plots will be saved at:\n{0}".format(self.final_dir))
                    else:
                        print("Plots will NOT be saved.")
                print("===========================\n\n")
            # We have to reset certain values that are calculated through the main loop.
            # Easiest way to do this is re-initialize the loop.
            # However, to avoid more user-interaction than strictly required, we do want to
            # exempt certain values from being overwritten. Namely, ones that were previously-selected.
            if index != 0:
                spinor_line_cores = self.spinor_line_cores
                fts_line_cores = self.fts_line_cores
                flip_wave = self.flip_wave
                analysis_ranges = self.analysis_ranges

            self.__init__(self.camera, self.config_file)
            self.spinor_configure_run()
            self.spinor_get_cal_images(index)
            if self.plot:
                plt.pause(2)
                plt.close("all")
            self.science_files = self.science_map_file_list[index]
            # And repopulate..
            if index != 0:
                self.spinor_line_cores = spinor_line_cores
                self.fts_line_cores = fts_line_cores
                self.flip_wave = flip_wave
                self.analysis_ranges = analysis_ranges
            self.reduce_spinor_maps(index)
        return

    def spinor_configure_run(self) -> None:
        """Reads configuration file and sets up parameters for calibration sequence"""

        """
        Block of file list find statements goes here.
        Rather than other systems, which, you know, work,
        SPINOR frequently fails to complete an observation.
        So instead of averaging multiple files together, we'll
        set up a single lamp/solar flat file and a single polcal file
        based on the sizes of the respective files, and set those up as the
        canonical calibration files. The user should be able to define a file
        pattern, i.e., *lamp_flat* or point to a specific file to override this
        behaviour.
        """
        self.spinor_parse_configfile()
        self.spinor_organize_directory()

        return

    def spinor_get_cal_images(self, index: int) -> None:
        """
        Loads or creates flat, dark, lamp & solar gain, polcal arrays
        """
        self.spinor_get_solar_flat(index)
        self.spinor_get_lamp_gain()
        self.spinor_get_solar_gain(index)
        self.spinor_get_polcal()

        return

    def spinor_parse_configfile(self) -> None:
        """
        Parses configureation file and sets up the class structure for reductions
        """
        config = configparser.ConfigParser()
        config.read(self.config_file)

        # Locations [Required]
        self.indir = config[self.camera]["rawFileDirectory"]
        self.final_dir = config[self.camera]["reducedFileDirectory"]
        self.reduced_file_pattern = config[self.camera]["reducedFilePattern"]
        self.parameter_map_pattern = config[self.camera]["reducedParameterMapPattern"]

        # Optional calibration file definitions. If these are left undefined, the directory parser below
        # sets these, however, it may be desirable to specify a flat file under certain circumstances.

        # [Optional] entry
        self.polcal_file = config[self.camera]["polcalFile"] if (
                "polcalfile" in config[self.camera].keys()
        ) else None
        # Reset to None for case where the key is present with an empty string
        self.polcal_file = None if self.polcal_file == "" else self.polcal_file
        self.solar_flat_file = config[self.camera]["solarFlatFile"] if (
                "solarflatfile" in config[self.camera].keys()
        ) else None
        self.solar_flat_file = None if self.solar_flat_file == "" else self.solar_flat_file
        self.lamp_flat_file = config[self.camera]["lampFlatFile"] if (
                "lampflatfile" in config[self.camera].keys()
        ) else None
        self.lamp_flat_file = None if self.lamp_flat_file == "" else self.lamp_flat_file
        self.science_files = [config[self.camera]["scienceFile"]] if (
                "sciencefile" in config[self.camera].keys()
        ) else None
        self.science_files = None if self.science_files == "" else self.science_files

        # Required channel-specific params
        self.pixel_size = 16 if "sarnoff" in self.camera.lower() else 25
        self.central_wavelength = float(config[self.camera]["centralWavelength"])
        self.spectral_order = int(config[self.camera]["spectralOrder"])

        # Overrides for some channel-specific default vaules
        self.n_subslits = int(config[self.camera]["slitDivisions"]) if (
                "slitdivisions" in config[self.camera].keys()
        ) else self.n_subslits
        self.verbose = config[self.camera]["verbose"] if "verbose" in config[self.camera].keys() else "False"
        if "t" in self.verbose.lower():
            self.verbose = True
        else:
            self.verbose = False
        self.v2q = config[self.camera]["v2q"] if "v2q" in config[self.camera].keys() else "True"
        if self.v2q.lower() == "true":
            self.v2q = True
        elif self.v2q.lower() == "full":
            self.v2q = "full"
        else:
            self.v2q = False

        self.v2u = config[self.camera]["v2u"] if "v2u" in config[self.camera].keys() else "True"
        if self.v2u.lower() == "true":
            self.v2u = True
        elif self.v2u.lower() == "full":
            self.v2u = "full"
        else:
            self.v2u = False
        self.u2v = config[self.camera]["u2v"] if "u2v" in config[self.camera].keys() else "True"
        if self.u2v.lower() == "true":
            self.u2v = True
        elif self.u2v.lower() == "full":
            self.u2v = "full"
        else:
            self.u2v = False
        self.i2quv_residual = config[
            self.camera['residualCrosstalk'] if "residualcrosstalk" in config[self.camera].keys() else "False"
        ]
        if self.i2quv_residual.lower() == "true":
            self.i2quv_residual = True
        else:
            self.i2quv_residual = False
        self.plot = config[self.camera]["plot"] if "plot" in config[self.camera].keys() else "False"
        if "t" in self.plot.lower():
            self.plot = True
        else:
            self.plot = False
        self.save_figs = config[self.camera]["savePlot"] if "saveplot" in config[self.camera].keys() else "False"
        if "t" in self.save_figs.lower():
            self.save_figs = True
        else:
            self.save_figs = False
        self.despike = config[self.camera]['despike'] if "despike" in config[self.camera].keys() else "False"
        if "t" in self.despike.lower():
            self.despike = True
        else:
            self.despike = False

        self.nhair = int(config[self.camera]["totalHairlines"]) if (
                "totalhairlines" in config[self.camera].keys()
        ) else self.nhair
        self.beam_threshold = float(config[self.camera]["intensityThreshold"]) if (
                "intensitythreshold" in config[self.camera].keys()
        ) else self.beam_threshold
        self.hairline_width = float(config[self.camera]["hairlineWidth"]) if (
                "hairlinewidth" in config[self.camera].keys()
        ) else self.hairline_width
        self.cal_retardance = float(config[self.camera]["calRetardance"]) if (
                "calretardance" in config[self.camera].keys()
        ) else self.cal_retardance
        self.camera_lens = float(config[self.camera]["cameraLens"]) if (
                "cameralens" in config[self.camera].keys()
        ) else self.camera_lens
        if "polcalclipthreshold" in config[self.camera].keys():
            if config[self.camera]['polcalClipThreshold'] != "":
                self.ilimit = [float(frac) for frac in config[self.camera]["polcalClipThresold"].split(",")]
        self.polcal_processing = config[self.camera]["polcalProcessing"] if (
                "polcalProcessing" in config[self.camera].keys()
        ) else "True"
        if "t" in self.polcal_processing.lower():
            self.polcal_processing = True
        else:
            self.polcal_processing = False

        # Case where someone wants the old crosstalk determination, and has defined it themselves
        if "crosstalkcontinuum" in config[self.camera].keys():
            if config[self.camera]['crosstalkContinuum'] != "":
                self.crosstalk_continuum = [int(idx) for idx in config[self.camera]['crosstalkContinuum'].split(",")]

        if "hairselect" in config[self.camera].keys():
            if config[self.camera]['hairSelect'].lower() == "true":
                self.manual_hairline_selection = True
        if "alignselect" in config[self.camera].keys():
            if config[self.camera]['hairSelect'].lower() == "true":
                self.manual_alignment_selection = True

        # Required global values
        self.t_matrix_file = config["SHARED"]["tMatrixFile"]

        # Overrides for Global defaults
        self.grating_rules = float(config["SHARED"]["gratingRules"]) if (
                "gratingrules" in config["SHARED"].keys()
        ) else self.grating_rules
        self.blaze_angle = float(config["SHARED"]["blazeAngle"]) if (
                "blazeangle" in config["SHARED"].keys()
        ) else self.blaze_angle

        self.site_latitude = float(config["SHARED"]["telescopeLatitude"]) if (
                "telescopelatitude" in config["SHARED"].keys()
        ) else self.site_latitude
        self.slit_camera_lens = float(config["SHARED"]["slitCameraLens"]) if (
                "slitcameralens" in config["SHARED"].keys()
        ) else self.slit_camera_lens
        self.telescope_plate_scale = float(config["SHARED"]["basePlateScale"]) if (
                "baseplatescale" in config["SHARED"].keys()
        ) else self.telescope_plate_scale
        self.dst_collimator = float(config["SHARED"]["telescopeCollimator"]) if (
                "telescopecollimator" in config["SHARED"].keys()
        ) else self.dst_collimator
        self.spectrograph_collimator = float(config["SHARED"]["spectrographCollimator"]) if (
                "spectrographcollimator" in config["SHARED"].keys()
        ) else self.spectrograph_collimator

        # Allow the user to define alternate QUV modulation pattern IFF
        # All of QUV modulation patterns are given.
        if (
                ("qmodulationpattern" in config["SHARED"].keys()) &
                ("umodulationpattern" in config["SHARED"].keys()) &
                ("vmodulationpattern" in config["SHARED"].keys())
        ):
            qmod = [int(mod) for mod in config["SHARED"]["qModulationPattern"].split(",")]
            umod = [int(mod) for mod in config["SHARED"]["uModulationPattern"].split(",")]
            vmod = [int(mod) for mod in config["SHARED"]["vModulationPattern"].split(",")]
            self.pol_demod = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1],
                qmod,
                umod,
                vmod
            ], dtype=int)

        return

    def spinor_organize_directory(self) -> None:
        """
        Organizes files in the Level-0 SPINOR directory.
        Pairs science maps with their nearest solar and lamp flat files chronologically.
            (Provided the flat has at least 21 data extensions)
        Finds the Target image map and Line Grid image map.
        """

        filelist = sorted(
            glob.glob(
                os.path.join(
                    self.indir,
                    "*.fits"
                )
            )
        )
        science_files = []
        solar_flats = []
        lamp_flats = []
        polcal_files = []
        line_grids = []
        target_files = []

        for file in filelist:
            with fits.open(file) as hdul:
                if ("USER1" in hdul[1].header['PT4_FS']) & ("map" in file):
                    science_files.append(file)
                elif ("USER2" in hdul[1].header['PT4_FS']) & ("map" in file):
                    line_grids.append(file)
                elif ("TARGET" in hdul[1].header['PT4_FS']) & ("map" in file):
                    target_files.append(file)
                elif ("sun.flat" in file) & (len(hdul) >= 16):
                    solar_flats.append(file)
                elif ("lamp.flat" in file) & (len(hdul) >= 16):
                    lamp_flats.append(file)
                elif ("cal" in file) & (len(hdul) >= 16):
                    polcal_files.append(file)
        # Select polcal, linegrid, and target files by total filesize
        if self.polcal_file is None:
            polcal_filesizes = np.array([os.path.getsize(pf) for pf in polcal_files])
            self.polcal_file = polcal_files[polcal_filesizes.argmax()] if len(polcal_filesizes) != 0 else None
        else:
            self.polcal_file = os.path.join(self.indir, self.polcal_file)
        line_grid_filesizes = np.array([os.path.getsize(lg) for lg in line_grids])
        self.line_grid_file = line_grids[line_grid_filesizes.argmax()] if len(line_grid_filesizes) != 0 else None
        target_filesizes = np.array([os.path.getsize(tg) for tg in target_files])
        self.target_file = target_files[target_filesizes.argmax()] if len(target_filesizes) != 0 else None

        # Case where a science file is defined, rather than allowing the code to run cals on the full day
        if self.science_files is not None:
            self.science_file_list = [os.path.join(self.indir, self.science_files)]
        else:
            self.science_file_list = science_files

        # 2025-01-27: Slight problem. SPINOR won't put more than 252 steps in a single file. Longer maps are split
        # across multiple files. It would be good to have these combined into a single file during reductions.
        # Unfortunately, nothing in the file headers indicates that this will happen. The only indicator is in the
        # filename. Each map in a series has the same map number in the filename, i.e., the file name is
        # YYMMDD.HHMMSS.0.cccXX.c-hrt.map.NNNN.fits
        # NNNN is the same for different halves of the map in the same series....
        # Get the map number...
        map_list = [x.split("c-hrt")[1] for x in self.science_file_list]
        # Deduplicate
        map_list = sorted(list(set(map_list)))
        self.science_map_file_list = [sorted(glob.glob(os.path.join(self.indir, "*" + x))) for x in map_list]
        science_start_times = np.array(
            [
                fits.open(x[0])[1].header['DATE-OBS'] for x in self.science_map_file_list
            ],
            dtype='datetime64[ms]'
        )

        # Allow user to override and choose flat files to use
        if self.solar_flat_file is not None:
            self.solar_flat_file_list = [os.path.join(self.indir, self.solar_flat_file)] * len(self.science_map_file_list)
            self.solar_gain_reduced = [os.path.join(
                self.final_dir, "{0}_{1}_SOLARGAIN.fits"
            ).format(self.camera, 0)] * len(self.science_map_file_list)
        else:
            solar_flat_start_times = np.array(
                [
                    fits.open(x)[1].header['DATE-OBS'] for x in solar_flats
                ],
                dtype='datetime64[ms]'
            )
            self.solar_flat_file_list = [
                solar_flats[spex.find_nearest(solar_flat_start_times, x)] for x in science_start_times
            ]
            self.solar_gain_reduced = [
                os.path.join(
                    self.final_dir,
                    "{0}_{1}_SOLARGAIN.fits"
                ).format(self.camera, spex.find_nearest(solar_flat_start_times, x)) for x in science_start_times
            ]
        if self.lamp_flat_file is None:
            lamp_flat_filesizes = np.array([os.path.getsize(lf) for lf in lamp_flats])
            self.lamp_flat_file = lamp_flats[lamp_flat_filesizes.argmax()] if len(lamp_flat_filesizes) != 0 else None
        else:
            self.lamp_flat_file = os.path.join(self.indir, self.lamp_flat_file)

        # Set up the list of reduced dark/gain/lamp/polcal files, so we can save out or restore previous calibrations

        self.lamp_gain_reduced = os.path.join(
            self.final_dir,
            "{0}_LAMPGAIN.fits"
        ).format(self.camera)
        self.tx_matrix_reduced = os.path.join(
            self.final_dir,
            "{0}_POLCAL.fits"
        ).format(self.camera)

        return

    def spinor_average_dark_from_hdul(self, hdulist: fits.HDUList) -> np.ndarray:
        """Computes an average dark image from a SPINOR fits file HDUList.
        Since SPINOR takes 4 dark frames at the start and end of a flat, lampflat,
        and polcal, but no separate dark files, the only way to get dark current is
        from these files.

        Parameters:
        -----------
        hdulist : astropy.io.fits.HDUList
            HDUList of the file that should have dark frames somewhere in it.

        Returns:
        --------
        average_dark : numpy.ndarray
            2D numpy array with the average dark current per pixel.
        """
        # hdulist might have leading empty extension.
        # Each extension has shape (8, ny, nx)
        # One image per mod state
        average_dark = np.zeros((hdulist[-1].data.shape[1], hdulist[-1].data.shape[2]))
        darkctr = 0
        for hdu in hdulist:
            if "PT4_FS" in hdu.header.keys():
                if "DARK" in hdu.header['PT4_FS']:
                    if self.despike:
                        data = self.despike_image(hdu.data, footprint=self.despike_footprint)
                        average_dark += np.nanmean(data, axis=0)
                    else:
                        average_dark += np.nanmean(hdu.data, axis=0)
                    darkctr += 1
        average_dark /= darkctr
        return average_dark

    def spinor_average_flat_from_hdul(self, hdulist: fits.HDUList) -> np.ndarray:
        """Computes an average flat image from a SPINOR fits file HDUList.
        Since SPINOR takes 4 dark frames at the start and end of a flat or lampflat,
        we need to process the whole file for flats and darks.

        Parameters:
        -----------
        hdulist : astropy.io.fits.HDUList
            FITS file HDUlist opened with astropy.io.fits.open()

        Returns:
        --------
        average_flat : numpy.ndarray
            Averaged flat field
        """
        # hdulist might have leading empty extension.
        # Each extension has shape (8, ny, nx)
        # One image per mod state
        average_flat = np.zeros((hdulist[-1].data.shape[1], hdulist[-1].data.shape[2]))
        flatctr = 0
        for hdu in hdulist:
            if "PT4_FS" in hdu.header.keys():
                if "DARK" not in hdu.header['PT4_FS']:
                    if self.despike:
                        data = self.despike_image(hdu.data, footprint=self.despike_footprint)
                        average_flat += np.nanmean(data, axis=0)
                    else:
                        average_flat += np.nanmean(hdu.data, axis=0)
                    flatctr += 1
        average_flat /= flatctr
        return average_flat

    def demodulate_spinor(self, poldata: np.ndarray) -> np.ndarray:
        """
        Applies demodulation, and returns 4-array of IQUV
        Parameters
        ----------
        poldata : numpy.ndarray
            3D array of polarization data. Should have the shape (8, ny, nx).

        Returns
        -------
        stokes : numpy.ndarray
            3D array of stokes data, of shape (4, ny, nx)
        """

        stokes = np.zeros((4, *poldata.shape[1:]))
        if self.despike:
            poldata = self.despike_image(poldata, footprint=self.despike_footprint)
        for i in range(stokes.shape[0]):
            for j in range(poldata.shape[0]):
                stokes[i] += self.pol_demod[i, j] * poldata[j, :, :]
        return stokes

    def spinor_get_lamp_gain(self) -> None:
        """
        Creates or loads the SPINOR lamp gain.
        If there's no reduced lamp gain file, creates one from Level-0.
        If there's no reduced lamp gain and no Level-0 lamp flat, creates a dummy array of ones

        Returns
        -------
        """
        # Restore previously-created lamp gain
        if os.path.exists(self.lamp_gain_reduced):
            with fits.open(self.lamp_gain_reduced) as hdu:
                self.lamp_gain = hdu[0].data
        # Create new lamp gain and save to file
        elif self.lamp_flat_file is not None:
            with fits.open(self.lamp_flat_file) as lhdul:
                lamp_dark = self.spinor_average_dark_from_hdul(lhdul)
                lamp_flat = self.spinor_average_flat_from_hdul(lhdul)
            cleaned_lamp_flat = self.clean_lamp_flat(lamp_flat - lamp_dark)
            self.lamp_gain = cleaned_lamp_flat / np.nanmedian(cleaned_lamp_flat)
            hdu = fits.PrimaryHDU(self.lamp_gain)
            hdu.header["DATE"] = np.datetime64("now").astype(str)
            hdu.header["COMMENT"] = "Created from file {0}".format(self.lamp_flat_file)
            fits.HDUList([hdu]).writeto(self.lamp_gain_reduced, overwrite=True)
        # No lamp gain available for the day. Creates an array of ones to mimic a lamp gain
        else:
            self.lamp_gain = np.ones(self.solar_dark.shape)
            warnings.warn("No lamp flat available. Reduced data may show strong internal fringes.")

        return

    def clean_lamp_flat(self, lamp_flat_image: np.ndarray) -> np.ndarray:
        """
        Cleans lamp flat image by finding hairlines and removing them via scipy.interpolate.griddata
        Uses spectraTools.detect_beams_hairlines to get hairlines, replace them with NaNs and interpolate
        over the NaNs.

        Unfortunately griddata is one of those scipy functions that's just... really slow.
        I'll keep researching faster methods. Might be worth modifying the solar gain routines to iteratively
        remove hairlines in the same manner that the solar spectrum is removed.

        Parameters
        ----------
        lamp_flat_image : numpy.ndarray
            Dark-subtracted lamp flat image

        Returns
        -------
        cleaned_lamp_flat_image : numpy.ndarray
            Lamp flat with hairlines removed

        """

        _, _, hairlines = spex.detect_beams_hairlines(
            lamp_flat_image,
            threshold=self.beam_threshold, hairline_width=self.hairline_width,
            expected_hairlines=self.nhair, expected_beams=2,
            fallback=True  # Hate relying on it, but safer for now
        )
        # Reset recursive counter since we'll need to use the function again later
        spex.detect_beams_hairlines.num_calls = 0
        for line in hairlines:
            # range + 1 to compensate for casting a float to an int, plus an extra 2-wide pad for edge effects
            lamp_flat_image[int(line - self.hairline_width - 2):int(line + self.hairline_width + 3), :] = np.nan
        x = np.arange(0, lamp_flat_image.shape[1])
        y = np.arange(0, lamp_flat_image.shape[0])
        masked_lamp_flat = np.ma.masked_invalid(lamp_flat_image)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~masked_lamp_flat.mask]
        y1 = yy[~masked_lamp_flat.mask]
        new_flat = masked_lamp_flat[~masked_lamp_flat.mask]
        cleaned_lamp_flat_image = scinterp.griddata(
            (x1, y1),
            new_flat.ravel(),
            (xx, yy),
            method='nearest',
            fill_value=np.nanmedian(lamp_flat_image)
        )
        return cleaned_lamp_flat_image

    def spinor_get_solar_gain(self, index: int) -> None:
        """
        Creates or loads a solar gain file.
        Parameters
        ----------
        index : int
            Since there's a possibility that there's a list of flat files, corrsponding to the one closest
            to the observation time, we need to be able to index the file list

        Returns
        -------

        """
        # Restore previously-created solar gain
        if os.path.exists(self.solar_gain_reduced[index]):
            with fits.open(self.solar_gain_reduced[index]) as hdu:
                self.combined_coarse_gain_table = hdu["COARSE-GAIN"].data
                self.combined_gain_table = hdu["GAIN"].data
                self.beam_edges = hdu["BEAM-EDGES"].data
                self.hairlines = hdu["HAIRLINES"].data
                self.slit_edges = hdu["SLIT-EDGES"].data
                self.beam1_yshift = hdu["BEAM1-SHIFTS"].data[0]
                self.beam1_xshift = hdu["BEAM1-SHIFTS"].data[1]
                self.spinor_line_cores = [hdu[0].header['LC1'], hdu[0].header['LC2']]
                self.fts_line_cores = [hdu[0].header['FTSLC1'], hdu[0].header["FTSLC2"]]
                self.flip_wave_idx = hdu[0].header["SPFLIP"]
                if self.flip_wave_idx == -1:
                    self.flip_wave = True
        else:
            self.spinor_create_solar_gain()
            self.spinor_save_gaintables(index)

        if self.plot:
            self.spinor_plot_gaintables(index)

        return

    def spinor_get_solar_flat(self, index: int) -> None:
        """
        Creates solar flat and solar dark from file
        Parameters
        ----------
        index

        Returns
        -------

        """
        if os.path.exists(self.solar_gain_reduced[index]):
            with fits.open(self.solar_gain_reduced[index]) as hdu:
                self.solar_flat = hdu["SOLAR-FLAT"].data
                self.solar_dark = hdu["SOLAR-DARK"].data
                self.slit_width = hdu[0].header["HSG_SLW"]
                self.grating_angle = hdu[0].header["HSG_GRAT"]
        else:
            self.solar_flat_file = self.solar_flat_file_list[index]
            with fits.open(self.solar_flat_file) as fhdul:
                self.solar_dark = self.spinor_average_dark_from_hdul(fhdul)
                self.solar_flat = self.spinor_average_flat_from_hdul(fhdul)
                self.slit_width = fhdul[1].header["HSG_SLW"]
                self.grating_angle = fhdul[1].header["HSG_GRAT"]
        return

    def spinor_create_solar_gain(self) -> None:
        """
        Creates solar gain tables from the indexed file in a list of viable flat files.
        This also determines the beam/slit edges, hairline, positions,
        and rough offsets between the upper and lower beams

        Returns
        -------

        """

        self.beam_edges, self.slit_edges, self.hairlines = spex.detect_beams_hairlines(
            self.solar_flat - self.solar_dark,
            threshold=self.beam_threshold,
            hairline_width=self.hairline_width,
            expected_hairlines=self.nhair,  # Possible that FLIR cams only have one hairline
            expected_beams=2,  # If there's only one beam, use hsgCal
            expected_slits=1,  # ...we're not getting a multislit unit for SPINOR.
            fallback=True  # FLIR cameras in particular have difficulties with the automated detection
        )
        self.slit_edges = self.slit_edges[0]
        if self.verbose:
            print("Lower Beam Edges in Y: ", self.beam_edges[0])
            print("Upper Beam Edges in Y: ", self.beam_edges[1])
            print("Shared X-range: ", self.slit_edges)
            print("There are {0} hairlines at ".format(self.nhair), self.hairlines)
            print("===========================\n\n")

        # Determine which beam is smaller, clip larger to same size
        smaller_beam = np.argmin(np.diff(self.beam_edges, axis=1))
        larger_beam = np.argmax(np.diff(self.beam_edges, axis=1))
        # If 4 hairlines are detected, clip the beams by the hairline positions.
        # This avoids overclipping the beam, and makes the beam alignment easier.
        # And if the beams are the same size, we can skip this next step.
        if (len(self.hairlines) == 4) and (smaller_beam != larger_beam):
            # It does matter which beam is smaller. Must pair inner & outer hairlines.
            if smaller_beam == 0:
                self.beam_edges[larger_beam, 0] = int(round(
                    self.hairlines[2] - (self.beam_edges[smaller_beam, 1] - self.hairlines[1]), 0
                ))
                self.beam_edges[larger_beam, 1] = int(round(
                    self.hairlines[3] + (self.hairlines[0] - self.beam_edges[smaller_beam, 0]), 0
                ))
            else:
                self.beam_edges[larger_beam, 0] = int(round(
                    self.hairlines[0] - (self.beam_edges[smaller_beam, 1] - self.hairlines[3]), 0
                ))
                self.beam_edges[larger_beam, 1] = int(round(
                    self.hairlines[1] + (self.hairlines[2] - self.beam_edges[smaller_beam, 0]), 0
                ))
        # Mainly for FLIR where one of the hairlines might be clipped by the chip's edge
        elif (len(self.hairlines) != 4) and (smaller_beam != larger_beam):
            self.beam_edges[larger_beam, 0] = int(
                np.nanmean(self.beam_edges[larger_beam, :]) - np.diff(self.beam_edges[smaller_beam, :]) / 2
            )
            self.beam_edges[larger_beam, 1] = int(
                np.nanmean(self.beam_edges[larger_beam, :]) + np.diff(self.beam_edges[smaller_beam, :]) / 2
            )

        # Might still be off by up to 2 in size due to errors in casting float to int
        diff = int(np.diff(np.diff(self.beam_edges, axis=1).flatten(), axis=0)[0])
        self.beam_edges[0, 1] += diff

        self.hairlines = self.hairlines.reshape(2, int(self.nhair / 2))

        beam0 = self.solar_flat[self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0]:self.slit_edges[1]]
        beam1 = np.flipud(
            self.solar_flat[self.beam_edges[1, 0]: self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]]
        )
        # Aligning the beams is tough with the type of mostly-1D structure in these spectral images
        # Rather than doing a 2D cross-correlation, we should average in both directions and do two 1D
        # cross correlations to determine the offset using numpy.correlate. Make sure mode='full'.
        # We'll also do the x-shift. It should be taken care of during deskew, but sometimes the shift is large
        yshift = np.correlate(
            (np.nanmean(beam0, axis=0) - np.nanmean(beam0)),
            (np.nanmean(beam1, axis=0) - np.nanmean(beam1)),
            mode='full'
        ).argmax() - beam0.shape[1]
        beam0meanprof = np.nanmean(beam0[int(beam0.shape[0] / 2 - 50):int(beam0.shape[0] / 2 + 50), 50:-50], axis=0)
        beam1meanprof = np.nanmean(beam1[int(beam1.shape[0] / 2 - 50):int(beam1.shape[0] / 2 + 50), 50:-50], axis=0)
        beam0meanprof -= beam0meanprof.mean()
        beam1meanprof -= beam1meanprof.mean()
        self.beam1_xshift = np.correlate(
            beam0meanprof, beam1meanprof, mode='full'
        ).argmax() - beam0meanprof.shape[0]

        # Rather than doing a scipy.ndimage.shift, the better way to do it would
        # be to shift the self.beamEdges for beam 1.
        # Need to be mindful of shifts that would move beamEdges out of frame.
        excess_shift = self.beam_edges[1, 1] - yshift - self.solar_flat.shape[0]
        if excess_shift > 0:
            # Y-Shift takes beam out of bounds. Update beamEdges to be at the edge of frame,
            # Store the excess shift for scipy.ndimage.shift at a later date.
            self.beam1_yshift = -excess_shift
            self.beam_edges[1] += -yshift - excess_shift
        else:
            # Y-shift does not take beam out of bound. Update beamEdges, and move on.
            self.beam1_yshift = 0
            self.beam_edges[1] += -yshift

        # Clean the hairlines out of the solar flat:
        cleaned_solar_flat = self.clean_lamp_flat(self.solar_flat.copy())  # We'll use this going forward

        # Redefine beams with new edges. Do not flip beam1, otherwise, everything has to flip
        beam0 = cleaned_solar_flat[self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0]:self.slit_edges[1]]
        beam1 = cleaned_solar_flat[self.beam_edges[1, 0]: self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]]

        # Now we need to grab the spectral line we'll be using for the gain table
        # Since this will require popping up a widget, we might as well fine-tune
        # our wavelength scale for the final product, so we don't have to do too
        # many widgets overall. Of course, to grab the right section of the FTS
        # atlas, we need a central wavelength and a wavelength scale... Rather than
        # Having the user figure this out, we can grab all the grating parameters all
        # at once.

        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens, slit_width=self.slit_width,
        )

        beam0_lamp_gain_corrected = (
                                            beam0 - self.solar_dark[self.beam_edges[0, 0]:self.beam_edges[0, 1],
                                                    self.slit_edges[0]:self.slit_edges[1]]
                                    ) / self.lamp_gain[self.beam_edges[0, 0]:self.beam_edges[0, 1],
                                        self.slit_edges[0]:self.slit_edges[1]]

        beam1_lamp_gain_corrected = (
                                            beam1 - self.solar_dark[self.beam_edges[1, 0]:self.beam_edges[1, 1],
                                                    self.slit_edges[0]:self.slit_edges[1]]
                                    ) / self.lamp_gain[self.beam_edges[1, 0]:self.beam_edges[1, 1],
                                        self.slit_edges[0]:self.slit_edges[1]]

        avg_profile = np.nanmedian(
            beam0_lamp_gain_corrected[
            int(beam0_lamp_gain_corrected.shape[0] / 2 - 30):int(beam0_lamp_gain_corrected.shape[0] / 2 + 30), :
            ],
            axis=0
        )

        if self.spinor_line_cores is None or self.fts_line_cores is None:
            # If these are defined elsewhere, we can refrain from selecting another window
            self.spinor_line_cores, self.fts_line_cores, self.flip_wave = self.spinor_fts_line_select(
                grating_params, avg_profile
            )
        if self.verbose & self.flip_wave:
            print("Spectrum flipped along the wavelength axis... Correcting.")
        if self.verbose and not self.flip_wave:
            print("Spectrum is not flipped, no correction necessary.")

        # Rather than building in logic every time we need to flip/not flip a spectrum,
        # We'll define a flip index, and slice by it every time. So if we flip, we'll be
        # indexing [::-1]. Otherwise, we'll index [::1], i.e., doing nothing to the array
        if self.flip_wave:
            self.flip_wave_idx = -1
        else:
            self.flip_wave_idx = 1

        beam0_gain_table, beam0_coarse_gain_table, beam0_skews = spex.create_gaintables(
            beam0_lamp_gain_corrected,
            [self.spinor_line_cores[0] - 7, self.spinor_line_cores[0] + 9],
            # hairline_positions=self.hairlines[0] - self.beamEdges[0, 0],
            neighborhood=12,
            hairline_width=self.hairline_width / 2
        )

        beam1_gain_table, beam1_coarse_gain_table, beam1_skews = spex.create_gaintables(
            beam1_lamp_gain_corrected,
            [self.spinor_line_cores[0] - 7 - self.beam1_xshift, self.spinor_line_cores[0] + 9 - self.beam1_xshift],
            # hairline_positions=self.hairlines[1] - self.beamEdges[1, 0],
            neighborhood=12,
            hairline_width=self.hairline_width / 2
        )

        self.combined_gain_table = np.ones(self.solar_flat.shape)
        self.combined_coarse_gain_table = np.ones(self.solar_flat.shape)

        self.combined_gain_table[
        self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0]:self.slit_edges[1]
        ] = beam0_gain_table
        self.combined_gain_table[
        self.beam_edges[1, 0]:self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]
        ] = beam1_gain_table

        self.combined_coarse_gain_table[
        self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0]:self.slit_edges[1]
        ] = beam0_coarse_gain_table
        self.combined_coarse_gain_table[
        self.beam_edges[1, 0]:self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]
        ] = beam1_coarse_gain_table

        return

    def spinor_plot_gaintables(self, index: int) -> None:
        """
        Helper method to plot gaintables in case the user wants to
            A.) Deal with just... so many popups
            B.) Check on the quality of the corrections as they go
        """
        aspect_ratio = self.solar_flat.shape[1] / self.solar_flat.shape[0]
        gain_fig = plt.figure("SPINOR Gain Tables", figsize=(4 * 2.5, 2.5 / aspect_ratio))
        ax_lamp = gain_fig.add_subplot(141)
        ax_flat = gain_fig.add_subplot(142)
        ax_coarse = gain_fig.add_subplot(143)
        ax_fine = gain_fig.add_subplot(144)
        ax_lamp.imshow(
            self.lamp_gain, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        corr_flat = (self.solar_flat - self.solar_dark) / self.lamp_gain
        ax_flat.imshow(
            corr_flat, origin='lower', cmap='gray',
            vmin=corr_flat.mean() - 2*np.std(corr_flat),
            vmax=corr_flat.mean() + 2*np.std(corr_flat)
        )
        ax_coarse.imshow(
            self.combined_coarse_gain_table, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_fine.imshow(
            self.combined_gain_table, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_lamp.set_title("LAMP GAIN")
        ax_flat.set_title("SOLAR FLAT")
        ax_coarse.set_title("COARSE GAIN")
        ax_fine.set_title("FINE GAIN")
        for beam in self.beam_edges.flatten():
            ax_lamp.axhline(beam, c="C1", linewidth=1)
            ax_flat.axhline(beam, c="C1", linewidth=1)
            ax_coarse.axhline(beam, c="C1", linewidth=1)
            ax_fine.axhline(beam, c="C1", linewidth=1)
        for hair in self.hairlines.flatten():
            ax_lamp.axhline(hair, c="C2", linewidth=1)
            ax_flat.axhline(hair, c="C2", linewidth=1)
            ax_coarse.axhline(hair, c="C2", linewidth=1)
            ax_fine.axhline(hair, c="C2", linewidth=1)
        for edge in self.slit_edges:
            ax_lamp.axvline(edge, c="C1", linewidth=1)
            ax_flat.axvline(edge, c="C1", linewidth=1)
            ax_coarse.axvline(edge, c="C1", linewidth=1)
            ax_fine.axvline(edge, c="C1", linewidth=1)
        gain_fig.tight_layout()
        if self.save_figs:
            filename = os.path.join(self.final_dir, "gain_tables_{0}.png".format(index))
            gain_fig.savefig(filename, bbox_inches="tight")

        plt.show(block=False)
        plt.pause(0.1)

        return

    def spinor_save_gaintables(self, index: int) -> None:
        """
        Writes gain tables to appropriate FITS files.
        Format on these is nonstandard. Has to contain dark current, coarse and fine gain, beam/slit edges, hairlines,
        locations of lines used in gain determination,
        Parameters
        ----------
        index : int

        Returns
        -------

        """
        # Only write if the file doesn't already exist.
        if os.path.exists(self.solar_gain_reduced[index]):
            if self.verbose:
                print("File exists: {0}\nSkipping file write.".format(self.solar_gain_reduced[index]))
            return

        phdu = fits.PrimaryHDU()
        phdu.header["DATE"] = np.datetime64("now").astype(str)
        phdu.header["LC1"] = self.spinor_line_cores[0]
        phdu.header["LC2"] = self.spinor_line_cores[1]
        phdu.header["FTSLC1"] = self.fts_line_cores[0]
        phdu.header["FTSLC2"] = self.fts_line_cores[1]
        phdu.header["SPFLIP"] = self.flip_wave_idx
        phdu.header["HSG_SLW"] = self.slit_width
        phdu.header["HSG_GRAT"] = self.grating_angle

        flat = fits.ImageHDU(self.solar_flat)
        flat.header["EXTNAME"] = "SOLAR-FLAT"
        dark = fits.ImageHDU(self.solar_dark)
        dark.header["EXTNAME"] = "SOLAR-DARK"
        cgain = fits.ImageHDU(self.combined_coarse_gain_table)
        cgain.header["EXTNAME"] = "COARSE-GAIN"
        fgain = fits.ImageHDU(self.combined_gain_table)
        fgain.header["EXTNAME"] = "GAIN"
        bedge = fits.ImageHDU(self.beam_edges)
        bedge.header["EXTNAME"] = "BEAM-EDGES"
        hairs = fits.ImageHDU(self.hairlines)
        hairs.header["EXTNAME"] = "HAIRLINES"
        slits = fits.ImageHDU(self.slit_edges)
        slits.header["EXTNAME"] = "SLIT-EDGES"
        shifts = fits.ImageHDU(np.array([self.beam1_yshift, self.beam1_xshift]))
        shifts.header["EXTNAME"] = "BEAM1-SHIFTS"

        hdul = fits.HDUList([phdu, flat, dark, cgain, fgain, bedge, hairs, slits, shifts])
        hdul.writeto(self.solar_gain_reduced[index], overwrite=True)

        return

    def spinor_fts_line_select(
            self, grating_params: np.rec.recarray, average_profile: np.ndarray
    ) -> tuple[list, list, bool]:
        """
        Pops up the line selection widget for gain table creation and wavelength determination
        Parameters
        ----------
        average_profile
        grating_params : numpy.records.recarray
            From spectraTools.grating_calculations

        Returns
        -------

        """
        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        apx_wavemin = self.central_wavelength - np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax = self.central_wavelength + np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemin -= 30 * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax += 30 * grating_params['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apx_wavemin, apx_wavemax)

        print("Top: SPINOR Spectrum (uncorrected). Bottom: FTS Reference Spectrum")
        print("Select the same two spectral lines on each plot.")
        spinor_lines, fts_lines = spex.select_lines_doublepanel(
            average_profile,
            fts_spec,
            4
        )
        spinor_line_cores = [
            int(spex.find_line_core(average_profile[x - 5:x + 5]) + x - 5) for x in spinor_lines
        ]
        fts_line_cores = [
            spex.find_line_core(fts_spec[x - 20:x + 9]) + x - 20 for x in fts_lines
        ]

        spinor_pix_per_fts_pix = np.abs(np.diff(spinor_line_cores)) / np.abs(np.diff(fts_line_cores))

        flip_wave = self.determine_spectrum_flip(
            fts_spec, average_profile, spinor_pix_per_fts_pix,
            spinor_line_cores, fts_line_cores
        )

        return spinor_line_cores, fts_line_cores, flip_wave

    def spinor_get_polcal(self) -> None:
        """
        Loads or creates SPINOR polcal

        Returns
        -------

        """
        if os.path.exists(self.tx_matrix_reduced):
            with fits.open(self.tx_matrix_reduced) as hdul:
                self.input_stokes = hdul["STOKES-IN"].data
                self.calcurves = hdul["CALCURVES"].data
                self.txmat = hdul["TXMAT"].data
                self.txchi = hdul["TXCHI"].data
                self.txmat00 = hdul["TX00"].data
                self.txmatinv = hdul["TXMATINV"].data
        else:
            self.spinor_polcal()
            self.spinor_save_polcal()
            if self.plot:
                self.spinor_plot_polcal()

        return

    def spinor_save_polcal(self) -> None:
        """
        Writes FITS file with SPINOR polcal parameters

        Returns
        -------

        """
        # Only write if the file doesn't already exist.
        if os.path.exists(self.tx_matrix_reduced):
            if self.verbose:
                print("File exists: {0}\nSkipping file write.".format(self.tx_matrix_reduced))
            return

        phdu = fits.PrimaryHDU()
        phdu.header["DATE"] = np.datetime64("now").astype(str)

        in_stoke = fits.ImageHDU(self.input_stokes)
        in_stoke.header["EXTNAME"] = "STOKES-IN"
        out_stoke = fits.ImageHDU(self.calcurves)
        out_stoke.header["EXTNAME"] = "CALCURVES"

        txmat = fits.ImageHDU(self.txmat)
        txmat.header["EXTNAME"] = "TXMAT"
        txchi = fits.ImageHDU(self.txchi)
        txchi.header["EXTNAME"] = "TXCHI"
        tx00 = fits.ImageHDU(self.txmat00)
        tx00.header["EXTNAME"] = "TX00"
        txinv = fits.ImageHDU(self.txmatinv)
        txinv.header["EXTNAME"] = "TXMATINV"

        hdul = fits.HDUList([phdu, in_stoke, out_stoke, txmat, txchi, tx00, txinv])
        hdul.writeto(self.tx_matrix_reduced, overwrite=True)

        return

    def spinor_polcal(self) -> None:
        """
        Performs polarization calibration on SPINOR data.

        Returns
        -------

        """

        # Get obsdate of science files to determine whether gain correction can be safely applied
        if os.path.exists(self.science_file_list[0]):
            with fits.open(self.science_file_list[0]) as hdul:
                base_obsdate = np.datetime64(hdul[1].header["DATE-OBS"], "D")
        else:
            base_obsdate = None

        polfile = fits.open(self.polcal_file)
        polfile_obsdate = np.datetime64(polfile[1].header["DATE-OBS"], "D")

        polcal_dark_current = self.spinor_average_dark_from_hdul(polfile)

        field_stops = [i.header['PT4_FS'] for i in polfile if "PT4_FS" in i.header.keys()]

        open_field_stops = [i for i in field_stops if "DARK" not in i]

        # Grab the ICU parameters for every non-dark frame
        polarizer_staged = np.array([
            1 if "IN" in i.header['PT4_PSTG'] else 0 for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        retarder_staged = np.array([
            1 if "IN" in i.header['PT4_RSTG'] else 0 for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        polarizer_angle = np.array([
            i.header['PT4_POL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        retarder_angle = np.array([
            i.header['PT4_RET'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        llvl_pol = np.array([
            i.header['DST_LLVL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        az_pol = np.array([
            i.header['DST_AZ'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        el_pol = np.array([
            i.header['DST_EL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        ta_pol = np.array([
            i.header['DST_TBL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])

        polcal_stokes_beams = np.zeros(
            (len(open_field_stops),
             2,
             4,
             self.beam_edges[0, 1] - self.beam_edges[0, 0],
             self.slit_edges[1] - self.slit_edges[0])
        )
        shift = self.beam1_yshift if self.beam1_yshift else 0
        xshift = self.beam1_xshift
        ctr = 0
        for hdu in polfile[1:]:
            if "DARK" not in hdu.header['PT4_FS']:
                # If the polcal was completed on the same date as the gain tables,
                # we can clean the polcals up with the gain to get a better estimate across the slit
                # If the polcals are from a different date, we should be content with dark-subtraction
                if (base_obsdate == polfile_obsdate) & self.polcal_processing:
                    data = self.demodulate_spinor(
                        (hdu.data - polcal_dark_current) / self.lamp_gain / self.combined_gain_table
                    )
                else:
                    data = self.demodulate_spinor(
                        (hdu.data - polcal_dark_current)
                    )

                # Cut out beams, flip n shift the upper beam
                polcal_stokes_beams[ctr, 0, :, :, :] = data[
                                                       :, self.beam_edges[0, 0]:self.beam_edges[0, 1],
                                                       self.slit_edges[0]:self.slit_edges[1]
                                                       ]
                polcal_stokes_beams[ctr, 1, :, :, :] = np.flip(
                    scind.shift(
                        data[
                        :, self.beam_edges[1, 0]:self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]
                        ], (0, shift, -xshift), order=1
                    ),
                    axis=1
                )
                ctr += 1

        # Close the polcal file
        polfile.close()

        merged_beams = np.zeros((
            polcal_stokes_beams.shape[0],
            polcal_stokes_beams.shape[2],
            polcal_stokes_beams.shape[3],
            polcal_stokes_beams.shape[4]
        ))
        merged_beams[:, 0, :, :] = np.nanmean(polcal_stokes_beams[:, :, 0, :, :], axis=1)
        merged_beams[:, 1:, :, :] = (polcal_stokes_beams[:, 0, 1:, :, :] - polcal_stokes_beams[:, 1, 1:, :, :]) / 2.
        # Additional step: mask values outside ilimit*mean(I).
        # Original spinor_cal.pro used [0.5, 1.5]
        # That's probably fine to first order.
        merged_beam_mask = (
                (merged_beams[:, 0, :, :] < self.ilimit[1] * np.nanmean(merged_beams[:, 0, :, :])) &
                (merged_beams[:, 0, :, :] > self.ilimit[0] * np.nanmean(merged_beams[:, 0, :, :]))
        )
        # Repeat the mask along the Stokes-axis
        merged_beam_mask = np.repeat(
            merged_beam_mask[:, np.newaxis, :, :], 4, axis=1
        )
        self.calcurves = np.zeros(
            (
                polcal_stokes_beams.shape[0],
                polcal_stokes_beams.shape[2],
                self.n_subslits
            )
        )
        subarrays = np.array_split(merged_beams, self.n_subslits, axis=2)
        submasks = np.array_split(merged_beam_mask, self.n_subslits, axis=2)

        for i in range(self.n_subslits):
            # Replace values outside ilimit with nans
            # That way, nanmean gets rid of them while preserving the array shape
            masked_subarray = subarrays[i]
            masked_subarray[~submasks[i]] = np.nan
            # Normalize QUV curve by I
            self.calcurves[:, 0, i] = np.nanmean(masked_subarray[:, 0, :, :], axis=(-2, -1))
            for j in range(1, 4):
                self.calcurves[:, j, i] = np.nanmean(
                    masked_subarray[:, j, :, :] / masked_subarray[:, 0, :, :], axis=(-2, -1)
                )

        # SPINOR IDL polcal takes measurements with LP+RET in beam,
        # Normalizes I to linear fit across. So I ranges ~0.75 -- 1.25
        # We'll modify this to use the full cal-set, but normalize with each set.
        # So norm of clear frames, sep norm for ret only, sep norm for lp+ret
        # Better method would be to normalize by clear frame I
        # Use the *whole* cow, y'know?
        # To do this, we'll need the 17th I measurements.
        # Or just grab the frames with PSTG and RSTG clear. Right.

        for i in range(self.n_subslits):
            clear_i_selection = ~polarizer_staged.astype(bool) & ~retarder_staged.astype(bool)
            clear_i = self.calcurves[:, 0, i][clear_i_selection]
            # Need an x-range that lines up with cal meas.
            clear_i_xrun = np.arange(self.calcurves.shape[0])[clear_i_selection]
            polyfit_clear = np.polyfit(clear_i_xrun, clear_i, 1)
            self.calcurves[:, 0, i][
                clear_i_selection
            ] = self.calcurves[:, 0, i][clear_i_selection] / (
                    np.arange(self.calcurves.shape[0]) * polyfit_clear[0] + polyfit_clear[1]
            )[clear_i_selection]

            ret_only_selection = ~polarizer_staged.astype(bool) & retarder_staged.astype(bool)
            ret_only = self.calcurves[:, 0, i][ret_only_selection]
            ret_only_xrun = np.arange(self.calcurves.shape[0])[ret_only_selection]
            polyfit_ret = np.polyfit(ret_only_xrun, ret_only, 1)
            self.calcurves[:, 0, i][ret_only_selection] = self.calcurves[:, 0, i][ret_only_selection] / (
                    np.arange(self.calcurves.shape[0]) * polyfit_ret[0] + polyfit_ret[1]
            )[ret_only_selection]

            lpret_selection = polarizer_staged.astype(bool) & retarder_staged.astype(bool)
            lpret = self.calcurves[:, 0, i][lpret_selection]
            lpret_xrun = np.arange(self.calcurves.shape[0])[lpret_selection]
            polyfit_lp_ret = np.polyfit(lpret_xrun, lpret, 1)
            self.calcurves[:, 0, i][lpret_selection] = self.calcurves[:, 0, i][lpret_selection] / (
                    np.arange(self.calcurves.shape[0]) * polyfit_lp_ret[0] + polyfit_lp_ret[1]
            )[lpret_selection]

            lp_only_selection = polarizer_staged.astype(bool) & ~retarder_staged.astype(bool)
            lp_only = self.calcurves[:, 0, i][lp_only_selection]
            lp_only_xrun = np.arange(self.calcurves.shape[0])[lp_only_selection]
            polyfit_lp = np.polyfit(lp_only_xrun, lp_only, 1)
            self.calcurves[:, 0, i][lp_only_selection] = self.calcurves[:, 0, i][lp_only_selection] / (
                    np.arange(self.calcurves.shape[0]) * polyfit_lp[0] + polyfit_lp[1]
            )[lp_only_selection]

        self.calcurves = np.nan_to_num(self.calcurves)

        # Now create the input Stokes Vectors using the Telescope Matrix, plus cal unit parameters.
        # The cal train is Sky -> Telescope -> Polarizer -> Retarder -> Spectrograph
        input_stokes = np.zeros((self.calcurves.shape[0], 4))
        for i in range(self.calcurves.shape[0]):
            init_stokes = np.array([1, 0, 0, 0])
            tmtx = self.get_telescope_matrix([az_pol[i], el_pol[i], ta_pol[i]], 90)
            init_stokes = tmtx @ init_stokes
            if bool(polarizer_staged[i]):
                # Mult by 2 since we normalized our intensities earlier...
                init_stokes = 2 * spex.linear_analyzer_polarizer(
                    polarizer_angle[i] * np.pi / 180,
                    px=1,
                    py=0.005  # Estimate...
                ) @ init_stokes
            if bool(retarder_staged[i]):
                init_stokes = spex.linear_retarder(
                    retarder_angle[i] * np.pi / 180,
                    self.cal_retardance * np.pi / 180
                ) @ init_stokes
            input_stokes[i, :] = init_stokes

        self.input_stokes = np.nan_to_num(input_stokes)
        self.txmat = np.zeros((self.n_subslits, 4, 4))
        self.txchi = np.zeros(self.n_subslits)
        self.txmat00 = np.zeros(self.n_subslits)
        self.txmatinv = np.zeros((self.n_subslits, 4, 4))

        for i in range(self.n_subslits):
            errors, xmat = spex.matrix_inversion(
                input_stokes,
                self.calcurves[:, :, i]
            )
            self.txmat00[i] = xmat[0, 0]
            xmat /= xmat[0, 0]
            self.txmat[i] = xmat
            self.txmatinv[i] = np.linalg.inv(xmat)
            efficiencies = np.sqrt(np.sum(xmat ** 2, axis=1))
            #
            #     # Measurement of retardance from +-QU measurements
            #     # +Q
            lp_pos_q_selection = (
                    (polarizer_staged.astype(bool) &
                     ~retarder_staged.astype(bool)) &
                    ((np.abs(polarizer_angle) < 1) |
                     (np.abs(polarizer_angle - 180) < 1))
            )
            pos_q_vec = self.calcurves[:, :, i][np.repeat(lp_pos_q_selection[:, np.newaxis], 4, axis=1)]
            pos_q_vec = np.nanmean(
                pos_q_vec.reshape(int(pos_q_vec.shape[0] / 4), 4),
                axis=0
            )
            stokes_pos_q = self.txmatinv[i] @ pos_q_vec
            stokes_pos_q = stokes_pos_q / stokes_pos_q[0]
            pos_qsqdiff = np.sum((stokes_pos_q - np.array([1, 1, 0, 0])) ** 2)
            #     # -Q
            lp_neg_q_selection = (
                    (polarizer_staged.astype(bool) &
                     ~retarder_staged.astype(bool)) &
                    ((np.abs(polarizer_angle - 90) < 1) |
                     (np.abs(polarizer_angle - 270) < 1))
            )
            neg_q_vec = self.calcurves[:, :, i][np.repeat(lp_neg_q_selection[:, np.newaxis], 4, axis=1)]
            neg_q_vec = np.nanmean(
                neg_q_vec.reshape(int(neg_q_vec.shape[0] / 4), 4),
                axis=0
            )
            stokes_neg_q = self.txmatinv[i] @ neg_q_vec
            stokes_neg_q = stokes_neg_q / stokes_neg_q[0]
            neg_qsqdiff = np.sum((stokes_neg_q - np.array([1, -1, 0, 0])) ** 2)
            #     # +U
            lp_pos_u_selection = (
                    (polarizer_staged.astype(bool) &
                     ~retarder_staged.astype(bool)) &
                    ((np.abs(polarizer_angle - 45) < 1) |
                     (np.abs(polarizer_angle - 225) < 1))
            )
            pos_u_vec = self.calcurves[:, :, i][np.repeat(lp_pos_u_selection[:, np.newaxis], 4, axis=1)]
            pos_u_vec = np.nanmean(
                pos_u_vec.reshape(int(pos_u_vec.shape[0] / 4), 4),
                axis=0
            )
            stokes_pos_u = self.txmatinv[i] @ pos_u_vec
            stokes_pos_u = stokes_pos_u / stokes_pos_u[0]
            pos_usqdiff = np.sum((stokes_pos_u - np.array([1, 0, 1, 0])) ** 2)
            #     # -U
            lp_neg_u_selection = (
                    (polarizer_staged.astype(bool) &
                     ~retarder_staged.astype(bool)) &
                    ((np.abs(polarizer_angle - 135) < 1) |
                     (np.abs(polarizer_angle - 315) < 1))
            )
            neg_u_vec = self.calcurves[:, :, i][np.repeat(lp_neg_u_selection[:, np.newaxis], 4, axis=1)]
            neg_u_vec = np.nanmean(
                neg_u_vec.reshape(int(neg_u_vec.shape[0] / 4), 4),
                axis=0
            )
            stokes_neg_u = self.txmatinv[i] @ neg_u_vec
            stokes_neg_u = stokes_neg_u / stokes_neg_u[0]
            neg_usqdiff = np.sum((stokes_neg_u - np.array([1, 0, -1, 0])) ** 2)

            self.txchi[i] = pos_qsqdiff + neg_qsqdiff + pos_usqdiff + neg_usqdiff

            if self.verbose:
                print("TX Matrix:")
                print(xmat)
                print("Inverse:")
                print(self.txmatinv[i])
                print("Efficiencies:")
                print(
                    "Q: " + str(round(efficiencies[1], 4)),
                    "U: " + str(round(efficiencies[2], 4)),
                    "V: " + str(round(efficiencies[3], 4))
                )
                print("Average Deviation of cal vectors: ", np.sqrt(self.txchi[i]) / 4)
                print("===========================\n\n")

            # Check physicality & Efficiencies:
            if np.nanmax(efficiencies[1:]) > 0.866:
                name = ['Q ', 'U ', 'V ']
                warnings.warn(
                    str(name[efficiencies[1:].argmax()]) +
                    "is likely too high with a value of " +
                    str(efficiencies[1:].max())
                )
            mueller_check = spex.check_mueller_physicality(xmat)
            if not mueller_check[0]:
                print(
                    ("WARNING: TX Matrix for section {0} of {1}\nIs an unphysical " +
                     "Mueller matrix with output minimum I:\n{2},\n" +
                     "and output minimum I^2 - (Q^2 + U^2 + V^2):\n{3}").format(
                        i, self.n_subslits, mueller_check[1], mueller_check[2]
                    )
                )

        return

    def spinor_plot_polcal(self) -> None:
        """
        For the lovers of screen clutter, plots the polcal input and output vectors

        Returns
        -------

        """

        # Do plotting
        polcal_fig = plt.figure("Polcal Results", figsize=(8, 4))
        # Create 3 columns, 4 rows.
        # Column 1: Calcurves IQUV
        # Column 2: Input Stokes Vectors IQUV
        # Column 3: Output Stokes Vectors IQUV
        gs = polcal_fig.add_gridspec(ncols=3, nrows=4)

        out_stokes = np.array([self.txmat @ self.input_stokes[j, :] for j in range(self.input_stokes.shape[0])])
        names = ['I', 'Q', 'U', 'V']
        for i in range(4):
            ax_ccurve = polcal_fig.add_subplot(gs[i, 0])
            ax_incurve = polcal_fig.add_subplot(gs[i, 1])
            ax_outcurve = polcal_fig.add_subplot(gs[i, 2])
            # Column Titles
            if i == 0:
                ax_ccurve.set_title("POLCAL CURVES")
                ax_incurve.set_title("INPUT VECTORS")
                ax_outcurve.set_title("FIT VECTORS")
            # Plot statements. Default is fine.
            for j in range(self.n_subslits):
                ax_ccurve.plot(self.calcurves[:, i, j])
                ax_outcurve.plot(out_stokes[:, j, i])
            ax_incurve.plot(self.input_stokes[:, i])
            # Clip to x range of [0, end]
            ax_ccurve.set_xlim(0, self.calcurves.shape[0])
            ax_incurve.set_xlim(0, self.calcurves.shape[0])
            ax_outcurve.set_xlim(0, self.calcurves.shape[0])
            # Clip to common y range defined by max/min of all 3 columns
            ymax = np.array(
                [self.calcurves[:, i, :].max(), out_stokes[:, :, i].max(), self.input_stokes[:, i].max()]
            ).max()
            ymin = np.array(
                [self.calcurves[:, i, :].min(), out_stokes[:, :, i].min(), self.input_stokes[:, i].min()]
            ).min()
            ax_ccurve.set_ylim(ymin, ymax)
            ax_incurve.set_ylim(ymin, ymax)
            ax_outcurve.set_ylim(ymin, ymax)
            # Set up row label
            ax_ccurve.set_ylabel(names[i])
            # Reduce number of ticks to 3 for each y axis, 4 for x axis
            ax_ccurve.locator_params(axis="x", tight=True, nbins=4)
            ax_ccurve.locator_params(axis="y", tight=True, nbins=4)
            # Turn off tick labels for all except the first column in y, and the last row in x
            ax_incurve.set_yticklabels([])
            ax_outcurve.set_yticklabels([])
            if i != 3:
                ax_ccurve.set_xticklabels([])
                ax_incurve.set_xticklabels([])
                ax_outcurve.set_xticklabels([])
        polcal_fig.tight_layout()
        if self.save_figs:
            filename = os.path.join(self.final_dir, "polcal_curves.png")
            polcal_fig.savefig(filename, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)

        return

    def reduce_spinor_maps(self, index: int) -> None:
        """
        Performs reduction of SPINOR science data maps.
        Applies Dark Current, Lamp Gain, Solar Gain Corrections
        Applies inverse spectrograph correction matric and telescope correction matrix to data.
        Corrects QU for parallactic angle.
        Performs simple I->QUV crosstalk estimation, and (optionally) V<-->QU crosstalk estimation.

        Paramters
        ---------
        index: int
            Index of the map currently being reduced. Used in naming the crosstalk file

        Returns
        -------

        """
        total_slit_positions = 0
        for i in range(len(self.science_files)):
            with fits.open(self.science_files[i]) as hdul:
                total_slit_positions += len(hdul) - 1
        if self.verbose:
            print("{0} Slit Positions Observed in Sequence".format(total_slit_positions))
        # Check for existence of output file:
        with fits.open(self.science_files[0]) as hdul:
            date, time = hdul[1].header['DATE-OBS'].split("T")
            date = date.replace("-", "")
            time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
            outname = self.reduced_file_pattern.format(
                date,
                time,
                total_slit_positions
            )
            outfile = os.path.join(self.final_dir, outname)
            if os.path.exists(outfile):
                remake_file = input("File: {0}\nExists. (R)emake or (C)ontinue?  ".format(outname))
                if ("c" in remake_file.lower()) or (remake_file.lower() == ""):
                    plt.pause(2)
                    plt.close("all")
                    return
                elif ("r" in remake_file.lower()) and self.verbose:
                    print("Remaking file with user-specified corrections. This may take some time.")

        reduced_data = np.zeros((
            total_slit_positions,
            4,
            self.beam_edges[0, 1] - self.beam_edges[0, 0],
            self.slit_edges[1] - self.slit_edges[0]
        ))
        complete_i2_quv_crosstalk = np.zeros((
            total_slit_positions,
            3, 2, self.beam_edges[0, 1] - self.beam_edges[0, 0]
        ))
        complete_internal_crosstalks = np.zeros((
            total_slit_positions,
            3,
            self.beam_edges[0, 1] - self.beam_edges[0, 0]
        ))
        shift = self.beam1_yshift if self.beam1_yshift else 0

        # Interpolate telescope inverse matrix to entire slit from nsubslits
        xinv_interp = scinterp.CubicSpline(
            np.linspace(0, self.beam_edges[0, 1] - self.beam_edges[0, 0], self.n_subslits),
            self.txmatinv,
            axis=0,
        )(np.arange(0, self.beam_edges[0, 1] - self.beam_edges[0, 0]))

        step_index = 0
        # Setting these up for later
        master_hairline_centers = (0, 0)
        master_spectral_line_centers = (0, 0)
        with tqdm.tqdm(
                total=total_slit_positions,
                desc="Reducing Science Map"
        ) as pbar:
            for file in self.science_files:
                science_hdu = fits.open(file)
                for i in range(1, len(science_hdu)):
                    iquv = self.demodulate_spinor(
                        (science_hdu[i].data - self.solar_dark) / self.lamp_gain / self.combined_gain_table
                    )
                    science_beams = np.zeros(
                        (2, 4, self.beam_edges[0, 1] - self.beam_edges[0, 0], self.slit_edges[1] - self.slit_edges[0])
                    )
                    science_beams[0, :, :, :] = iquv[
                                               :,
                                                self.beam_edges[0, 0]:self.beam_edges[0, 1],
                                                self.slit_edges[0]:self.slit_edges[1]
                                               ]
                    science_beams[1, :, :, :] = np.flip(
                        scind.shift(
                            iquv[
                            :,
                            self.beam_edges[1, 0]:self.beam_edges[1, 1],
                            self.slit_edges[0]:self.slit_edges[1]
                            ], (0, shift, self.beam1_xshift), order=1
                        ), axis=1
                    )

                    # Reference beam for hairline/spectral line deskew shouldn't have full gain
                    # correction done, due to hairline residuals. It *should* be safe to use the
                    # lamp gain, as the hairlines in that should've been cleaned up.
                    alignment_beam = (np.mean(science_hdu[i].data, axis=0) - self.solar_dark) / self.lamp_gain
                    if step_index == 0:
                        hairline_skews, hairline_centers = self.subpixel_hairline_align(
                            alignment_beam, hair_centers=None
                        )
                        if self.nhair == 2:
                            if hairline_centers[0] < self.beam_edges[0, 1] - self.beam_edges[0, 0]:
                                # Only hairline is at the bottom of the beam. Set dummy second hairline at top.
                                master_hairline_centers = (hairline_centers[0],
                                                           self.beam_edges[0, 1] - self.beam_edges[0 ,0])
                            else:
                                # Only hairline is at the top of the beam. Set a dummy second hairline at bottom.
                                master_hairline_centers = (hairline_centers[0], 0)
                        else:
                            master_hairline_centers = (hairline_centers[0],
                                                     hairline_centers[0] + np.diff(self.hairlines, axis=1)[0][0])

                    else:
                        hairline_skews, hairline_centers = self.subpixel_hairline_align(
                            alignment_beam, hair_centers=np.array(hairline_centers)
                        )
                    # Perform hairline deskew
                    for beam in range(science_beams.shape[0]):
                        for hairProf in range(science_beams.shape[3]):
                            science_beams[beam, :, :, hairProf] = scind.shift(
                                science_beams[beam, :, :, hairProf],
                                (0, hairline_skews[beam, hairProf]),
                                mode='nearest', order=1
                            )
                    # Perform bulk hairline alignment on deskewed beams
                    science_beams[1] = scind.shift(
                        science_beams[1], (0, -np.diff(hairline_centers)[0], 0),
                        mode='nearest', order=1
                    )

                    if step_index == 0:
                        # Perform spectral deskew
                        science_beams, spectral_centers, spex_ranges = self.subpixel_spectral_align(
                            science_beams, hairline_centers
                        )
                    else:
                        science_beams, spectral_centers, spex_ranges = self.subpixel_spectral_align(
                            science_beams, hairline_centers, spectral_ranges=spex_ranges
                        )

                    # Perform bulk spectral alignment on deskewed beams
                    science_beams[1] = scind.shift(
                        science_beams[1], (0, 0, -np.diff(spectral_centers)[0]),
                        mode='nearest', order=1
                    )

                    # Common positions to register observation to.
                    if step_index == 0:
                        master_spectral_line_centers = spectral_centers
                    # Perform master registration to 0th slit image.
                    science_beams = scind.shift(
                        science_beams, (
                            0, 0,
                            -(hairline_centers[0] - master_hairline_centers[0]), 0 # Use the 0th master_hairline_center
                        ),
                        mode='nearest', order=1
                    )
                    norm_factor = np.mean(science_beams[0, 0, 50:-50]) / np.mean(science_beams[1, 0, 50:-50])
                    science_beams[1] *= norm_factor

                    combined_beams = np.zeros(science_beams.shape[1:])
                    combined_beams[0] = np.nanmean(science_beams[:, 0, :, :], axis=0)
                    combined_beams[1:] = (
                                                 science_beams[0, 1:, :, :] - science_beams[1, 1:, :, :]
                                         ) / 2
                    tmtx = self.get_telescope_matrix(
                        [science_hdu[i].header['DST_AZ'],
                         science_hdu[i].header['DST_EL'],
                         science_hdu[i].header['DST_TBL']],
                        180
                    )
                    inv_tmtx = np.linalg.inv(tmtx)
                    for j in range(combined_beams.shape[1]):
                        combined_beams[:, j, :] = np.nan_to_num(
                            inv_tmtx @ xinv_interp[j, :, :] @ combined_beams[:, j, :]
                        )

                    # Get parallactic angle for QU rotation correction
                    angular_geometry = self.spherical_coordinate_transform(
                        [science_hdu[i].header['DST_AZ'], science_hdu[i].header['DST_EL']]
                    )
                    # Sub off P0 angle
                    rotation = np.pi + angular_geometry[2] - science_hdu[i].header['DST_PEE'] * np.pi / 180
                    crot = np.cos(-2 * rotation)
                    srot = np.sin(-2 * rotation)

                    # Make a copy, as the Q/U components are transformed from the originals.
                    qtmp = combined_beams[1, :, :].copy()
                    utmp = combined_beams[2, :, :].copy()
                    combined_beams[1, :, :] = crot * qtmp + srot * utmp
                    combined_beams[2, :, :] = -srot * qtmp + crot * utmp

                    combined_beams, i2quv_crosstalk, internal_crosstalks = self.solve_spinor_crosstalks(
                        combined_beams
                    )

                    # Reverse the wavelength axis if required.
                    combined_beams = combined_beams[:, :, ::self.flip_wave_idx]

                    reduced_data[step_index] = combined_beams
                    complete_i2_quv_crosstalk[step_index] = i2quv_crosstalk
                    complete_internal_crosstalks[step_index] = internal_crosstalks

                    # Choose lines for analysis. Use same method of choice as hsgCal, where user sets
                    # approx. min/max, the code changes the bounds, and
                    if (step_index == 0) and (self.analysis_ranges is None):
                        mean_profile = np.nanmean(combined_beams[0], axis=0)
                        approx_wavelength_array = self.tweak_wavelength_calibration(mean_profile)
                        print("Select spectral ranges (xmin, xmax) for overview maps. Close window when done.")
                        # Approximate indices of line cores
                        coarse_indices = spex.select_spans_singlepanel(
                            mean_profile, xarr=approx_wavelength_array, fig_name="Select Lines for Analysis"
                        )
                        # Location of minimum in the range
                        min_indices = [
                            spex.find_nearest(
                                mean_profile[coarse_indices[x][0]:coarse_indices[x][1]],
                                mean_profile[coarse_indices[x][0]:coarse_indices[x][1]].min()
                            ) + coarse_indices[x][0] for x in range(coarse_indices.shape[0])
                        ]
                        # Location of exact line core
                        line_cores = [
                            spex.find_line_core(mean_profile[x - 5:x + 7]) + x - 5 for x in min_indices
                        ]
                        # Find start and end indices that put line cores at the center of the window.
                        self.analysis_ranges = np.zeros(coarse_indices.shape)
                        for j in range(coarse_indices.shape[0]):
                            average_delta = np.mean(np.abs(coarse_indices[j, :] - line_cores[j]))
                            self.analysis_ranges[j, 0] = int(round(line_cores[j] - average_delta, 0))
                            self.analysis_ranges[j, 1] = int(round(line_cores[j] + average_delta, 0) + 1)
                    elif (step_index == 0) and (self.analysis_ranges is not None):
                        mean_profile = np.nanmean(combined_beams[0], axis=0)
                        self.analysis_ranges = self.analysis_ranges.astype(int)
                        line_cores = [
                            spex.find_line_core(
                                mean_profile[self.analysis_ranges[x, 0]:self.analysis_ranges[x, 1]]
                            ) + self.analysis_ranges[x, 0] for x in range(self.analysis_ranges.shape[0])
                        ]

                    if self.plot:
                        plt.ion()
                        # Set up overview maps to blit new data into.
                        # Need maps for the slit images (IQUV) that are replaced at each step,
                        # As well as IQUV maps of the full field for each line selected.
                        # These latter will be filled as the map is processed.
                        if step_index == 0:
                            field_images = np.zeros((
                                len(line_cores),  # Number of lines
                                4,  # Stokes-IQUV values
                                combined_beams.shape[1],
                                total_slit_positions
                            ))
                        for j in range(len(line_cores)):
                            field_images[j, 0, :, step_index] = combined_beams[0, :, int(round(line_cores[j], 0))]
                            for k in range(1, 4):
                                field_images[j, k, :, step_index] = scinteg.trapezoid(
                                    np.nan_to_num(
                                        combined_beams[
                                            k, :, int(self.analysis_ranges[j, 0]):int(self.analysis_ranges[j, 1])
                                        ] /
                                        combined_beams[
                                            0, :, int(self.analysis_ranges[j, 0]):int(self.analysis_ranges[j, 1])
                                        ]
                                    ),
                                    axis=-1
                                )
                        if step_index == 0:
                            slit_plate_scale = self.telescope_plate_scale * self.dst_collimator / self.slit_camera_lens
                            camera_dy = slit_plate_scale * (self.spectrograph_collimator / self.camera_lens) * (
                                    self.pixel_size / 1000)
                            map_dx = science_hdu[1].header['HSG_STEP']

                            plot_params = self.set_up_live_plot(
                                field_images, combined_beams, internal_crosstalks, camera_dy, map_dx
                            )
                        self.update_live_plot(
                            *plot_params, field_images, combined_beams, internal_crosstalks, step_index
                        )
                    step_index += 1
                    pbar.update(1)
                science_hdu.close()

        # Save final plots if applicable
        if self.plot & self.save_figs:
            for fig in range(len(plot_params[0])):
                filename = os.path.join(self.final_dir, "field_image_map{0}_line{1}.png".format(index, fig))
                plot_params[0][fig].savefig(filename, bbox_inches="tight")

        mean_profile = np.nanmean(reduced_data[:, 0, :, :], axis=(0, 1))
        approx_wavelength_array = self.tweak_wavelength_calibration(mean_profile)
        # Swap axes to make X/Y contigent with data X/Y
        reduced_data = np.swapaxes(reduced_data, 0, 2)

        reduced_filename = self.package_scan(reduced_data, approx_wavelength_array, master_hairline_centers)
        crosstalk_filename = self.package_crosstalks(complete_i2_quv_crosstalk, complete_internal_crosstalks, index)
        parameter_maps, reference_wavelengths, tweaked_indices, mean_profile, wavelength_array = self.spinor_analysis(
            reduced_data, self.analysis_ranges
        )
        param_filename = self.package_analysis(
            parameter_maps,
            reference_wavelengths,
            tweaked_indices,
            mean_profile,
            wavelength_array,
            reduced_filename
        )

        if self.verbose:
            print("\n\n=====================")
            print("Saved Reduced Data at: {0}".format(reduced_filename))
            print("Saved Parameter Maps at: {0}".format(param_filename))
            print("Saved Crosstalk Coefficients at: {0}".format(crosstalk_filename))
            print("=====================\n\n")

        if self.plot:
            plt.pause(2)
            plt.close("all")

        return

    def subpixel_hairline_align(self, slit_image: np.ndarray, hair_centers=None) -> tuple[np.ndarray, tuple]:
        """
        Performs subpixel hairline alignment of two beams into a single image and returns the necessary translations.
        Returns shift map for hairline alignment.

        Parameters
        ----------
        slit_image : numpy.ndarray
            Minimally-processed image of the spectrum. Beams will be cut out and aligned from this. Shape (ny, nx)
        hair_centers : numpy.ndarray or None
            If None, calls ssosoft.spectraTools.detect_beams_hairlines to get a guess of hairline position
            that's a bit more robust than what we can fit in this function.
            If given as a tuple, takes that as the centers of the hairlines to deskew

        Returns
        -------
        hairline_skews : numpy.ndarray
            Array of shape (2, nx) containing subpixel shifts for hairline registration
        hairline_center : tuple
            Centers of the registered hairline. Should assist in getting an evenly-registered image across all slit pos.
        """

        beam0 = slit_image[self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0]:self.slit_edges[1]]
        beam1 = np.flip(
            scind.shift(
                slit_image[self.beam_edges[1, 0]:self.beam_edges[1, 1], self.slit_edges[0]:self.slit_edges[1]],
                (self.beam1_yshift, self.beam1_xshift), order=1
            ), axis=0
        )

        dual_beams = np.stack([beam0, beam1], axis=0)
        deskewed_dual_beams = dual_beams.copy()

        if self.manual_hairline_selection and hair_centers is None:
            first_step = True
            beam0_profile = beam0.mean(axis=1)
            beam1_profile = beam1.mean(axis=1)
            beam0_range, beam1_range = spex.select_spans_doublepanel(beam0_profile, beam1_profile, 1)
            hairline_minimum = np.array(
                [beam0_range[0, 0], beam1_range[0, 0]]
            )
            hairline_maximum = np.array(
                [beam0_range[0, 1], beam1_range[0, 1]]
            )
            if (hairline_minimum <= 0).any():
                hairline_minimum[np.where(hairline_minimum < 0)[0]] = 0
            if (hairline_maximum >= dual_beams.shape[1]).any():
                hairline_maximum[np.where(
                    hairline_maximum >= dual_beams.shape[1]
                )[0]] = dual_beams.shape[1]
        elif hair_centers is None:
            first_step = True
            spex.detect_beams_hairlines.num_calls = 0
            # Undo the lamp gain for a second to get the intensity jumps in the right direction
            _, _, tmp_hairlines = spex.detect_beams_hairlines(
                slit_image * self.lamp_gain, threshold=self.beam_threshold, hairline_width=self.hairline_width,
                expected_hairlines=self.nhair, expected_slits=1, expected_beams=2, fallback=True  # Just in case
            )
            hair_centers = tmp_hairlines.reshape(2, int(self.nhair / 2))
            hair_centers[0] -= self.beam_edges[0, 0]
            # Have to flip hairlines and index to beam edge
            hair_centers[1] = np.abs(hair_centers[1] - self.beam_edges[1, 0] - np.diff(self.beam_edges[0]))[::-1]
            hair_centers = np.array(
                (hair_centers[0, 0],
                 hair_centers[1, 0] + self.beam1_yshift)
            )
            hairline_minimum = hair_centers - 4
            hairline_maximum = hair_centers + 4
            if (hairline_minimum <= 0).any():
                hairline_minimum = np.array([0, 0])
                hairline_maximum = (2 * hair_centers).astype(int)
            if (hairline_maximum >= dual_beams.shape[1]).any():
                hairline_maximum[:] = dual_beams.shape[1]
                hairline_minimum[:] = hairline_maximum - 2 * (hairline_maximum - hair_centers)
            hairline_minimum = hairline_minimum.astype(int)
            hairline_maximum = hairline_maximum.astype(int)
        else:
            first_step = False
            hairline_minimum = hair_centers - 4
            hairline_maximum = hair_centers + 4
            if (hairline_minimum <= 0).any():
                hairline_minimum = np.array([0, 0])
                hairline_maximum = (2 * hair_centers).astype(int)
            if (hairline_maximum >= dual_beams.shape[1]).any():
                hairline_maximum[:] = dual_beams.shape[1]
                hairline_minimum[:] = hairline_maximum - 2 * (hairline_maximum - hair_centers)
            hairline_minimum = hairline_minimum.astype(int)
            hairline_maximum = hairline_maximum.astype(int)

        # Final catch-all fallback option if something is *still* wrong.
        if (
                (hairline_minimum < 0).any() or
                (hairline_minimum >= dual_beams.shape[1]).any() or
                (hairline_maximum <= 0).any() or
                (hairline_maximum >= dual_beams.shape[1]).any()
        ) and first_step:
            beam0_profile = beam0.mean(axis=1)
            beam1_profile = beam1.mean(axis=1)
            beam0_range, beam1_range = spex.select_spans_doublepanel(beam0_profile, beam1_profile, 1)
            hairline_minimum = np.array(
                [beam0_range[0, 0], beam1_range[0, 0]]
            )
            hairline_maximum = np.array(
                [beam0_range[0, 1], beam1_range[0, 1]]
            )

        hairline_skews = np.zeros((2, dual_beams.shape[2]))
        for i in range(dual_beams.shape[0]):
            medfilt_hairline_image = scind.median_filter(
                dual_beams[i, hairline_minimum[i]:hairline_maximum[i], :],
                size=(2, 25)
            )
            hairline_skews[i, :] = spex.spectral_skew(
                np.rot90(medfilt_hairline_image), order=1, slit_reference=0.5
            )
            for j in range(hairline_skews.shape[1]):
                deskewed_dual_beams[i, :, j] = scind.shift(
                    dual_beams[i, :, j], hairline_skews[i, j],
                    mode='nearest', order=1
                )
        # Find bulk hairline center for full alignment
        hairline_center = (
            spex.find_line_core(
                np.nanmedian(deskewed_dual_beams[0, hairline_minimum[0]:hairline_maximum[0], :], axis=1)
            ) + hairline_minimum[0],
            spex.find_line_core(
                np.nanmedian(deskewed_dual_beams[1, hairline_minimum[1]:hairline_maximum[1], :], axis=1)
            ) + hairline_minimum[1]
        )

        return hairline_skews, hairline_center

    def subpixel_spectral_align(
            self, cutout_beams: np.ndarray, hairline_center: tuple, spectral_ranges: np.ndarray or None=None
    ) -> tuple[np.ndarray, float]:
        """
        Performs iterative deskew and align along the spectral axis.
        Returns the aligned beam and spectral line center of the aligned beam.
        Parameters
        ----------
        cutout_beams : numpy.ndarray
            Cut-out and vertically-aligned
        hairline_center : tuple
            Position of hairline in each beam for masking
        spectral_ranges : numpy.ndarray or None
            If provided and manual spectral selection is set to True, sets range of deskew to provided. Shape (2, 2)

        Returns
        -------
        cutoutBeams : numpy.ndarray
            Horizontally-aligned
        lineCenter : numpy.ndarray
            Position of line used for alignment. Can be used to register the entire image sequence
        """
        # For good measure (and masking) determine upper hairline approx. center
        # From initial spacing between upper/lower hairline pairs.
        # FLIR may not have another set of hairlines.
        if self.hairlines.shape[1] > 1:
            upper_hairline_center = (
                hairline_center[0] + np.diff(self.hairlines, axis=1)[0],
                hairline_center[1] + np.diff(self.hairlines, axis=1)[1]
            )
        else:
            upper_hairline_center = None

        if self.manual_alignment_selection and spectral_ranges is None:
            # Case: iter 0, select alignment range is set to True
            beam0_profile = cutout_beams[0, 0, :, :].mean(axis=0)
            beam1_profile = cutout_beams[1, 0, :, :].mean(axis=0)
            beam0_range, beam1_range = spex.select_spans_doublepanel(beam0_profile, beam1_profile, 1)
            spex_minimum = np.array([beam0_range[0, 0], beam1_range[0, 0]])
            spex_maximum = np.array([beam0_range[0, 1], beam1_range[0, 1]])
            for beam in range(2):
                spectral_image = cutout_beams[beam, 0, :, spex_minimum[beam]:spex_maximum[beam]].copy()
                # Mask hairlines
                hair_min = int(hairline_center[beam] - 4)
                hair_max = int(hairline_center[beam] + 5)
                hair_min = 0 if hair_min < 0 else hair_min
                hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                spectral_image[hair_min:hair_max, :] = np.nan
                # FLIR may not have another set of hairlines.
                if self.hairlines.shape[1] > 1:
                    hair_min = int(upper_hairline_center[beam] - 4)
                    hair_max = int(upper_hairline_center[beam] + 5)
                    hair_min = 0 if hair_min < 0 else hair_min
                    hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                    spectral_image[hair_min:hair_max, :] = np.nan
                spectral_skews = spex.spectral_skew(
                    spectral_image, order=2, slit_reference=0.5
                )
                for prof in range(cutout_beams.shape[2]):
                    cutout_beams[beam, :, prof, :] = scind.shift(
                        cutout_beams[beam, :, prof, :], (0, spectral_skews[prof]), mode='nearest', order=1
                    )
            spex_range = np.array([
                [spex_minimum[0], spex_maximum[0]],
                [spex_minimum[1], spex_maximum[1]]
            ])
        elif self.manual_alignment_selection and spectral_ranges is not None:
            spex_minimum = spectral_ranges[:, 0]
            spex_maximum = spectral_ranges[:, 1]
            for beam in range(2):
                spectral_image = cutout_beams[beam, 0, :, spex_minimum[beam]:spex_maximum[beam]].copy()
                # Mask hairlines
                hair_min = int(hairline_center[beam] - 4)
                hair_max = int(hairline_center[beam] + 5)
                hair_min = 0 if hair_min < 0 else hair_min
                hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                spectral_image[hair_min:hair_max, :] = np.nan
                # FLIR may not have another set of hairlines.
                if self.hairlines.shape[1] > 1:
                    hair_min = int(upper_hairline_center[beam] - 4)
                    hair_max = int(upper_hairline_center[beam] + 5)
                    hair_min = 0 if hair_min < 0 else hair_min
                    hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                    spectral_image[hair_min:hair_max, :] = np.nan
                spectral_skews = spex.spectral_skew(
                    spectral_image, order=2, slit_reference=0.5
                )
                for prof in range(cutout_beams.shape[2]):
                    cutout_beams[beam, :, prof, :] = scind.shift(
                        cutout_beams[beam, :, prof, :], (0, spectral_skews[prof]), mode='nearest', order=1
                    )
            spex_range = np.array([
                [spex_minimum[0], spex_maximum[0]],
                [spex_minimum[1], spex_maximum[1]]
            ])
        else:
            # Default behaviour
            x1, x2 = 20, 21
            for spiter in range(5):
                order = 2 if spiter < 2 else 2
                for beam in range(2):
                    spectral_image = cutout_beams[
                                    beam, 0, :, int(self.spinor_line_cores[0] - x1):int(self.spinor_line_cores[0] + x2)
                                    ].copy()
                    # Deskew function is written to cope with NaNs.
                    # It is NOT written to deal with a hairline.
                    # Mask hairlines
                    hair_min = int(hairline_center[beam] - 4)
                    hair_max = int(hairline_center[beam] + 5)
                    hair_min = 0 if hair_min < 0 else hair_min
                    hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                    spectral_image[hair_min:hair_max, :] = np.nan
                    # FLIR may not have another set of hairlines.
                    if self.hairlines.shape[1] > 1:
                        hair_min = int(upper_hairline_center[beam] - 4)
                        hair_max = int(upper_hairline_center[beam] + 5)
                        hair_min = 0 if hair_min < 0 else hair_min
                        hair_max = int(spectral_image.shape[0] - 1) if hair_max > spectral_image.shape[0] - 1 else hair_max
                        spectral_image[hair_min:hair_max, :] = np.nan
                    spectral_skews = spex.spectral_skew(
                        spectral_image, order=order, slit_reference=0.5
                    )
                    for prof in range(cutout_beams.shape[2]):
                        cutout_beams[beam, :, prof, :] = scind.shift(
                            cutout_beams[beam, :, prof, :], (0, spectral_skews[prof]), mode='nearest', order=1
                        )
                x1 -= 3
                x2 -= 3
            spex_range = np.array([
                [int(self.spinor_line_cores[0] - (x1 + 3)), int(self.spinor_line_cores[0] + (x2 + 3))],
                [int(self.spinor_line_cores[0] - (x1 + 3)), int(self.spinor_line_cores[0] + (x2 + 3))]
            ])
        # Find bulk spectral line center for full alignment
        spectral_center = (
            spex.find_line_core(
                np.nanmedian(
                    cutout_beams[0, 0, :, int(spex_range[0, 0]): int(spex_range[0, 1])],
                    axis=0
                )
            ) + int(spex_range[0, 0]),
            spex.find_line_core(
                np.nanmedian(
                    cutout_beams[1, 0, :, int(spex_range[1, 0]): int(spex_range[1, 1])],
                    axis=0
                )
            ) + int(spex_range[1, 0]),
        )
        return cutout_beams, spectral_center, spex_range

    def solve_spinor_crosstalks(self, iquv_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper for all crosstalk solving terms in reduce_spinor_maps.

        This is done both for cleanliness, and because many of the methods contained within
        are static. An additional benefit is the ability to write crosstalk files for the user to check the
        quality of corrections and revert them if desired.

        In a future release, this function could be expanded to omit "self" in favor of
        a more complete set of keyword argument switches. At that point, this and the other crosstalk
        functions could be moved to top-level and parallelized using either dask.delayed or dask array
        calls.

        The original, iquvCube[1:] can be recovered by performing the following operations:
        1.) Undo U -> V
            iquvCube[3] += np.repeat(internal_crosstalk[2, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[2]
        2.) Undo V -> U
            iquvCube[2] += np.repeat(internal_crosstalk[1, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[3]
        3.) Undo V -> Q
            iquvCube[1] += np.repeat(internal_crosstalk[0, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[3]
        4.) Undo I -> QUV
            iquvCube[1:] += (np.repeat(crosstalk_i2_quv[:, 0, :, np.newaxis], iquvCube.shape[2], axis=-1) *
                    np.arange(iquvCube.shape[2] +
                    np.repeat(crosstalk_i2_quv[:, 1, :, np.newaxis], iquvCube.shape[2], axis=-1)) *
                np.repeat(iquvCube[0][np.newaxis, :, :], 3, axis=0)

        Parameters
        ----------
        iquv_cube : numpy.ndarray
            Mostly-corrected 3D array of a combined full-Stokes slit image. Should have shape (4, ny, nlambda)

        Returns
        -------
        iquvCube : numpy.ndarray
            Shape (4, ny, nlamba), crosstalk-isolated datacube.
        crosstalk_i2_quv : numpy.ndarray
            Has shape (3, 2, ny), contains I->QUV crosstalk coefficients.
        internal_crosstalk : numpy.ndarray
            Has shape (3, ny), contains V->Q, V->U, U->V crosstalks
        """

        crosstalk_i2_quv = np.zeros((3, 2, iquv_cube.shape[1]))
        internal_crosstalk = np.zeros((3, iquv_cube.shape[1]))

        if self.crosstalk_continuum is not None:
            # Shape 3xNY
            i2quv = np.mean(
                iquv_cube[1:, :, self.crosstalk_continuum[0]:self.crosstalk_continuum[1]] /
                np.repeat(
                    iquv_cube[0, :, self.crosstalk_continuum[0]:self.crosstalk_continuum[1]][np.newaxis, :, :],
                    3, axis=0
                ), axis=2
            )
            iquv_cube[1:] = iquv_cube[1:] - np.repeat(
                i2quv[:, :, np.newaxis], iquv_cube.shape[2], axis=2
            ) * np.repeat(iquv_cube[0][np.newaxis, :, :], 3, axis=0)
            crosstalk_i2_quv[:, 1, :] = i2quv
        else:
            for j in range(iquv_cube.shape[1]):
                # I->QUV crosstalk correction
                for k in range(1, 4):
                    iquv_cube[k, j, :], crosstalk_i2_quv[k - 1, :, j] = self.i2quv_crosstalk(
                        iquv_cube[0, j, :],
                        iquv_cube[k, j, :]
                    )
        if self.i2quv_residual:
            for j in range(iquv_cube.shape[1]):
                for k in range(1, 4):
                    iquv_cube[k, j, :], residual_i2quv = self.residual_i2quv_crosstalk(
                        iquv_cube[0, j, :],
                        iquv_cube[k, j, :]
                    )
                    crosstalk_i2_quv[k - 1, :, j] += residual_i2quv

        # V->QU crosstalk correction
        if self.v2q:
            bulk_v2_q_crosstalk = self.internal_crosstalk_2d(
                iquv_cube[1, :, :], iquv_cube[3, :, :]
            )
            iquv_cube[1, :, :] = iquv_cube[1, :, :] - bulk_v2_q_crosstalk * iquv_cube[3, :, :]
            if self.v2q == "full":
                for j in range(iquv_cube.shape[1]):
                    iquv_cube[1, j, :], internal_crosstalk[0, j] = self.v2qu_crosstalk(
                        iquv_cube[3, j, :],
                        iquv_cube[1, j, :]
                    )
            internal_crosstalk[0] += bulk_v2_q_crosstalk
        if self.v2u:
            bulk_v2_u_crosstalk = self.internal_crosstalk_2d(
                iquv_cube[2, :, :], iquv_cube[3, :, :]
            )
            iquv_cube[2, :, :] = iquv_cube[2, :, :] - bulk_v2_u_crosstalk * iquv_cube[3, :, :]
            if self.v2u == "full":
                for j in range(iquv_cube.shape[1]):
                    iquv_cube[2, j, :], internal_crosstalk[1, j] = self.v2qu_crosstalk(
                        iquv_cube[3, j, :],
                        iquv_cube[2, j, :]
                    )
            internal_crosstalk[1] += bulk_v2_u_crosstalk
        if self.u2v:
            bulk_u2_v_crosstalk = self.internal_crosstalk_2d(
                iquv_cube[3, :, :], iquv_cube[2, :, :]
            )
            iquv_cube[3, :, :] = iquv_cube[3, :, :] - bulk_u2_v_crosstalk * iquv_cube[2, :, :]
            if self.u2v == "full":
                for j in range(iquv_cube.shape[1]):
                    iquv_cube[3, j, :], internal_crosstalk[2, j] = self.v2qu_crosstalk(
                        iquv_cube[2, j, :],
                        iquv_cube[3, j, :]
                    )
            internal_crosstalk[2] += bulk_u2_v_crosstalk

        return iquv_cube, crosstalk_i2_quv, internal_crosstalk

    def package_crosstalks(self, i2quv_crosstalks: np.ndarray, internal_crosstalks: np.ndarray, index: int) -> str:
        """
        Places Stokes-vector crosstalk parameters in a file for interested end users.

        Parameters
        ----------
        i2quv_crosstalks : numpy.ndarray
            Array of shape (nx, 3, 2, ny) of I->QUV crosstalk parameters.
            The option exists to use a 1D fit for crosstalk, hence the 2-axis, where the 2 entries are "m" and "b"
            in y=mx+b
        internal_crosstalks : numpy.ndarray
            Array of shape (nx, 3, ny) of internal crosstalk. (:, 0, :) is V->Q, (:, 1, :) is V->U, (:, 2, :) is U->V
        index : int
            For formatting the output filename

        Returns
        -------
        crosstalk_file : str
            Name of file where crosstalk parameters are stored.

        """
        ext0 = fits.PrimaryHDU()
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File Creation Date and Time")
        ext0.header['ORIGIN'] = "NMSU/SSOC"
        if self.crosstalk_continuum is not None:
            ext0.header['I2QUV'] = ("CONST", "0-D I2QUV Crosstalk")
        else:
            ext0.header['I2QUV'] = ("1DFIT", "1-D I2QUV Crosstalk")
        ext0.header['V2Q'] = (self.v2q, "True=by slit, Full=by slit and row")
        ext0.header['V2U'] = (self.v2u, "True=by slit, Full=by slit and row")
        ext0.header['U2V'] = (self.u2v, "True=by slit, Full=by slit and row")
        ext0.header['COMMENT'] = "Crosstalks applied in order:"
        ext0.header['COMMENT'] = "I->QUV"
        ext0.header['COMMENT'] = "V->Q"
        ext0.header['COMMENT'] = "V->U"
        ext0.header['COMMENT'] = "U->V"

        i2quv_ext = fits.ImageHDU(i2quv_crosstalks)
        i2quv_ext.header['EXTNAME'] = "I2QUV"
        i2quv_ext.header[""] = "<QUV> = <QUV> - (coef[0]*[0, 1, ... nlambda] + coef[1]) * I"

        v2q_ext = fits.ImageHDU(internal_crosstalks[:, 0, :])
        v2q_ext.header['EXTNAME'] = "V2Q"
        v2q_ext.header[""] = "Q = Q - coef*V"

        v2u_ext = fits.ImageHDU(internal_crosstalks[:, 1, :])
        v2u_ext.header['EXTNAME'] = "V2U"
        v2u_ext.header[""] = "U = U - coef*V"

        u2v_ext = fits.ImageHDU(internal_crosstalks[:, 2, :])
        u2v_ext.header['EXTNAME'] = "U2V"
        u2v_ext.header[""] = "V = V - coef*U"

        hdul = fits.HDUList([ext0, i2quv_ext, v2q_ext, v2u_ext, u2v_ext])
        filename = "{0}_MAP_{1}_CROSSTALKS.fits".format(self.camera, index)
        crosstalk_file = os.path.join(self.final_dir, filename)
        hdul.writeto(crosstalk_file, overwrite=True)
        return crosstalk_file

    def set_up_live_plot(
            self, field_images: np.ndarray, slit_images: np.ndarray, internal_crosstalks: np.ndarray,
            dy: float, dx: float
    ) -> tuple:
        """
        Initializes live plotting statements for monitoring progress of reductions

        Parameters
        ----------
        field_images : numpy.ndarray
            Array of dummy field images for line cores. Shape nlines, 4, ny, nx where:
                1.) nlines is the number of spectral lines selected by the user to keep an eye on
                2.) 4 is from the IQUV stokes parameters
                3.) ny is the length of the slit
                4.) nx is the number of slit positions in the scan
        slit_images : numpy.ndarray
            Array of IQUV slit images. Shape 4, ny, nlambda where:
                5.) nlambda is the wavelength axis
        internal_crosstalks : numpy.ndarray
            3 x ny array of V<->QU crosstalk values for monitoring
        dy : float
            Approximate resolution scale in y for sizing the map extent
        dx : float
            Approximate resolution scale in x for sizing the map extent

        Returns
        -------
        field_fig_list : list
            List of matplotlib figures, with an entry for each line of interest
        field_i : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-I subplot
        field_q : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-Q subplot
        field_u : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-U subplot
        field_v : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-V subplot
        slit_fig : matplotlib.pyplot.figure
            Matplotlib figure containing the slit IQUV image. The entire image will be blitted each time
        slit_i : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-I image
        slit_q : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-Q image
        slit_u : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-U image
        slit_v : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-V image
        """
        # Close all figures to reset plotting
        plt.close("all")

        # Required for live plotting
        plt.ion()
        plt.pause(0.005)

        slit_aspect_ratio = slit_images.shape[2] / slit_images.shape[1]

        # Set up the spectral data first, since it's only one window
        slit_fig = plt.figure("Reduced Slit Images", figsize=(5, 5 / slit_aspect_ratio))
        slit_gs = slit_fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
        slit_ax_i = slit_fig.add_subplot(slit_gs[0, 0])
        slit_i = slit_ax_i.imshow(slit_images[0], cmap='gray', origin='lower')
        slit_ax_i.text(10, 10, "I", color='C1')
        slit_ax_q = slit_fig.add_subplot(slit_gs[0, 1])
        slit_q = slit_ax_q.imshow(slit_images[1], cmap='gray', origin='lower')
        slit_ax_q.text(10, 10, "Q", color='C1')
        slit_ax_u = slit_fig.add_subplot(slit_gs[1, 0])
        slit_u = slit_ax_u.imshow(slit_images[2], cmap='gray', origin='lower')
        slit_ax_u.text(10, 10, "U", color='C1')
        slit_ax_v = slit_fig.add_subplot(slit_gs[1, 1])
        slit_v = slit_ax_v.imshow(slit_images[3], cmap='gray', origin='lower')
        slit_ax_v.text(10, 10, "V", color='C1')

        # Now the multiple windows for the multiple lines of interest
        field_aspect_ratio = (dx * field_images.shape[3]) / (dy * field_images.shape[2])

        field_fig_list = []
        field_gs = []
        field_i = []
        field_q = []
        field_u = []
        field_v = []
        field_i_ax = []
        field_q_ax = []
        field_u_ax = []
        field_v_ax = []
        for j in range(field_images.shape[0]):
            field_fig_list.append(
                plt.figure("Line " + str(j), figsize=(5, 5 / field_aspect_ratio + 1))
            )
            field_gs.append(
                field_fig_list[j].add_gridspec(2, 2, hspace=0.1, wspace=0.1)
            )
            field_i_ax.append(
                field_fig_list[j].add_subplot(field_gs[j][0, 0])
            )
            field_i.append(
                field_i_ax[j].imshow(
                    field_images[j, 0], origin='lower', cmap='gray',
                    extent=[0, dx * field_images.shape[3], 0, dy * field_images.shape[2]]
                )
            )
            field_q_ax.append(
                field_fig_list[j].add_subplot(field_gs[j][0, 1])
            )
            field_q.append(
                field_q_ax[j].imshow(
                    field_images[j, 1], origin='lower', cmap='gray',
                    extent=[0, dx * field_images.shape[3], 0, dy * field_images.shape[2]]
                )
            )
            field_u_ax.append(
                field_fig_list[j].add_subplot(field_gs[j][1, 0])
            )
            field_u.append(
                field_u_ax[j].imshow(
                    field_images[j, 2], origin='lower', cmap='gray',
                    extent=[0, dx * field_images.shape[3], 0, dy * field_images.shape[2]]
                )
            )
            field_v_ax.append(
                field_fig_list[j].add_subplot(field_gs[j][1, 1])
            )
            field_v.append(
                field_v_ax[j].imshow(
                    field_images[j, 2], origin='lower', cmap='gray',
                    extent=[0, dx * field_images.shape[3], 0, dy * field_images.shape[2]]
                )
            )

            # Beautification; Turn off some x/y tick labels, set titles, axes labels, etc...
            # Turn off tick labels for all except the first column in y, and the last row in x
            field_i_ax[j].set_xticklabels([])
            field_i_ax[j].set_ylabel("Extent [arcsec]")
            field_i_ax[j].set_title("Line Core  Stokes-I")

            field_q_ax[j].set_yticklabels([])
            field_q_ax[j].set_xticklabels([])
            field_q_ax[j].set_title("Integrated Stokes-Q")

            field_u_ax[j].set_ylabel("Extent [arcsec]")
            field_u_ax[j].set_xlabel("Extent [arcsec]")
            field_u_ax[j].set_title("Integrated Stokes-U")

            field_v_ax[j].set_yticklabels([])
            field_v_ax[j].set_xlabel("Extent [arcsec]")
            field_v_ax[j].set_title("Integrated Stokes-V")

        if not any((self.v2q, self.v2u, self.u2v)):

            plt.show(block=False)
            plt.pause(0.05)

            return (field_fig_list, field_i, field_q, field_u, field_v, slit_fig, slit_i, slit_q, slit_u, slit_v,
                    None, None, None, None)
        else:
            crosstalk_fig = plt.figure("Internal Crosstalks Along Slit", figsize=(8, 5))
            v2q_ax = crosstalk_fig.add_subplot(131)
            v2u_ax = crosstalk_fig.add_subplot(132)
            u2v_ax = crosstalk_fig.add_subplot(133)
            v2q = v2q_ax.plot(
                internal_crosstalks[0, :], np.arange(internal_crosstalks.shape[1]),
                color='C1'
            )
            v2q_ax.set_xlim(-1.05, 1.05)
            v2q_ax.set_ylim(0, internal_crosstalks.shape[1])
            v2q_ax.set_title("V->Q Crosstalk")
            v2q_ax.set_ylabel("Position Along Slit")

            v2u = v2u_ax.plot(
                internal_crosstalks[1, :], np.arange(internal_crosstalks.shape[1]),
                color='C1'
            )
            v2u_ax.set_xlim(-1.05, 1.05)
            v2u_ax.set_ylim(0, internal_crosstalks.shape[1])
            v2u_ax.set_title("V->U Crosstalk")
            v2u_ax.set_xlabel("Crosstalk Value")

            u2v = u2v_ax.plot(
                internal_crosstalks[2, :], np.arange(internal_crosstalks.shape[1]),
                color="C1"
            )
            u2v_ax.set_xlim(-1.05, 1.05)
            u2v_ax.set_ylim(0, internal_crosstalks.shape[1])
            u2v_ax.set_title("U->V Crosstalk [residual]")

            plt.show(block=False)
            plt.pause(0.05)

            return (
                field_fig_list, field_i, field_q, field_u, field_v,
                slit_fig, slit_i, slit_q, slit_u, slit_v,
                crosstalk_fig, v2q, v2u, u2v)

    def update_live_plot(
            self,
            field_fig_list: list, field_i: list, field_q: list, field_u: list, field_v: list,
            slit_fig: matplotlib.pyplot.figure, slit_i: matplotlib.image.AxesImage, slit_q: matplotlib.image.AxesImage,
            slit_u: matplotlib.image.AxesImage, slit_v: matplotlib.image.AxesImage,
            crosstalk_fig: matplotlib.pyplot.figure, v2q: matplotlib.image.AxesImage,
            v2u: matplotlib.image.AxesImage, u2v: matplotlib.image.AxesImage,
            field_images: np.ndarray, slit_images: np.ndarray, internal_crosstalks: np.ndarray,
            step: int
    ) -> None:
        """
        Updates the plots created in self.set_up_live_plot.

        Parameters
        ----------
        field_fig_list : list
        field_i : list
        field_q : list
        field_u : list
        field_v : list
        slit_fig : matplotlib.pyplot.figure
        slit_i : matplotlib.image.AxesImage
        slit_q : matplotlib.image.AxesImage
        slit_u : matplotlib.image.AxesImage
        slit_v : matplotlib.image.AxesImage
        crosstalk_fig : matplotlib.pyplot.figure
        v2q : matplotlib.image.AxesImage
        v2u : matplotlib.image.AxesImage
        u2v : matplotlib.image.AxesImage
        field_images : numpy.ndarray
        slit_images : numpy.ndarray
        internal_crosstalks : numpy.ndarray
        step : int
            Step of the reduction process we're on for normalization purposes.

        Returns
        -------

        """

        slit_i.set_array(slit_images[0])
        slit_i.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[0]) - 3 * np.std(slit_images[0]),
                vmax=np.mean(slit_images[0]) + 3 * np.std(slit_images[0])
            )
        )
        slit_q.set_array(slit_images[1])
        slit_q.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[1]) - 3 * np.std(slit_images[1]),
                vmax=np.mean(slit_images[1]) + 3 * np.std(slit_images[1])
            )
        )
        slit_u.set_array(slit_images[2])
        slit_u.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[2]) - 3 * np.std(slit_images[2]),
                vmax=np.mean(slit_images[2]) + 3 * np.std(slit_images[2])
            )
        )
        slit_v.set_array(slit_images[3])
        slit_v.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[3]) - 3 * np.std(slit_images[3]),
                vmax=np.mean(slit_images[3]) + 3 * np.std(slit_images[3])
            )
        )

        slit_fig.canvas.draw()
        slit_fig.canvas.flush_events()

        for j in range(field_images.shape[0]):
            field_i[j].set_array(field_images[j, 0])
            field_i[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(field_images[j, 0, :, :step]) - 3 * np.std(field_images[j, 0, :, :step]),
                    vmax=np.mean(field_images[j, 0, :, :step]) + 3 * np.std(field_images[j, 0, :, :step])
                )
            )
            field_q[j].set_array(field_images[j, 1])
            field_q[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(field_images[j, 1, :, :step]) - 3 * np.std(field_images[j, 1, :, :step]),
                    vmax=np.mean(field_images[j, 1, :, :step]) + 3 * np.std(field_images[j, 1, :, :step])
                )
            )
            field_u[j].set_array(field_images[j, 2])
            field_u[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(field_images[j, 2, :, :step]) - 3 * np.std(field_images[j, 2, :, :step]),
                    vmax=np.mean(field_images[j, 2, :, :step]) + 3 * np.std(field_images[j, 2, :, :step])
                )
            )
            field_v[j].set_array(field_images[j, 3])
            field_v[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(field_images[j, 3, :, :step]) - 3 * np.std(field_images[j, 3, :, :step]),
                    vmax=np.mean(field_images[j, 3, :, :step]) + 3 * np.std(field_images[j, 3, :, :step])
                )
            )
            field_fig_list[j].canvas.draw()
            field_fig_list[j].canvas.flush_events()

        if crosstalk_fig is not None:
            v2q[0].set_data(internal_crosstalks[0], np.arange(internal_crosstalks.shape[1]))
            v2u[0].set_data(internal_crosstalks[1], np.arange(internal_crosstalks.shape[1]))
            u2v[0].set_data(internal_crosstalks[2], np.arange(internal_crosstalks.shape[1]))
            crosstalk_fig.canvas.draw()
            crosstalk_fig.canvas.flush_events()
        return

    def package_scan(self, datacube: np.ndarray, wavelength_array: np.ndarray, hairline_centers: tuple) -> str:
        """
        Packages reduced scan into FITS HDUList. HDUList has 7 extensions:
            1.) Empty data attr with top-level header info
            2--5.) Stokes-I, Q, U, V
            6.) Wavelength Array
            7.) Metadata array (Contains:)
                Pointing Lat/Lon, Timestamp, scintillation, light level, slit position
        Parameters
        ----------
        datacube : numpy.ndarray
            4D reduced stokes data in shape ny, 4, nx, nlambda
        wavelength_array : numpy.ndarray
            1D array containing the wavelengths corrsponding to nlambda in datacube
        hairline_centers : tuple
            Tuple containing the subpixel center of the hairline(s) used in registering the slit images

        EDIT 2025-01-27: Found out that maps are often split across multiple files. Rewrote code to take list of maps.
        As a result, have to iterate over self.scienceFiles for necessary header information...

        """

        prsteps = [
            'DARK-SUBTRACTION',
            'FLATFIELDING',
            'WAVELENGTH-CALIBRATION',
            'TELESCOPE-MULLER',
            'SPECTROGRAPH-MULLER',
            'I->QUV CROSSTALK'
        ]
        prstep_comments = [
            'spinorCal/SSOSoft',
            'spinorCal/SSOSoft',
            'FTS Atlas',
            '2010 Measurements',
            'spinorCal/SSOSoft',
            'spinorCal/SSOSoft'
        ]

        if self.v2q:
            prsteps.append(
                'V->Q CROSSTALK'
            )
            prstep_comments.append(
                'spinorCal/SSOSoft'
            )
        if self.v2u:
            prsteps.append(
                'V->U CROSSTALK'
            )
            prstep_comments.append(
                'spinorCal/SSOSoft'
            )
        if self.u2v:
            prsteps.append(
                'U->V CROSSTALK'
            )
            prstep_comments.append(
                'spinorCal/SSOSoft'
            )

        slit_plate_scale = self.telescope_plate_scale * self.dst_collimator / self.slit_camera_lens
        camera_dy = slit_plate_scale * (self.spectrograph_collimator / self.camera_lens) * (self.pixel_size / 1000)

        with fits.open(self.science_files[0]) as hdul:
            exptime = hdul[1].header['EXPTIME']
            xposure = int(hdul[1].header['SUMS'] * exptime)
            nsumexp = hdul[1].header['SUMS']
            slitwidth = hdul[1].header['HSG_SLW']
            stepsize = hdul[1].header['HSG_STEP']
            reqmapsize = hdul[1].header['HSG_MAP']
            actmapsize = stepsize * (datacube.shape[0] - 1)
            gratingangle = hdul[1].header['HSG_GRAT']
            rsun = hdul[1].header['DST_SDIM'] / 2
            camera_name = hdul[0].header['CAMERA']

        step_startobs = []
        solar_x = []
        solar_y = []
        rotan = []
        llvl = []
        scin = []
        slitpos = []

        for file in self.science_files:
            with fits.open(file) as hdul:
                for hdu in hdul[1:]:
                    step_startobs.append(hdu.header['DATE-OBS'])
                    rotan.append(hdu.header['DST_GDRN'] - 13.3)
                    llvl.append(hdu.header['DST_LLVL'])
                    scin.append(hdu.header['DST_SEE'])
                    slitpos.append(hdu.header['HSG_SLP'])
                    center_coord = SkyCoord(
                        hdu.header['DST_SLNG'] * u.deg, hdu.header['DST_SLAT'] * u.deg,
                        obstime=hdu.header['DATE-OBS'], observer='earth', frame=frames.HeliographicStonyhurst
                    ).transform_to(frames.Helioprojective)
                    solar_x.append(center_coord.Tx.value)
                    solar_y.append(center_coord.Ty.value)

        rotan = np.nanmean(rotan)

        date, time = step_startobs[0].split("T")
        date = date.replace("-", "")
        time = str(round(float(time.replace(":", "")), 0)).split(".")[0]

        outname = self.reduced_file_pattern.format(
            date,
            time,
            datacube.shape[2]
        )
        outfile = os.path.join(self.final_dir, outname)

        # Need center, have to account for maps w/ even number of steps
        if len(slitpos) % 2 == 0:
            slit_pos_center = (slitpos[int(len(slitpos) / 2) - 1] + slitpos[int(len(slitpos) / 2)]) / 2
            center_x = (solar_x[int(len(solar_x) / 2) - 1] + solar_x[int(len(solar_x) / 2)]) / 2
            center_y = (solar_y[int(len(solar_y) / 2) - 1] + solar_y[int(len(solar_y) / 2)]) / 2
        else:
            slit_pos_center = slitpos[int(len(slitpos) / 2)]
            center_x = solar_x[int(len(slitpos) / 2)]
            center_y = solar_y[int(len(slitpos) / 2)]
        # SPINOR has issues with crashing partway through maps.
        # This poses an issue for determining the center point of a given map,
        # As a crash will cause a map to be off-center relative to the telescope center
        # If the requested and actual map sizes don't match, we'll have to
        # do some math to get the actual center of the map.
        # dX = cos(90 - rotan) * slitPos at halfway point
        # dY = sin(90 - rotan) * slitPos at halfway point
        if round(reqmapsize, 4) != round(actmapsize, 4):
            dx = slit_pos_center * np.cos((90 - rotan) * np.pi / 180)
            center_x += dx
            dy = slit_pos_center * np.sin((90 - rotan) * np.pi / 180)
            center_y -= dy  # Note sign

        # Start Assembling HDUList
        # Empty 0th HDU first
        ext0 = fits.PrimaryHDU()
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File Creation Date and Time (UTC)")
        ext0.header['ORIGIN'] = 'NMSU/SSOC'
        ext0.header['TELESCOP'] = ('DST', "Dunn Solar Telescope, Sacramento Peak NM")
        ext0.header['INSTRUME'] = ("SPINOR", "SPectropolarimetry of INfrared and Optical Regions")
        ext0.header['AUTHOR'] = "sellers"
        ext0.header['CAMERA'] = camera_name
        ext0.header['DATA_LEV'] = 1.5

        if self.central_wavelength == 6302:
            ext0.header['WAVEBAND'] = "Fe I 6301.5 AA, Fe I 6302.5 AA"
        elif self.central_wavelength == 8542:
            ext0.header['WAVEBAND'] = "Ca II 8542 AA"
        ext0.header['STARTOBS'] = step_startobs[0]
        ext0.header['ENDOBS'] = (np.datetime64(step_startobs[-1]) + np.timedelta64(xposure, 'ms')).astype(str)
        ext0.header['BTYPE'] = 'Intensity'
        ext0.header['BUNIT'] = 'Corrected DN'
        ext0.header['EXPTIME'] = (exptime, 'ms for single exposure')
        ext0.header['XPOSUR'] = (xposure, 'ms for total coadded exposure')
        ext0.header['NSUMEXP'] = (nsumexp, "Summed images per modulation state")
        ext0.header['SLIT_WID'] = (slitwidth, "[um] HSG Slit Width")
        ext0.header['SLIT_ARC'] = (
            round(slit_plate_scale * slitwidth / 1000, 2),
            "[arcsec, approx] HSG Slit Width"
        )
        ext0.header['MAP_EXP'] = (round(reqmapsize, 3), "[arcsec] Requested Map Size")
        ext0.header['MAP_ACT'] = (round(actmapsize, 3), "[arcsec] Actual Map Size")

        ext0.header['WAVEUNIT'] = (-10, "10^(WAVEUNIT), Angstrom")
        ext0.header['WAVEREF'] = ("FTS", "Kurucz 1984 Atlas Used in Wavelength Determination")
        ext0.header['WAVEMIN'] = (round(wavelength_array[0], 3), "[AA] Angstrom")
        ext0.header['WAVEMAX'] = (round(wavelength_array[-1], 3), "[AA], Angstrom")
        ext0.header['GRPERMM'] = (self.grating_rules, "[mm^-1] Lines per mm of Grating")
        ext0.header['GRBLAZE'] = (self.blaze_angle, "[degrees] Blaze Angle of Grating")
        ext0.header['GRANGLE'] = (gratingangle, "[degreed] Operating Angle of Grating")
        ext0.header['SPORDER'] = (self.spectral_order, "Spectral Order")
        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, gratingangle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens, slit_width=slitwidth,
        )
        ext0.header['SPEFF'] = (round(float(grating_params['Total_Efficiency']), 3), 'Approx. Total Efficiency of Grating')
        ext0.header['LITTROW'] = (round(float(grating_params['Littrow_Angle']), 3), '[degrees] Littrow Angle')
        ext0.header['RESOLVPW'] = (
            round(np.nanmean(wavelength_array) / (0.001 * float(grating_params['Spectrograph_Resolution'])), 0),
            "Maximum Resolving Power of Spectrograph"
        )
        for h in range(len(hairline_centers)):
            ext0.header['HAIRLIN{0}'.format(h)] = (round(hairline_centers[h], 3), "Center of registration hairline")

        ext0.header['RSUN_ARC'] = rsun
        ext0.header['XCEN'] = (round(center_x, 2), "[arcsec], Solar-X of Map Center")
        ext0.header['YCEN'] = (round(center_y, 2), "[arcsec], Solar-Y of Map Center")
        ext0.header['FOVX'] = (round(actmapsize, 3), "[arcsec], Field-of-view of raster-x")
        ext0.header['FOVY'] = (round(datacube.shape[0] * camera_dy, 3), "[arcsec], Field-of-view of raster-y")
        ext0.header['ROT'] = (round(rotan, 3), "[degrees] Rotation from Solar-North")

        for i in range(len(prsteps)):
            ext0.header['PRSTEP' + str(int(i + 1))] = (prsteps[i], prstep_comments[i])
        ext0.header['COMMENT'] = "Full WCS Information Contained in Individual Data HDUs"

        ext0.header.insert(
            "DATA_LEV",
            ('', '======== DATA SUMMARY ========'),
            after=True
        )
        ext0.header.insert(
            "WAVEUNIT",
            ('', '======== SPECTROGRAPH CONFIGURATION ========')
        )
        ext0.header.insert(
            "RSUN_ARC",
            ('', '======== POINTING INFORMATION ========')
        )
        ext0.header.insert(
            "PRSTEP1",
            ('', '======== CALIBRATION PROCEDURE OUTLINE ========')
        )

        fits_hdus = [ext0]

        # Stokes-IQUV HDU Construction
        stokes = ['I', 'Q', 'U', 'V']
        for i in range(4):
            ext = fits.ImageHDU(datacube[:, i, :, :])
            ext.header['EXTNAME'] = 'STOKES-' + stokes[i]
            ext.header['RSUN_ARC'] = rsun
            ext.header['CDELT1'] = (stepsize, "arcsec")
            ext.header['CDELT2'] = (camera_dy, "arcsec")
            ext.header['CDELT3'] = (wavelength_array[1] - wavelength_array[0], "Angstrom")
            ext.header['CTYPE1'] = 'HPLN-TAN'
            ext.header['CTYPE2'] = 'HPLT-TAN'
            ext.header['CTYPE3'] = 'WAVE'
            ext.header['CUNIT1'] = 'arcsec'
            ext.header['CUNIT2'] = 'arcsec'
            ext.header['CUNIT3'] = 'Angstrom'
            ext.header['CRVAL1'] = (center_x, "Solar-X, arcsec")
            ext.header['CRVAL2'] = (center_y, "Solar-Y, arcsec")
            ext.header['CRVAL3'] = (wavelength_array[0], "Angstrom")
            ext.header['CRPIX1'] = np.mean(np.arange(datacube.shape[0])) + 1
            ext.header['CRPIX2'] = np.mean(np.arange(datacube.shape[2])) + 1
            ext.header['CRPIX3'] = 1
            ext.header['CROTA2'] = (rotan, "degrees")
            for h in range(len(hairline_centers)):
                ext.header['HAIRLIN{0}'.format(h)] = (round(hairline_centers[h], 3), "Center of registration hairline")
            fits_hdus.append(ext)

        ext_wvl = fits.ImageHDU(wavelength_array)
        ext_wvl.header['EXTNAME'] = 'lambda-coordinate'
        ext_wvl.header['BTYPE'] = 'lambda axis'
        ext_wvl.header['BUNIT'] = '[AA]'

        fits_hdus.append(ext_wvl)

        # Finally write the metadata extension.
        # This is a FITS table with
        #   1.) Elapsed Time
        #   2.) Telescope Solar-X
        #   3.) Telescope Solar-Y
        #   4.) Telescope Light Level
        #   5.) Telescope Scintillation
        timestamps = np.array([np.datetime64(t) for t in step_startobs])
        timedeltas = timestamps - timestamps[0].astype('datetime64[D]')
        timedeltas = timedeltas.astype('timedelta64[ms]').astype(float) / 1000
        columns = [
            fits.Column(
                name='T_ELAPSED',
                format='D',
                unit='SECONDS',
                array=timedeltas,
                time_ref_pos=timestamps[0].astype('datetime64[D]').astype(str)
            ),
            fits.Column(
                name='TEL_SOLX',
                format='D',
                unit='ARCSEC',
                array=np.array(solar_x)
            ),
            fits.Column(
                name='TEL_SOLY',
                format='D',
                unit='ARCSEC',
                array=np.array(solar_y)
            ),
            fits.Column(
                name='LIGHTLVL',
                format='D',
                unit='UNITLESS',
                array=np.array(llvl)
            ),
            fits.Column(
                name='TELESCIN',
                format='D',
                unit='ARCSEC',
                array=np.array(scin)
            )
        ]
        ext_met = fits.BinTableHDU.from_columns(columns)
        ext_met.header['EXTNAME'] = 'METADATA'
        fits_hdus.append(ext_met)

        fits_hdu_list = fits.HDUList(fits_hdus)
        fits_hdu_list.writeto(outfile, overwrite=True)

        return outfile

    def spinor_analysis(
            self, datacube: np.ndarray, bound_indices: np.ndarray
    ) -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
        """
        Performs moment analysis, determines mean circular/linear polarization, and net circular polarization
        maps for each of the given spectral windows. See Martinez Pillet et.al., 2011 discussion of mean polarization
        For net circular polarization, see Solanki & Montavon 1993

        Parameters
        ----------
        datacube : numpy.ndarray
            Reduced FIRS data
        bound_indices : numpy.ndarray
            List of indices for spectral regions of interest. Each entry in the list is a tuple of (xmin, xmax).

        Returns
        -------
        parameter_maps : numpy.ndarray
            Array of derived parameter maps. Has shape (number of regions, 6, ny, nx), where the number of regions is
            chosen by the user during map reduction. 6 is from the number of parameters derived: Intensity, velocity,
            velocity width, net circular polarization, mean circular polarization, and mean linear polarization

        """
        # 7 maps per region of interest: Core intensity, integrated circ. pol., mean circ. pol., net circ. pol.,
        # mean lin. pol., velocity, velocity width
        parameter_maps = np.zeros((bound_indices.shape[0], 6, datacube.shape[0], datacube.shape[2]))
        mean_profile = np.nanmean(datacube[:, 0, :, :], axis=(0, 1))
        wavelength_array = self.tweak_wavelength_calibration(mean_profile)
        # Tweak indices to be an even range around the line core
        tweaked_indices = []
        reference_wavelengths = []
        for i in range(bound_indices.shape[0]):
            # Integer line core
            line_core = spex.find_line_core(
                mean_profile[int(bound_indices[i][0]):int(bound_indices[i][1])]
            ) + int(bound_indices[i][0])
            reference_wavelengths.append(
                float(scinterp.interp1d(np.arange(len(wavelength_array)), wavelength_array)(line_core))
            )
            # New min
            min_range = int(round(line_core - np.abs(np.diff(bound_indices[i]))[0] / 2, 0))
            max_range = int(round(line_core + np.abs(np.diff(bound_indices[i]))[0] / 2, 0)) + 1
            tweaked_indices.append((min_range, max_range))
        with tqdm.tqdm(
                total=parameter_maps.shape[0] * parameter_maps.shape[2] * parameter_maps.shape[3],
                desc="Constructing Derived Parameter Maps"
        ) as pbar:
            for i in range(parameter_maps.shape[0]):
                for j in range(parameter_maps.shape[2]):
                    for k in range(parameter_maps.shape[3]):
                        spectral_profile = datacube[j, 0, k, tweaked_indices[i][0]:tweaked_indices[i][1]]
                        intens, vel, wid = spex.moment_analysis(
                            wavelength_array[tweaked_indices[i][0]:tweaked_indices[i][1]],
                            spectral_profile,
                            reference_wavelengths[i]
                        )
                        parameter_maps[i, 0:3, j, k] = np.array([intens, vel, wid])
                        # Rather than trying to calculate a continuum value, we'll take the average of the four
                        # values on the outsize of the profiles.
                        pseudo_continuum = np.nanmean(
                            spectral_profile.take([-2, -1, 0, 1])
                        )
                        # Net V
                        parameter_maps[i, 3, j, k] = spex.net_circular_polarization(
                            datacube[j, 3, k, tweaked_indices[i][0]:tweaked_indices[i][1]],
                            wavelength_array[tweaked_indices[i][0]:tweaked_indices[i][1]]
                        )
                        # Mean V
                        parameter_maps[i, 4, j, k] = spex.mean_circular_polarization(
                            datacube[j, 3, k, tweaked_indices[i][0]:tweaked_indices[i][1]],
                            wavelength_array[tweaked_indices[i][0]:tweaked_indices[i][1]],
                            reference_wavelengths[i],
                            pseudo_continuum
                        )
                        # Mean QU
                        parameter_maps[i, 5, j, k] = spex.mean_linear_polarization(
                            datacube[j, 1, k, tweaked_indices[i][0]:tweaked_indices[i][1]],
                            datacube[j, 2, k, tweaked_indices[i][0]:tweaked_indices[i][1]],
                            pseudo_continuum
                        )
                        pbar.update(1)

        return parameter_maps, reference_wavelengths, tweaked_indices, mean_profile, wavelength_array

    def package_analysis(
            self, analysis_maps: np.ndarray, rwvls: list, indices: list,
            mean_profile: np.ndarray, wavelength_array: np.ndarray, reference_file: str
    ) -> str:
        """
        Write SPINOR first-order analysis maps to FITS file.

        Parameters
        ----------
        analysis_maps : numpy.ndarray
        rwvls : list
        indices : list
        mean_profile : numpy.ndarray
        wavelength_array : numpy.ndarray
        reference_file : str

        Returns
        -------

        """
        extnames = [
            "INTENSITY",
            "VELOCITY",
            "WIDTH",
            "NET-CPL",
            "MEAN-CPL",
            "MEAN-LPL"
        ]
        methods = [
            "MOMENT-ANALYSIS",
            "MOMENT-ANALYSIS",
            "MOMENT-ANALYSIS",
            "SOLANKI-MONTAVON",
            "MARTINEZ-PILLET",
            "MARTINEZ-PILLET"
        ]
        method_comments = [
            "",
            "",
            "",
            "1993",
            "2011",
            "2011"
        ]
        # Write a FITS file per line selected. Different than FIRS/HSG.
        # Case where there're no selected lines
        outfile = ""
        for i in range(len(rwvls)):
            # Gonna crib the headers from the reduced science data file
            with fits.open(reference_file) as hdul:
                hdr0 = hdul[0].header.copy()
                hdr1 = hdul[1].header.copy()
                del hdr1['CDELT3']
                del hdr1['CTYPE3']
                del hdr1['CUNIT3']
                del hdr1['CRVAL3']
                del hdr1['CRPIX3']
            ext0 = fits.PrimaryHDU()
            ext0.header = hdr0
            ext0.header["BTYPE"] = "Derived"
            del ext0.header["BUNIT"]
            prsteps = [x for x in ext0.header.keys() if "PRSTEP" in x]
            ext0.header.insert(
                prsteps[-1],
                ("PRSTEP{0}".format(len(prsteps) + 1), "SPEC-ANALYSIS", "User-Chosen Spectral ROI"),
                after=True
            )
            ext0.header.insert(
                "WAVEMIN",
                ("REFWVL", round(rwvls[i], 3), "Reference Wavelength Value")
            )
            ext0.header["WAVEMIN"] = (round(wavelength_array[indices[i][0]], 3), "Lower Bound for Analysis")
            ext0.header["WAVEMAX"] = (round(wavelength_array[indices[i][1]], 3), "Upper Bound for Analysis")
            ext0.header["COMMENT"] = "File contains derived parameters from moment analysis and polarization analysis"
            fits_hdus = [ext0]
            for j in range(analysis_maps.shape[1]):
                ext = fits.ImageHDU(analysis_maps[i, j, :, :])
                ext.header = hdr1.copy()
                ext.header['DATE-OBS'] = ext0.header['STARTOBS']
                ext.header['DATE-END'] = ext0.header['ENDOBS']
                dt = (np.datetime64(ext0.header['ENDOBS']) - np.datetime64(ext0.header['STARTOBS'])) / 2
                date_avg = (np.datetime64(ext0.header['STARTOBS']) + dt).astype(str)
                ext.header['DATE-AVG'] = (date_avg, "UTC, time at map midpoint")
                ext.header['EXTNAME'] = extnames[j]
                ext.header["METHOD"] = (methods[j], method_comments[j])
                fits_hdus.append(ext)

            ext_wvl = fits.ImageHDU(wavelength_array)
            ext_wvl.header['EXTNAME'] = 'lambda-coordinate'
            ext_wvl.header['BTYPE'] = 'lambda axis'
            ext_wvl.header['BUNIT'] = '[AA]'
            ext_wvl.header['COMMENT'] = "Reference Wavelength Array. For use with reference profile and WAVEMIN/MAX."
            fits_hdus.append(ext_wvl)

            ext_ref = fits.ImageHDU(mean_profile)
            ext_ref.header['EXTNAME'] = 'reference-profile'
            ext_ref.header['BTYPE'] = 'Intensity'
            ext_ref.header['BUNIT'] = 'Corrected DN'
            ext_ref.header['COMMENT'] = "Mean spectral profile. For use with WAVEMIN/MAX."
            fits_hdus.append(ext_ref)

            date, time = ext0.header['STARTOBS'].split("T")
            date = date.replace("-", "")
            time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
            outname = self.parameter_map_pattern.format(
                date,
                time,
                round(ext0.header['WAVEMIN'], 2),
                round(ext0.header['WAVEMAX'], 2)
            )
            outfile = os.path.join(self.final_dir, outname)
            fits_hdu_list = fits.HDUList(fits_hdus)
            fits_hdu_list.writeto(outfile, overwrite=True)

        return outfile

    @staticmethod
    def i2quv_crosstalk(stokes_i: np.ndarray, stokes_quv: np.ndarray) -> np.ndarray:
        """
        Corrects for Stokes-I => QUV crosstalk. In the old pipeline, this was done by
        taking the ratio of a continuum section in I, and in QUV, then subtracting
        QUV_nu = QUV_old - ratio * I.

        We're going to take a slightly different approach. Instead of a single ratio value,
        we'll use a line, mx+b, such that QUV_nu = QUV_old - (mx+b)*I.
        We'll solve for m, b such that a second line m'x+b' fit to QUV_nu has m'=b'=0

        Parameters
        ----------
        stokes_i : numpy.ndarray
            1D array of Stokes-I
        stokes_quv : numpy.ndarray
            1D array of Stokes-Q, U, or V

        Returns
        -------
        corrected_quv : numpy.ndarray
            1D array containing the Stokes-I crosstalk-corrected Q, U or V profile.

        """

        def model_function(list_of_params, i, quv):
            """Fit model"""
            xrange = np.arange(len(i))
            ilinear = list_of_params[0] * xrange + list_of_params[1]
            return quv - ilinear * i

        def error_function(list_of_params, i, quv):
            """Error function"""
            quv_corr = model_function(list_of_params, i, quv)
            xrange = np.arange(len(i))
            polyfit = np.polyfit(xrange, quv_corr, 1)
            return (xrange * polyfit[0] + polyfit[1]) - np.zeros(len(i))

        fit_result = scopt.least_squares(
            error_function,
            x0=np.array([0, 0]),
            args=[stokes_i[50:-50], stokes_quv[50:-50]],
            jac='3-point', tr_solver='lsmr'
        )

        ilinear_params = fit_result.x

        corrected_quv = stokes_quv - (np.arange(len(stokes_i)) * ilinear_params[0] + ilinear_params[1]) * stokes_i

        return corrected_quv, ilinear_params

    @staticmethod
    def residual_i2quv_crosstalk(stokes_i: np.ndarray, stokes_quv: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Residual crosstalks are becoming a problem in Stokes-Q/U particularly.
        The majority is corrected by the i2quv_crosstalk() function, meaning that
        the continuum is pretty reliably at 0, but line residuals are showing.
        So for our Stokes-I template, we'll need to subtract off the continuum.
        Problem is that 8542 and other chromospheric lines can go into emission.
        So we'll fit the points between the 60th and 95th percentile in Stokes-I,
        then subtract that off of the Stokes-I profile.

        Parameters
        ----------
        stokes_i : numpy.ndarray
            Stokes-I profile
        stokes_quv : numpy.ndarray
            Stokes-Q, U, or V profile

        Returns
        -------
        corrected_quv : numpy.ndarray
            Residual-corrected Stokes-Q, U, or V profile
        residual_crosstalk : float
        """
        def subtract_continuum(profile):
            arange = np.arange(0, len(profile))
            pct95 = np.percentile(profile, 95)
            pct60 = np.percentile(profile, 60)
            mask = (profile < pct60) | (profile > pct95)
            arange_cut = arange[~mask]
            profile_cut = profile[~mask]
            pfit = np.polyfit(arange_cut, profile_cut, 1)
            return profile - ((arange * pfit[0]) + pfit[1])

        continuum_subtracted = subtract_continuum(stokes_i)
        corrected_quv, residual_crosstalk = self.v2qu_crosstalk(continuum_subtracted, stokes_quv)

        return corrected_quv, residual_crosstalk

    @staticmethod
    def v2qu_crosstalk(stokes_v: np.ndarray, stokes_qu: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Contrary to I->QUV crosstalk, we want the Q/U profiles to be dissimilar to V.
        Q in particular is HEAVILY affected by crosstalk from V.
        Take the assumption that QU = QU - aV, where a is chosen such that the error between
        QU & V is maximized
        Parameters
        ----------
        stokes_v : numpy.ndarray
            Stokes-V profile
        stokes_qu : numpy.ndarray
            Stokes Q or U profile

        Returns
        -------
        corrected_qu : numpy.ndarray
            Crosstalk-corrected Q or U profile
        """

        def model_function(param, v, qu):
            """Fit model"""
            return qu - param * v

        def error_function(param, v, qu):
            """Error function
            We'll use cosine similarity for this, as a cosine similarity of 0
            should indicate completely orthogonal, i.e., dissimilar vectors
            """
            qu_corr = model_function(param, v, qu)

            return np.dot(v, qu_corr) / (np.linalg.norm(v) * np.linalg.norm(qu_corr))

        try:
            fit_result = scopt.least_squares(
                error_function,
                x0=0,
                args=[stokes_v[50:-50], stokes_qu[50:-50]],
                bounds=[-0.5, 0.5]
            )
            v2qu_crosstalk = fit_result.x

        except ValueError:
            # No fit found; usually happens with NaNs or Infs
            v2qu_crosstalk = 0

        corrected_qu = stokes_qu - v2qu_crosstalk * stokes_v

        return corrected_qu, v2qu_crosstalk

    @staticmethod
    def internal_crosstalk_2d(base_image: np.ndarray, contamination_image: np.ndarray) -> float:
        """
        Determines a single crosstalk value for a pair of 2D images.
        Minimizes the linear correlation between:
            baseImage - crosstalk_value * contaminationImage
                and
            contaminationImage
        The v2qu_crosstalk function should be used for individual vectors (uses cosine similarity,
        which scales to 2D poorly). This should be used as an inital guess, with v2qu_crosstalk providing
        fine corrections.

        Parameters
        ----------
        base_image : numpy.ndarray
            2D image of a spatially-resolved Stokes vector
        contamination_image : numpy.ndarray
            2D image of a different spatially-resolved Stokes vector that is contaminating baseImage

        Returns
        -------
        crosstalk_value : float
            Value that, when baseImage - crosstalk_value*contaminationImage is considered, minimizes correlation
        """

        def model_function(param, contam, img):
            """Fit model"""
            return img - param * contam

        def error_function(param, contam, img):
            """Error function
            We'll use cosine similarity for this, as a cosine similarity of 0
            should indicate completely orthogonal, i.e., dissimilar vectors
            """
            contam_corr = model_function(param, contam, img)
            lin_corr = np.nansum(contam_corr * contam) / np.sqrt(np.nansum(contam_corr ** 2) * np.nansum(contam ** 2))

            return lin_corr

        # Clean up array for correlation
        base_image = np.abs(base_image) - np.nanmean(np.abs(base_image))
        contamination_image = np.abs(contamination_image) - np.nanmean(np.abs(contamination_image))
        fit_result = scopt.least_squares(
            error_function,
            x0=0,
            args=[contamination_image[50:-50, 50:-50], base_image[50:-50, 50:-50]],
            bounds=[-1, 1]
        )

        crosstalk_value = fit_result.x

        return crosstalk_value

    def tweak_wavelength_calibration(self, reference_profile: np.ndarray) -> np.ndarray:
        """
        Determines wavelength array from grating parameters and FTS reference

        Parameters
        ----------
        reference_profile : numpy.ndarray
            1D array containing a reference spectral profile from SPINOR

        Returns
        -------
        wavelength_array : numpy.ndarray
            1D array containing corresponding wavelengths.

        """

        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens, slit_width=self.slit_width,
        )

        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        # Same selection process as in flat fielding.
        apx_wavemin = self.central_wavelength - np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax = self.central_wavelength + np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemin -= 30 * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax += 30 * grating_params['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apx_wavemin, apx_wavemax)

        fts_core = sorted(np.array(self.fts_line_cores))
        if self.flip_wave:
            spinor_line_cores = sorted(self.slit_edges[1] - np.array(self.spinor_line_cores))
        else:
            spinor_line_cores = sorted(np.array(self.spinor_line_cores))

        fts_core_waves = [scinterp.CubicSpline(np.arange(len(fts_wave)), fts_wave)(lam) for lam in fts_core]
        # Update SPINOR selected line cores by redoing core finding with wide, then narrow range
        spinor_line_cores = np.array([
            spex.find_line_core(
                reference_profile[int(lam) - 10:int(lam) + 11]
            ) + int(lam) - 10 for lam in spinor_line_cores
        ])
        spinor_line_cores = np.array([
            spex.find_line_core(
                reference_profile[int(lam) - 5:int(lam) + 7]
            ) + int(lam) - 5
            for lam in spinor_line_cores
        ])
        angstrom_per_pixel = np.abs(fts_core_waves[1] - fts_core_waves[0]) / np.abs(spinor_line_cores[1] - spinor_line_cores[0])
        zerowvl = fts_core_waves[0] - (angstrom_per_pixel * spinor_line_cores[0])
        wavelength_array = (np.arange(0, len(reference_profile)) * angstrom_per_pixel) + zerowvl
        return wavelength_array

    def spherical_coordinate_transform(self, telescope_angles: list) -> list:
        """
        Transforms from telescope pointing to parallatic angle using the site latitude

        Parameters
        ----------
        telescope_angles : list
            List of telescope angles. In order, these should be (telescope_azimuth, telescope_elevation)

        Returns
        -------
        coordinate_angles : list
            List of telescope angles. In order, these will be (hour_angle, declination, parallatic angle)

        """

        sin_lat = np.sin(self.site_latitude * np.pi / 180.)
        cos_lat = np.cos(self.site_latitude * np.pi / 180.)

        sin_az = np.sin(telescope_angles[0] * np.pi / 180.)
        cos_az = np.cos(telescope_angles[0] * np.pi / 180.)

        sin_el = np.sin(telescope_angles[1] * np.pi / 180.)
        cos_el = np.cos(telescope_angles[1] * np.pi / 180.)

        sin_x = -cos_el * sin_az
        cos_x = sin_el * cos_lat - cos_el * cos_az * sin_lat

        sin_y = sin_el * sin_lat + cos_el * cos_az * cos_lat
        sin_z = cos_lat * sin_az
        cos_z = sin_lat * cos_el - sin_el * cos_lat * cos_az

        x = np.arctan(sin_x / cos_x)
        y = np.arcsin(sin_y)
        z = -np.arctan(sin_z / cos_z)

        coordinate_angles = [x, y, z]

        return coordinate_angles

    def get_telescope_matrix(self, telescope_geometry: list, reference_frame: float) -> np.ndarray:
        """
        Gets telescope matrix from IDL save (2010 matrix) or numpy save (TBD, hopefully we measure it in the future)
        file. Returns the Mueller matrix of the telescope from these measurements.

        Parameters
        ----------
        telescope_geometry : numpy.ndarray
            3-element vector containing the coelostat azimuth, coelostat elevation, and Coude table angle
        reference_frame : float
            Spectrograph reference frame. Different for polcals and observations, for reasons I'm not totally clear on.
            ---- Figured out why ---- Polcals, the telescope matrix takes you up to the insertion of the linear
            polarizer and retarder units, which are just above the exit window. For observations, the telescope
            matrix includes the plane change to get to the table.

        Returns
        -------
        tmatrix : numpy.ndarray
            4x4 Mueller matrix of telescope parameters
        """

        filename, filetype = os.path.splitext(self.t_matrix_file)
        if "idl" in filetype:
            txparams = scio.readsav(self.t_matrix_file)
        else:
            txparams = scio.readsav(self.t_matrix_file)

        # In these files, the telescope parameters are 'tt'. The structure of tt is a bit odd:
        # tt[0]: Number of wavelength windows
        # tt[1]: Entrance Window Retardance Orientation
        # tt[2]: Exit Window Retarder Orientation
        # tt[3]: Polarimeter-Telescope reference frame rotation
        # tt[4]: Entrance window polarizer angle offset
        # Then, if "i" is the wavelength index, starting with 0:
        # Given that there are 7 more parameters, this is equivalent to indexing by [IDX::7]
        # tt[5+i*7]: Wavelength [AA]
        # tt[6+i*7]: Entrance Window Retardance
        # tt[7+i*7]: Exit Window Retardance
        # tt[8+i*7]: Coelostat Reflectances
        # tt[9+i*7]: Retardance of Coelostat
        # tt[10+i*7]: Primary Mirror Reflectance
        # tt[11+i*7]: Primary Mirror Retardance

        entrance_window_orientation = txparams['tt'][1] * np.pi / 180
        exit_window_orientation = txparams['tt'][2] * np.pi / 180
        ref_frame_orientation = reference_frame * np.pi/180
        entrance_window_polarizer_offset = txparams['tt'][4]

        wvls = txparams['tt'][5::7]
        entrance_window_retardance = scinterp.interp1d(
            wvls, txparams['tt'][6::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength) * np.pi / 180
        exit_window_retardance = scinterp.interp1d(
            wvls, txparams['tt'][7::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength) * np.pi / 180
        coelostat_reflectance = scinterp.interp1d(
            wvls, txparams['tt'][8::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength)
        coelostat_retardance = scinterp.interp1d(
            wvls, txparams['tt'][9::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength) * np.pi / 180
        primary_reflectance = scinterp.interp1d(
            wvls, txparams['tt'][10::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength)
        primary_retardance = scinterp.interp1d(
            wvls, txparams['tt'][11::7], kind='linear', fill_value='extrapolate'
        )(self.central_wavelength) * np.pi / 180

        phi_elevation = (telescope_geometry[1] + 90) * np.pi / 180
        phi_azimuth = (telescope_geometry[2] - telescope_geometry[0] - 30.) * np.pi / 180.

        # In order, the DST optical train is:
        #   1.) Entrance Window (Retarder)
        #   2.) Elevation Coelostat (Mirror)
        #   3.) Coordinate Transform Horizontal (Rotation)
        #   4.) Azimuth Coelostat (Mirror)
        #   5.) Coordinate Transform Vertical (Rotation)
        #   6.) Primary (Mirror)
        #   7.) Exit Window (Retarder)
        #   8.) Coordinate Transform Horizontal (Rotation)

        entrance_window_mueller = spex.linear_retarder(
            entrance_window_orientation, entrance_window_retardance
        )
        elevation_mueller = spex.mirror(
            coelostat_reflectance, coelostat_retardance
        )
        azel_rotation_mueller = spex.rotation_mueller(phi_elevation)
        azimuth_mueller = spex.mirror(
            coelostat_reflectance, coelostat_retardance
        )
        azvert_rotation_mueller = spex.rotation_mueller(phi_azimuth)
        primary_mueller = spex.mirror(
            primary_reflectance, primary_retardance
        )
        exit_window_mueller = spex.linear_retarder(
            exit_window_orientation, exit_window_retardance
        )
        refframe_rotation_mueller = spex.rotation_mueller(
            ref_frame_orientation
        )

        # There's probably a more compact way to do this,
        # but for now, we'll just go straight down the optical chain
        tmatrix = elevation_mueller @ entrance_window_mueller
        tmatrix = azel_rotation_mueller @ tmatrix
        tmatrix = azimuth_mueller @ tmatrix
        tmatrix = azvert_rotation_mueller @ tmatrix
        tmatrix = primary_mueller @ tmatrix
        tmatrix = exit_window_mueller @ tmatrix
        tmatrix = refframe_rotation_mueller @ tmatrix

        # Normalize the Mueller matrix
        tmatrix /= tmatrix[0, 0]

        return tmatrix

    @staticmethod
    def determine_spectrum_flip(
            fts_spec: np.ndarray, spinor_spex: np.ndarray, spin_pix_per_fts_pix: float,
            spinor_cores: list, fts_cores: list
    ) -> bool:
        """
        Determine if SPINOR spectra are flipped by correlation value against interpolated
        FTS atlas spectrum. Have to interpolate FTS to SPINOR, determine offset via correlation.
        Parameters
        ----------
        fts_spec
        spinor_spex
        spin_pix_per_fts_pix
        spinor_cores
        fts_cores

        Returns
        -------
        bool
            True if the spectrum is flipped, false otherwise

        """

        # As of 2025-01-31, this hasn't been particularly reliable.
        # Altering to clip spectra to the range between the selected lines.
        # 2025-02-11, Still has trouble with broad profiles with few features.
        # Going to increase the range used slightly
        spinor_spex /= spinor_spex.max()
        spinor_edge_pad = 10
        # Edge case cleaning just in case one of the selected lines is near the edge
        if (min(spinor_cores) < 10) & (spinor_spex.shape[0] - 10 < max(spinor_cores)):
            # Pad out to the edge of the beam
            spinor_edge_pad = min([min(spinor_cores), spinor_spex.shape[0] - max(spinor_cores)]) - 1

        spinor_spex = spinor_spex[min(spinor_cores) - spinor_edge_pad:max(spinor_cores) + spinor_edge_pad]

        fts_edge_pad = int(spinor_edge_pad / spin_pix_per_fts_pix)
        fts_spec = fts_spec[int(min(fts_cores) - fts_edge_pad):int(max(fts_cores) + fts_edge_pad)]

        fts_interp = scinterp.interp1d(
            np.arange(0, fts_spec.shape[0] * spin_pix_per_fts_pix, spin_pix_per_fts_pix),
            fts_spec,
            kind='linear',
            fill_value='extrapolate'
        )(np.arange(len(spinor_spex)))

        fts_interp_reversed = fts_interp[::-1]

        lin_corr = np.nansum(
            fts_interp * spinor_spex
        ) / np.sqrt(np.nansum(fts_interp ** 2) * np.nansum(spinor_spex ** 2))

        lin_corr_rev = np.nansum(
            fts_interp_reversed * spinor_spex
        ) / np.sqrt(np.nansum(fts_interp_reversed ** 2) * np.nansum(spinor_spex ** 2))

        if lin_corr_rev > lin_corr:
            return True
        else:
            return False

    @staticmethod
    def despike_image(image: np.ndarray, footprint: tuple = (5, 1), spike_range: tuple = (0.75, 1.25)) -> np.ndarray:
        """Removes spikes in image caused by cosmic rays, hot pixels, etc. Works off median filtering image.
        Placeholder for now. Will be replaced by a more robust function in the future.

        Parameters
        ----------
        image : numpy.ndarray
            ND image array. Length of footpoint should match the number of axes
        footprint : tuple
            Footpoint used in scipy.ndimage.median_filter to create median-smoothed image
        spike_range : tuple
            Range to clip hot pixels from. Pixels in image/median_image exceeding range will be replaced
            by corresponding pixels in median_image

        Returns
        -------
        despiked_image : numpy.ndarray
        """

        medfilt_image = scind.median_filter(image, size=footprint)
        spike_image = image / medfilt_image
        despiked_image = image.copy()
        despiked_image[
            (spike_image > max(spike_range)) | (spike_image < min(spike_range))
            ] = medfilt_image[(spike_image > max(spike_range)) | (spike_image < min(spike_range))]
        return despiked_image
