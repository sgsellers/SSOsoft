import astropy.io.fits as fits
import astropy.units as u
import configparser
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import scipy.ndimage as scind
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import scipy.io as scio
import scipy.optimize as scopt
import tqdm
from astropy.constants import c
import warnings

c_kms = c.value / 1e3
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from . import spectraTools as spex

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
    configFile : str
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

    def __init__(self, camera: str, configFile: str) -> None:
        """

        Parameters
        ----------
        camera : str
            String containing camera name
        configFile : str
            Path to configuration file
        """

        try:
            f = open(configFile, 'r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.configFile = configFile
        self.camera = camera.upper()

        self.solarDark = None
        self.solarFlat = None
        self.lampGain = None
        self.combinedGainTable = None
        self.combinedCoarseGainTable = None
        self.polcalVecs = None
        self.tMatrix = None
        self.flipWave = False

        # Locations
        self.indir = ""
        self.finalDir=""
        self.reducedFilePattern = None
        self.parameterMapPattern = None
        self.polcalFile = None
        self.solarFlatFileList = []
        self.solarFlatFile = None
        self.lampFlatFile = None
        self.scienceFileList = []
        # Need to re-combine longer map series that were split by SPINOR FITS daemons
        self.scienceMapFileList = []
        self.scienceFiles = None
        self.tMatrixFile = None

        self.lineGridFile = None
        self.targetFile = None

        # For saving the reduced calibration files:
        self.solarGainReduced = None # we'll include dark currents in these files
        self.lampGainReduced = ""
        self.txMatrixReduced = ""

        # Setting up variables to be filled:
        self.beamEdges = None
        self.slitEdges = None
        self.hairlines = None
        self.beam1Xshift = None
        self.beam1Yshift = None
        self.spinorLineCores = None
        self.ftsLineCores = None
        self.flipWaveIdx = 1

        # Polcal-specific variables
        self.polcalProcessing = True
        self.calcurves = None
        self.txmat = None
        self.inputStokes = None
        self.txchi = None
        self.txmat00 = None
        self.txmatinv = None


        # Some default vaules
        self.nhair = 4
        self.beamThreshold = 0.5
        self.hairlineWidth = 3
        self.grating_rules = 308.57 # lpmm
        self.blaze_angle = 52
        self.nSubSlits = 10
        self.verbose = False
        self.v2q = True
        self.v2u = True
        self.u2v = True
        self.despike = False
        self.despikeFootprint = (1, 5, 1)
        self.plot = False
        self.saveFigs = False
        self.crosstalkContinuum = None

        # Can be pulled from header:
        self.grating_angle = None
        self.slit_width = None

        # Must be set in config file, or surmised from config file
        self.pixel_size = None # 16 um for Sarnoffs, 25 um for Flirs
        self.centralWavelength = None # Should identify a spectral line, i.e., 6302, 8542
        self.spectral_order = None # To-do, solve the most likely spectral order from the grating info. Function exists.


        # Default polarization modulation from new modulator (2024-09)
        # I: ++++++++
        # Q: --++--++
        # U: +--++--+
        # V: +----+++
        self.polDemod = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, -1, -1, -1, 1, 1, 1]
        ], dtype=int)

        self.DSTLatitude = 32.786

        # Expressed as a fraction of mean(I). For polcals
        self.ilimit = [0.5, 1.5]
        # It's about a quarter-wave plate.
        self.calRetardance = 90

        # Basic SPINOR Feed Optics
        self.slitCameraLens = 780 # mm, f.l. of usual HSG feed lens
        self.dstPlateScale = 3.76 # asec/mm
        self.dstCollimator = 1559 # mm, f.l., of DST Port 4 Collimator mirror
        self.spectrographCollimator = 3040 # mm, f.l., of the SPINOR/HSG post-slit collimator
        self.cameraLens = 1700 # mm, f.l., of the SPINOR final camera lenses

        return


    def spinor_assert_file_list(self, flist: list) -> None:
        assert (len(flist) != 0), "List contains no matches."


    def spinor_run_calibration(self) -> None:
        """Main SPINOR calibration module"""

        self.spinor_configure_run()
        if self.verbose:
            print("Found {0} science map files in base directory:\n{1}\nReduced files will be saved to:\n{2}".format(
                len(self.scienceFileList), self.indir, self.finalDir
            ))
        for index in range(len(self.scienceMapFileList)):
            if self.verbose:
                for subindex in range(len(self.scienceMapFileList[index])):
                    if subindex == 0:
                        print(
                            "Proceeding with calibration of: {0}".format(
                                os.path.split(self.scienceMapFileList[index][subindex])[1]
                            )
                        )
                    else:
                        print(
                            "                                {0}".format(
                                os.path.split(self.scienceMapFileList[index][subindex])[1]
                            )
                        )
                print("Using Solar Flat File: {0}".format(os.path.split(self.solarFlatFileList[index])[1]))
                if self.lampFlatFile is not None:
                    print("Using Lamp Flat File: {0}".format(os.path.split(self.lampFlatFile)[1]))
                else:
                    print("No Lamp Flat File")
                print("Using Polcal File: {0}".format(os.path.split(self.polcalFile)[1]))
                if self.plot:
                    print("Plotting is currently ON.")
                    if self.saveFigs:
                        print("Plots will be saved at:\n{0}".format(self.finalDir))
                    else:
                        print("Plots will NOT be saved.")
                print("===========================\n\n")
            self.__init__(self.camera, self.configFile)
            self.spinor_configure_run()
            self.spinor_get_cal_images(index)
            if self.plot:
                plt.pause(2)
                plt.close("all")
            self.scienceFiles = self.scienceMapFileList[index]
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
        config.read(self.configFile)

        # Locations [Required]
        self.indir = config[self.camera]["rawFileDirectory"]
        self.finalDir = config[self.camera]["reducedFileDirectory"]
        self.reducedFilePattern = config[self.camera]["reducedFilePattern"]
        self.parameterMapPattern = config[self.camera]["reducedParameterMapPattern"]

        # Optional calibration file definitions. If these are left undefined, the directory parser below
        # sets these, however, it may be desirable to specify a flat file under certain circumstances.

        # [Optional] entry
        self.polcalFile = config[self.camera]["polcalFile"] if (
            "polcalfile" in config[self.camera].keys()
        ) else None
        # Reset to None for case where the key is present with an empty string
        self.polcalFile = None if self.polcalFile == "" else self.polcalFile
        self.solarFlatFile = config[self.camera]["solarFlatFile"] if (
            "solarflatfile" in config[self.camera].keys()
        ) else None
        self.solarFlatFile = None if self.solarFlatFile == "" else self.solarFlatFile
        self.lampFlatFile = config[self.camera]["lampFlatFile"] if (
            "lampflatfile" in config[self.camera].keys()
        ) else None
        self.lampFlatFile = None if self.lampFlatFile == "" else self.lampFlatFile
        self.scienceFiles = [config[self.camera]["scienceFile"]] if (
            "sciencefile" in config[self.camera].keys()
        ) else None
        self.scienceFiles = None if self.scienceFiles == "" else self.scienceFiles

        # Required channel-specific params
        self.pixel_size = 16 if "sarnoff" in self.camera.lower() else 25
        self.centralWavelength = float(config[self.camera]["centralWavelength"])
        self.spectral_order = int(config[self.camera]["spectralOrder"])

        # Overrides for some channel-specific default vaules
        self.nSubSlits = int(config[self.camera]["slitDivisions"]) if (
            "slitdivisions" in config[self.camera].keys()
        ) else self.nSubSlits
        self.verbose = config[self.camera]["verbose"] if "verbose" in config[self.camera].keys() else "False"
        if "t" in self.verbose.lower():
            self.verbose = True
        else:
            self.verbose = False
        self.v2q = config[self.camera]["v2q"] if "v2q" in config[self.camera].keys() else "True"
        if "t" in self.v2q.lower():
            self.v2q = True
        else:
            self.v2q = False
        self.v2u = config[self.camera]["v2u"] if "v2u" in config[self.camera].keys() else "True"
        if "t" in self.v2u.lower():
            self.v2u = True
        else:
            self.v2u = False
        self.u2v = config[self.camera]["u2v"] if "u2v" in config[self.camera].keys() else "True"
        if "t" in self.u2v.lower():
            self.u2v = True
        else:
            self.u2v = False
        self.plot = config[self.camera]["plot"] if "plot" in config[self.camera].keys() else "False"
        if "t" in self.plot.lower():
            self.plot = True
        else:
            self.plot = False
        self.saveFigs = config[self.camera]["savePlot"] if "saveplot" in config[self.camera].keys() else "False"
        if "t" in self.saveFigs.lower():
            self.saveFigs = True
        else:
            self.saveFigs = False
        self.despike = config[self.camera]['despike'] if "despike" in config[self.camera].keys() else "False"
        if "t" in self.despike.lower():
            self.despike = True
        else:
            self.despike = False

        self.nhair = int(config[self.camera]["totalHairlines"]) if (
            "totalhairlines" in config[self.camera].keys()
        ) else self.nhair
        self.beamThreshold = float(config[self.camera]["intensityThreshold"]) if (
            "intensitythreshold" in config[self.camera].keys()
        ) else self.beamThreshold
        self.hairlineWidth = float(config[self.camera]["hairlineWidth"]) if (
            "hairlinewidth" in config[self.camera].keys()
        ) else self.hairlineWidth
        self.calRetardance = float(config[self.camera]["calRetardance"]) if (
            "calretardance" in config[self.camera].keys()
        ) else self.calRetardance
        self.cameraLens = float(config[self.camera]["cameraLens"]) if (
            "cameralens" in config[self.camera].keys()
        ) else self.cameraLens
        if "polcalclipthreshold" in config[self.camera].keys():
            if config[self.camera]['polcalClipThreshold'] != "":
                self.ilimit = [float(frac) for frac in config[self.camera]["polcalClipThresold"].split(",")]
        self.polcalProcessing = config[self.camera]["polcalProcessing"] if (
            "polcalProcessing" in config[self.camera].keys()
        ) else "True"
        if "t" in self.polcalProcessing.lower():
            self.polcalProcessing = True
        else:
            self.polcalProcessing = False

        # Case where someone wants the old crosstalk determination, and has defined it themselves
        if "crosstalkcontinuum" in config[self.camera].keys():
            if config[self.camera]['crosstalkContinuum'] != "":
                self.crosstalkContinuum = [int(idx) for idx in config[self.camera]['crosstalkContinuum'].split(",")]

        # Required global values
        self.tMatrixFile = config["SHARED"]["tMatrixFile"]

        # Overrides for Global defaults
        self.grating_rules = float(config["SHARED"]["gratingRules"]) if (
            "gratingrules" in config["SHARED"].keys()
        ) else self.grating_rules
        self.blaze_angle = float(config["SHARED"]["blazeAngle"]) if (
            "blazeangle" in config["SHARED"].keys()
        ) else self.blaze_angle

        self.DSTLatitude = float(config["SHARED"]["telescopeLatitude"]) if (
            "telescopelatitude" in config["SHARED"].keys()
        ) else self.DSTLatitude
        self.slitCameraLens = float(config["SHARED"]["slitCameraLens"]) if (
            "slitcameralens" in config["SHARED"].keys()
        ) else self.slitCameraLens
        self.dstPlateScale = float(config["SHARED"]["basePlateScale"]) if (
            "baseplatescale" in config["SHARED"].keys()
        ) else self.dstPlateScale
        self.dstCollimator = float(config["SHARED"]["telescopeCollimator"]) if (
            "telescopecollimator" in config["SHARED"].keys()
        ) else self.dstCollimator
        self.spectrographCollimator = float(config["SHARED"]["spectrographCollimator"]) if (
            "spectrographcollimator" in config["SHARED"].keys()
        ) else self.spectrographCollimator

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
            self.polDemod = np.array([
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
        scienceFiles = []
        solarFlats = []
        lampFlats = []
        polcalFiles = []
        lineGrids = []
        targetFiles = []

        for file in filelist:
            with fits.open(file) as hdul:
                if ("USER1" in hdul[1].header['PT4_FS']) & ("map" in file):
                    scienceFiles.append(file)
                elif ("USER2" in hdul[1].header['PT4_FS']) & ("map" in file):
                    lineGrids.append(file)
                elif ("TARGET" in hdul[1].header['PT4_FS']) & ("map" in file):
                    targetFiles.append(file)
                elif ("sun.flat" in file) & (len(hdul) >= 16):
                    solarFlats.append(file)
                elif ("lamp.flat" in file) & (len(hdul) >= 16):
                    lampFlats.append(file)
                elif ("cal" in file) & (len(hdul) >= 16):
                    polcalFiles.append(file)
        # Select polcal, linegrid, and target files by total filesize
        if self.polcalFile is None:
            polcalFilesizes = np.array([os.path.getsize(pf) for pf in polcalFiles])
            self.polcalFile = polcalFiles[polcalFilesizes.argmax()] if len(polcalFilesizes) != 0 else None
        else:
            self.polcalFile = os.path.join(self.indir, self.polcalFile)
        lineGridFilesizes = np.array([os.path.getsize(lg) for lg in lineGrids])
        self.lineGridFile = lineGrids[lineGridFilesizes.argmax()] if len(lineGridFilesizes) != 0 else None
        targetFilesizes = np.array([os.path.getsize(tg) for tg in targetFiles])
        self.targetFile = targetFiles[targetFilesizes.argmax()] if len(targetFilesizes) != 0 else None

        # Case where a science file is defined, rather than allowing the code to run cals on the full day
        if self.scienceFiles is not None:
            self.scienceFileList = [os.path.join(self.indir, self.scienceFiles)]
        else:
            self.scienceFileList = scienceFiles

        # 2025-01-27: Slight problem. SPINOR won't put more than 252 steps in a single file. Longer maps are split
        # across multiple files. It would be good to have these combined into a single file during reductions.
        # Unfortunately, nothing in the file headers indicates that this will happen. The only indicator is in the
        # filename. Each map in a series has the same map number in the filename, i.e., the file name is
        # YYMMDD.HHMMSS.0.cccXX.c-hrt.map.NNNN.fits
        # NNNN is the same for different halves of the map in the same series....
        # Get the map number...
        mapList = [x.split("c-hrt")[1] for x in self.scienceFileList]
        # Deduplicate
        mapList = sorted(list(set(mapList)))
        self.scienceMapFileList = [sorted(glob.glob(os.path.join(self.indir, "*"+x))) for x in mapList]
        scienceStartTimes = np.array(
            [
                fits.open(x[0])[1].header['DATE-OBS'] for x in self.scienceMapFileList
            ],
            dtype='datetime64[ms]'
        )

        # Allow user to override and choose flat files to use
        if self.solarFlatFile is not None:
            self.solarFlatFileList = [os.path.join(self.indir, self.solarFlatFile)] * len(self.scienceMapFileList)
            self.solarGainReduced = [os.path.join(
                self.finalDir, "{0}_{1}_SOLARGAIN.fits"
            ).format(self.camera, 0)] * len(self.scienceMapFileList)
        else:
            solarFlatStartTimes = np.array(
                [
                    fits.open(x)[1].header['DATE-OBS'] for x in solarFlats
                ],
                dtype='datetime64[ms]'
            )
            self.solarFlatFileList = [
                solarFlats[spex.find_nearest(solarFlatStartTimes, x)] for x in scienceStartTimes
            ]
            self.solarGainReduced = [
                os.path.join(
                    self.finalDir,
                    "{0}_{1}_SOLARGAIN.fits"
                ).format(self.camera, spex.find_nearest(solarFlatStartTimes, x)) for x in scienceStartTimes
            ]
        if self.lampFlatFile is None:
            lampFlatFilesizes = np.array([os.path.getsize(lf) for lf in lampFlats])
            self.lampFlatFile = lampFlats[lampFlatFilesizes.argmax()] if len(lampFlatFilesizes) != 0 else None
        else:
            self.lampFlatFile = os.path.join(self.indir, self.lampFlatFile)

        # Set up the list of reduced dark/gain/lamp/polcal files, so we can save out or restore previous calibrations

        self.lampGainReduced = os.path.join(
            self.finalDir,
            "{0}_LAMPGAIN.fits"
        ).format(self.camera)
        self.txMatrixReduced = os.path.join(
            self.finalDir,
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
        averageDark : numpy.ndarray
            2D numpy array with the average dark current per pixel.
        """
        # hdulist might have leading empty extension.
        # Each extension has shape (8, ny, nx)
        # One image per mod state
        averageDark = np.zeros((hdulist[-1].data.shape[1], hdulist[-1].data.shape[2]))
        darkctr = 0
        for hdu in hdulist:
            if "PT4_FS" in hdu.header.keys():
                if "DARK" in hdu.header['PT4_FS']:
                    if self.despike:
                        data = self.despike_image(hdu.data, footprint=self.despikeFootprint)
                        averageDark += np.nanmean(data, axis=0)
                    else:
                        averageDark += np.nanmean(hdu.data, axis=0)
                    darkctr += 1
        averageDark /= darkctr
        return averageDark


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
        averageFlat : numpy.ndarray
            Averaged flat field
        """
        # hdulist might have leading empty extension.
        # Each extension has shape (8, ny, nx)
        # One image per mod state
        averageFlat = np.zeros((hdulist[-1].data.shape[1], hdulist[-1].data.shape[2]))
        flatctr = 0
        for hdu in hdulist:
            if "PT4_FS" in hdu.header.keys():
                if "DARK" not in hdu.header['PT4_FS']:
                    if self.despike:
                        data = self.despike_image(hdu.data, footprint=self.despikeFootprint)
                        averageFlat += np.nanmean(data, axis=0)
                    else:
                        averageFlat += np.nanmean(hdu.data, axis=0)
                    flatctr += 1
        averageFlat /= flatctr
        return averageFlat


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
            poldata = self.despike_image(poldata, footprint=self.despikeFootprint)
        for i in range(stokes.shape[0]):
            for j in range(poldata.shape[0]):
                stokes[i] += self.polDemod[i, j] * poldata[j, :, :]
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
        if os.path.exists(self.lampGainReduced):
            with fits.open(self.lampGainReduced) as hdu:
                self.lampGain = hdu[0].data
        # Create new lamp gain and save to file
        elif self.lampFlatFile is not None:
            with fits.open(self.lampFlatFile) as lhdul:
                lampDark = self.spinor_average_dark_from_hdul(lhdul)
                lampFlat = self.spinor_average_flat_from_hdul(lhdul)
            cleanedLampFlat = self.clean_lamp_flat(lampFlat - lampDark)
            self.lampGain = cleanedLampFlat / np.nanmedian(cleanedLampFlat)
            hdu = fits.PrimaryHDU(self.lampGain)
            hdu.header["DATE"] = np.datetime64("now").astype(str)
            hdu.header["COMMENT"] = "Created from file {0}".format(self.lampFlatFile)
            fits.HDUList([hdu]).writeto(self.lampGainReduced, overwrite=True)
        # No lamp gain available for the day. Creates an array of ones to mimic a lamp gain
        else:
            self.lampGain = np.ones(self.solarDark.shape)
            warnings.warn("No lamp flat available. Reduced data may show strong internal fringes.")

        return


    def clean_lamp_flat(self, lampFlatImage: np.ndarray) -> np.ndarray:
        """
        Cleans lamp flat image by finding hairlines and removing them via scipy.interpolate.griddata
        Uses spectraTools.detect_beams_hairlines to get hairlines, replace them with NaNs and interpolate
        over the NaNs.

        Unfortunately griddata is one of those scipy functions that's just... really slow.
        I'll keep researching faster methods. Might be worth modifying the solar gain routines to iteratively
        remove hairlines in the same manner that the solar spectrum is removed.

        Parameters
        ----------
        lampFlatImage : numpy.ndarray
            Dark-subtracted lamp flat image

        Returns
        -------
        cleanedLampFlatImage : numpy.ndarray
            Lamp flat with hairlines removed

        """

        _, _, hairlines = spex.detect_beams_hairlines(
            lampFlatImage,
            threshold=self.beamThreshold, hairline_width=self.hairlineWidth,
            expected_hairlines=self.nhair, expected_beams=2,
            fallback=True # Hate relying on it, but safer for now
        )
        # Reset recursive counter since we'll need to use the function again later
        spex.detect_beams_hairlines.num_calls=0
        for line in hairlines:
            # range + 1 to compensate for casting a float to an int
            lampFlatImage[int(line - self.hairlineWidth):int(line + self.hairlineWidth + 1), :] = np.nan
        x = np.arange(0, lampFlatImage.shape[1])
        y = np.arange(0, lampFlatImage.shape[0])
        maskedLampFlat = np.ma.masked_invalid(lampFlatImage)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~maskedLampFlat.mask]
        y1 = yy[~maskedLampFlat.mask]
        newFlat = maskedLampFlat[~maskedLampFlat.mask]
        cleanedLampFlatImage = scinterp.griddata(
            (x1, y1),
            newFlat.ravel(),
            (xx, yy),
            method='nearest',
            fill_value=np.nanmedian(lampFlatImage)
        )
        return cleanedLampFlatImage


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
        if os.path.exists(self.solarGainReduced[index]):
            with fits.open(self.solarGainReduced[index]) as hdu:
                self.combinedCoarseGainTable = hdu["COARSE-GAIN"].data
                self.combinedGainTable = hdu["GAIN"].data
                self.beamEdges = hdu["BEAM-EDGES"].data
                self.hairlines = hdu["HAIRLINES"].data
                self.slitEdges = hdu["SLIT-EDGES"].data
                self.beam1Yshift = hdu["BEAM1-SHIFTS"].data[0]
                self.beam1Xshift = hdu["BEAM1-SHIFTS"].data[1]
                self.spinorLineCores = [hdu[0].header['LC1'], hdu[0].header['LC2']]
                self.ftsLineCores = [hdu[0].header['FTSLC1'], hdu[0].header["FTSLC2"]]
                self.flipWaveIdx = hdu[0].header["SPFLIP"]
                if self.flipWaveIdx == -1:
                    self.flipWave = True
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
        if os.path.exists(self.solarGainReduced[index]):
            with fits.open(self.solarGainReduced[index]) as hdu:
                self.solarFlat = hdu["SOLAR-FLAT"].data
                self.solarDark = hdu["SOLAR-DARK"].data
                self.slit_width = hdu[0].header["HSG_SLW"]
                self.grating_angle = hdu[0].header["HSG_GRAT"]
        else:
            self.solarFlatFile = self.solarFlatFileList[index]
            with fits.open(self.solarFlatFile) as fhdul:
                self.solarDark = self.spinor_average_dark_from_hdul(fhdul)
                self.solarFlat = self.spinor_average_flat_from_hdul(fhdul)
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

        self.beamEdges, self.slitEdges, self.hairlines = spex.detect_beams_hairlines(
            self.solarFlat - self.solarDark,
            threshold=self.beamThreshold,
            hairline_width=self.hairlineWidth,
            expected_hairlines=self.nhair,  # Possible that FLIR cams only have one hairline
            expected_beams=2,  # If there's only one beam, use hsgCal
            expected_slits=1, # ...we're not getting a multislit unit for SPINOR.
            fallback=True # FLIR cameras in particular have difficulties with the automated detection
        )
        self.slitEdges = self.slitEdges[0]
        if self.verbose:
            print("Lower Beam Edges in Y: ", self.beamEdges[0])
            print("Upper Beam Edges in Y: ", self.beamEdges[1])
            print("Shared X-range: ", self.slitEdges)
            print("There are {0} hairlines at ".format(self.nhair), self.hairlines)
            print("===========================\n\n")

        # Determine which beam is smaller, clip larger to same size
        smaller_beam = np.argmin(np.diff(self.beamEdges, axis=1))
        larger_beam = np.argmax(np.diff(self.beamEdges, axis=1))
        # If 4 hairlines are detected, clip the beams by the hairline positions.
        # This avoids overclipping the beam, and makes the beam alignment easier.
        # And if the beams are the same size, we can skip this next step.
        if (len(self.hairlines) == 4) and (smaller_beam != larger_beam):
            # It does matter which beam is smaller. Must pair inner & outer hairlines.
            if smaller_beam == 0:
                self.beamEdges[larger_beam, 0] = int(round(
                    self.hairlines[2] - (self.beamEdges[smaller_beam, 1] - self.hairlines[1]), 0
                ))
                self.beamEdges[larger_beam, 1] = int(round(
                    self.hairlines[3] + (self.hairlines[0] - self.beamEdges[smaller_beam, 0]), 0
                ))
            else:
                self.beamEdges[larger_beam, 0] = int(round(
                    self.hairlines[0] - (self.beamEdges[smaller_beam, 1] - self.hairlines[3]), 0
                ))
                self.beamEdges[larger_beam, 1] = int(round(
                    self.hairlines[1] + (self.hairlines[2] - self.beamEdges[smaller_beam, 0]), 0
                ))
        # Mainly for FLIR where one of the hairlines might be clipped by the chip's edge
        elif (len(self.hairlines) != 4) and (smaller_beam != larger_beam):
            self.beamEdges[larger_beam, 0] = int(
                np.nanmean(self.beamEdges[larger_beam, :]) - np.diff(self.beamEdges[smaller_beam, :]) / 2
            )
            self.beamEdges[larger_beam, 1] = int(
                np.nanmean(self.beamEdges[larger_beam, :]) + np.diff(self.beamEdges[smaller_beam, :]) / 2
            )

        # Might still be off by up to 2 in size due to errors in casting float to int
        diff = int(np.diff(np.diff(self.beamEdges, axis=1).flatten(), axis=0)[0])
        self.beamEdges[0, 1] += diff

        self.hairlines = self.hairlines.reshape(2, int(self.nhair / 2))

        beam0 = self.solarFlat[self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]]
        beam1 = np.flipud(
            self.solarFlat[self.beamEdges[1, 0]: self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]]
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
        self.beam1Xshift = np.correlate(
            beam0meanprof, beam1meanprof, mode='full'
        ).argmax() - beam0meanprof.shape[0]

        # Rather than doing a scipy.ndimage.shift, the better way to do it would
        # be to shift the self.beamEdges for beam 1.
        # Need to be mindful of shifts that would move beamEdges out of frame.
        excess_shift = self.beamEdges[1, 1] - yshift - self.solarFlat.shape[0]
        if excess_shift > 0:
            # Y-Shift takes beam out of bounds. Update beamEdges to be at the edge of frame,
            # Store the excess shift for scipy.ndimage.shift at a later date.
            self.beam1Yshift = -excess_shift
            self.beamEdges[1] += -yshift - excess_shift
        else:
            # Y-shift does not take beam out of bound. Update beamEdges, and move on.
            self.beam1Yshift = 0
            self.beamEdges[1] += -yshift

        # Redefine beam1 with new edges. Do not flip, otherwise, everything has to flip
        beam1 = self.solarFlat[self.beamEdges[1, 0]: self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]]

        # Now we need to grab the spectral line we'll be using for the gain table
        # Since this will require popping up a widget, we might as well fine-tune
        # our wavelength scale for the final product, so we don't have to do too
        # many widgets overall. Of course, to grab the right section of the FTS
        # atlas, we need a central wavelength and a wavelength scale... Rather than
        # Having the user figure this out, we can grab all the grating parameters all
        # at once.

        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.centralWavelength, self.spectral_order,
            collimator=self.spectrographCollimator, camera=self.cameraLens, slit_width=self.slit_width,
        )

        beam0LampGainCorrected = (
            beam0 - self.solarDark[self.beamEdges[0, 0]:self.beamEdges[0, 1],
                    self.slitEdges[0]:self.slitEdges[1]]
        ) / self.lampGain[self.beamEdges[0, 0]:self.beamEdges[0, 1],
            self.slitEdges[0]:self.slitEdges[1]]

        beam1LampGainCorrected = (
             beam1 - self.solarDark[self.beamEdges[1, 0]:self.beamEdges[1, 1],
                     self.slitEdges[0]:self.slitEdges[1]]
        ) / self.lampGain[self.beamEdges[1, 0]:self.beamEdges[1, 1],
            self.slitEdges[0]:self.slitEdges[1]]

        avg_profile = np.nanmedian(
            beam0LampGainCorrected[
            int(beam0LampGainCorrected.shape[0] / 2 - 30):int(beam0LampGainCorrected.shape[0] / 2 + 30), :
            ],
            axis=0
        )

        self.spinorLineCores, self.ftsLineCores, self.flipWave = self.spinor_fts_line_select(
            grating_params, avg_profile
        )
        if self.verbose & self.flipWave:
            print("Spectrum flipped along the wavelength axis... Correcting.")
        if self.verbose and not self.flipWave:
            print("Spectrum is not flipped, no correction necessary.")

        # Rather than building in logic every time we need to flip/not flip a spectrum,
        # We'll define a flip index, and slice by it every time. So if we flip, we'll be
        # indexing [::-1]. Otherwise, we'll index [::1], i.e., doing nothing to the array
        if self.flipWave:
            self.flipWaveIdx = -1
        else:
            self.flipWaveIdx = 1

        beam0GainTable, beam0CoarseGainTable, beam0Skews = spex.create_gaintables(
            beam0LampGainCorrected,
            [self.spinorLineCores[0] - 5, self.spinorLineCores[0] + 7],
            hairline_positions=self.hairlines[0] - self.beamEdges[0, 0],
            neighborhood=12,
            hairline_width=self.hairlineWidth / 2
        )

        beam1GainTable, beam1CoarseGainTable, beam1Skews = spex.create_gaintables(
            beam1LampGainCorrected,
            [self.spinorLineCores[0] - 5 - self.beam1Xshift, self.spinorLineCores[0] + 7 - self.beam1Xshift],
            hairline_positions=self.hairlines[1] - self.beamEdges[1, 0],
            neighborhood=12,
            hairline_width=self.hairlineWidth / 2
        )

        self.combinedGainTable = np.ones(self.solarFlat.shape)
        self.combinedCoarseGainTable = np.ones(self.solarFlat.shape)

        self.combinedGainTable[
            self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]
        ] = beam0GainTable
        self.combinedGainTable[
            self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]
        ] = beam1GainTable

        self.combinedCoarseGainTable[
            self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]
        ] = beam0CoarseGainTable
        self.combinedCoarseGainTable[
            self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]
        ] = beam1CoarseGainTable

        return


    def spinor_plot_gaintables(self, index: int) -> None:
        """
        Helper method to plot gaintables in case the user wants to
            A.) Deal with just... so many popups
            B.) Check on the quality of the corrections as they go
        """
        aspect_ratio = self.solarFlat.shape[1] / self.solarFlat.shape[0]
        gainFig = plt.figure("SPINOR Gain Tables", figsize=(4*2.5, 2.5/aspect_ratio))
        ax_lamp = gainFig.add_subplot(141)
        ax_flat = gainFig.add_subplot(142)
        ax_coarse = gainFig.add_subplot(143)
        ax_fine = gainFig.add_subplot(144)
        ax_lamp.imshow(
            self.lampGain, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_flat.imshow(
            (self.solarFlat - self.solarDark)/self.lampGain, origin='lower', cmap='gray'
        )
        ax_coarse.imshow(
            self.combinedCoarseGainTable, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_fine.imshow(
            self.combinedGainTable, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_lamp.set_title("LAMP GAIN")
        ax_flat.set_title("SOLAR FLAT")
        ax_coarse.set_title("COARSE GAIN")
        ax_fine.set_title("FINE GAIN")
        for beam in self.beamEdges.flatten():
            ax_lamp.axhline(beam, c="C1", linewidth=1)
            ax_flat.axhline(beam, c="C1", linewidth=1)
            ax_coarse.axhline(beam, c="C1", linewidth=1)
            ax_fine.axhline(beam, c="C1", linewidth=1)
        for hair in self.hairlines.flatten():
            ax_lamp.axhline(hair, c="C2", linewidth=1)
            ax_flat.axhline(hair, c="C2", linewidth=1)
            ax_coarse.axhline(hair, c="C2", linewidth=1)
            ax_fine.axhline(hair, c="C2", linewidth=1)
        for edge in self.slitEdges:
            ax_lamp.axvline(edge, c="C1", linewidth=1)
            ax_flat.axvline(edge, c="C1", linewidth=1)
            ax_coarse.axvline(edge, c="C1", linewidth=1)
            ax_fine.axvline(edge, c="C1", linewidth=1)
        gainFig.tight_layout()
        if self.saveFigs:
            filename = os.path.join(self.finalDir, "gain_tables_{0}.png".format(index))
            gainFig.savefig(filename, bbox_inches="tight")

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
        if os.path.exists(self.solarGainReduced[index]):
            if self.verbose:
                print("File exists: {0}\nSkipping file write.".format(self.solarGainReduced[index]))
            return

        phdu = fits.PrimaryHDU()
        phdu.header["DATE"] = np.datetime64("now").astype(str)
        phdu.header["LC1"] = self.spinorLineCores[0]
        phdu.header["LC2"] = self.spinorLineCores[1]
        phdu.header["FTSLC1"] = self.ftsLineCores[0]
        phdu.header["FTSLC2"] = self.ftsLineCores[1]
        phdu.header["SPFLIP"] = self.flipWaveIdx
        phdu.header["HSG_SLW"] = self.slit_width
        phdu.header["HSG_GRAT"] = self.grating_angle

        flat = fits.ImageHDU(self.solarFlat)
        flat.header["EXTNAME"] = "SOLAR-FLAT"
        dark = fits.ImageHDU(self.solarDark)
        dark.header["EXTNAME"] = "SOLAR-DARK"
        cgain = fits.ImageHDU(self.combinedCoarseGainTable)
        cgain.header["EXTNAME"] = "COARSE-GAIN"
        fgain = fits.ImageHDU(self.combinedGainTable)
        fgain.header["EXTNAME"] = "GAIN"
        bedge = fits.ImageHDU(self.beamEdges)
        bedge.header["EXTNAME"] = "BEAM-EDGES"
        hairs = fits.ImageHDU(self.hairlines)
        hairs.header["EXTNAME"] = "HAIRLINES"
        slits = fits.ImageHDU(self.slitEdges)
        slits.header["EXTNAME"] = "SLIT-EDGES"
        shifts = fits.ImageHDU(np.array([self.beam1Yshift, self.beam1Xshift]))
        shifts.header["EXTNAME"] = "BEAM1-SHIFTS"

        hdul = fits.HDUList([phdu, flat, dark, cgain, fgain, bedge, hairs, slits, shifts])
        hdul.writeto(self.solarGainReduced[index], overwrite=True)

        return


    def spinor_fts_line_select(
            self, gratingParams: np.rec.recarray, averageProfile: np.ndarray
    ) -> tuple[list, list, bool]:
        """
        Pops up the line selection widget for gain table creation and wavelength determination
        Parameters
        ----------
        averageProfile
        gratingParams : numpy.records.recarray
            From spectraTools.grating_calculations

        Returns
        -------

        """
        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        apxWavemin = self.centralWavelength - np.nanmean(self.slitEdges) * gratingParams['Spectral_Pixel'] / 1000
        apxWavemax = self.centralWavelength + np.nanmean(self.slitEdges) * gratingParams['Spectral_Pixel'] / 1000
        apxWavemin -= 30 * gratingParams['Spectral_Pixel'] / 1000
        apxWavemax += 30 * gratingParams['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apxWavemin, apxWavemax)

        print("Top: SPINOR Spectrum (uncorrected). Bottom: FTS Reference Spectrum")
        print("Select the same two spectral lines on each plot.")
        spinorLines, ftsLines = spex.select_lines_doublepanel(
            averageProfile,
            fts_spec,
            4
        )
        spinorLineCores = [
            int(spex.find_line_core(averageProfile[x - 5:x + 5]) + x - 5) for x in spinorLines
        ]
        ftsLineCores = [
            spex.find_line_core(fts_spec[x - 20:x + 9]) + x - 20 for x in ftsLines
        ]

        spinorPixPerFTSPix = np.abs(np.diff(spinorLineCores)) / np.abs(np.diff(ftsLineCores))

        flipWave = self.determine_spectrum_flip(
            fts_spec, averageProfile, spinorPixPerFTSPix,
            spinorLineCores, ftsLineCores
        )

        return spinorLineCores, ftsLineCores, flipWave


    def spinor_get_polcal(self) -> None:
        """
        Loads or creates SPINOR polcal

        Returns
        -------

        """
        if os.path.exists(self.txMatrixReduced):
            with fits.open(self.txMatrixReduced) as hdul:
                self.inputStokes = hdul["STOKES-IN"].data
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
        if os.path.exists(self.txMatrixReduced):
            if self.verbose:
                print("File exists: {0}\nSkipping file write.".format(self.txMatrixReduced))
            return

        phdu = fits.PrimaryHDU()
        phdu.header["DATE"] = np.datetime64("now").astype(str)

        inStoke = fits.ImageHDU(self.inputStokes)
        inStoke.header["EXTNAME"] = "STOKES-IN"
        outStoke = fits.ImageHDU(self.calcurves)
        outStoke.header["EXTNAME"] = "CALCURVES"

        txmat = fits.ImageHDU(self.txmat)
        txmat.header["EXTNAME"] = "TXMAT"
        txchi = fits.ImageHDU(self.txchi)
        txchi.header["EXTNAME"] = "TXCHI"
        tx00 = fits.ImageHDU(self.txmat00)
        tx00.header["EXTNAME"] = "TX00"
        txinv = fits.ImageHDU(self.txmatinv)
        txinv.header["EXTNAME"] = "TXMATINV"

        hdul = fits.HDUList([phdu, inStoke, outStoke, txmat, txchi, tx00, txinv])
        hdul.writeto(self.txMatrixReduced, overwrite=True)

        return

    def spinor_polcal(self) -> None:
        """
        Performs polarization calibration on SPINOR data.

        Returns
        -------

        """

        # Get obsdate of science files to determine whether gain correction can be safely applied
        if os.path.exists(self.scienceFileList[0]):
            with fits.open(self.scienceFileList[0]) as hdul:
                baseObsdate = np.datetime64(hdul[1].header["DATE-OBS"], "D")
        else:
            baseObsdate = None

        polfile = fits.open(self.polcalFile)
        polfileObsdate = np.datetime64(polfile[1].header["DATE-OBS"], "D")

        polcalDarkCurrent = self.spinor_average_dark_from_hdul(polfile)

        fieldStops = [i.header['PT4_FS'] for i in polfile if "PT4_FS" in i.header.keys()]

        openFieldStops = [i for i in fieldStops if "DARK" not in i]

        # Grab the ICU parameters for every non-dark frame
        polarizerStaged = np.array([
            1 if "IN" in i.header['PT4_PSTG'] else 0 for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        retarderStaged = np.array([
            1 if "IN" in i.header['PT4_RSTG'] else 0 for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        polarizerAngle = np.array([
            i.header['PT4_POL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        retarderAngle = np.array([
            i.header['PT4_RET'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        llvlPol = np.array([
            i.header['DST_LLVL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        azPol = np.array([
            i.header['DST_AZ'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        elPol = np.array([
            i.header['DST_EL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])
        taPol = np.array([
            i.header['DST_TBL'] for i in polfile[1:] if "DARK" not in i.header['PT4_FS']
        ])

        polcalStokesBeams = np.zeros(
            (len(openFieldStops),
             2,
             4,
             self.beamEdges[0, 1] - self.beamEdges[0, 0],
             self.slitEdges[1] - self.slitEdges[0])
        )
        shift = self.beam1Yshift if self.beam1Yshift else 0
        xshift = self.beam1Xshift
        ctr = 0
        for hdu in polfile[1:]:
            if "DARK" not in hdu.header['PT4_FS']:
                # If the polcal was completed on the same date as the gain tables,
                # we can clean the polcals up with the gain to get a better estimate across the slit
                # If the polcals are from a different date, we should be content with dark-subtraction
                if (baseObsdate == polfileObsdate) & self.polcalProcessing:
                    data = self.demodulate_spinor(
                        (hdu.data - polcalDarkCurrent)/self.lampGain/self.combinedGainTable
                    )
                else:
                    data = self.demodulate_spinor(
                        (hdu.data - polcalDarkCurrent)
                    )

                # Cut out beams, flip n shift the upper beam
                polcalStokesBeams[ctr, 0, :, :, :] = data[
                    :, self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]
                ]
                polcalStokesBeams[ctr, 1, :, :, :] = np.flip(
                    scind.shift(
                        data[
                            :, self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]
                        ], (0, shift, -xshift)
                    ),
                    axis=1
                )
                ctr += 1

        # Close the polcal file
        polfile.close()

        merged_beams = np.zeros((
            polcalStokesBeams.shape[0],
            polcalStokesBeams.shape[2],
            polcalStokesBeams.shape[3],
            polcalStokesBeams.shape[4]
        ))
        merged_beams[:, 0, :, :] = np.nanmean(polcalStokesBeams[:, :, 0, :, :], axis=1)
        merged_beams[:, 1:, :, :] = (polcalStokesBeams[:, 0, 1:, :, :] - polcalStokesBeams[:, 1, 1:, :, :])/2.
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
                polcalStokesBeams.shape[0],
                polcalStokesBeams.shape[2],
                self.nSubSlits
            )
        )
        subarrays = np.array_split(merged_beams, self.nSubSlits, axis=2)
        submasks = np.array_split(merged_beam_mask, self.nSubSlits, axis=2)

        for i in range(self.nSubSlits):
            # Replace values outside ilimit with nans
            # That way, nanmean gets rid of them while preserving the array shape
            masked_subarray = subarrays[i]
            masked_subarray[~submasks[i]] = np.nan
            # Normalize QUV curve by I
            self.calcurves[:, 0, i] = np.nanmean(masked_subarray[:, 0, :, :], axis=(-2,-1))
            for j in range(1, 4):
                self.calcurves[:, j, i] = np.nanmean(
                    masked_subarray[:, j, :, :]/masked_subarray[:, 0, :, :], axis=(-2,-1)
                )

        # SPINOR IDL polcal takes measurements with LP+RET in beam,
        # Normalizes I to linear fit across. So I ranges ~0.75 -- 1.25
        # We'll modify this to use the full cal-set, but normalize with each set.
        # So norm of clear frames, sep norm for ret only, sep norm for lp+ret
        # Better method would be to normalize by clear frame I
        # Use the *whole* cow, y'know?
        # To do this, we'll need the 17th I measurements.
        # Or just grab the frames with PSTG and RSTG clear. Right.

        for i in range(self.nSubSlits):
            clearISelection = ~polarizerStaged.astype(bool) & ~retarderStaged.astype(bool)
            clearI = self.calcurves[:, 0, i][clearISelection]
            # Need an x-range that lines up with cal meas.
            clearIXrun = np.arange(self.calcurves.shape[0])[clearISelection]
            polyfitClear = np.polyfit(clearIXrun, clearI, 1)
            self.calcurves[:, 0, i][
                clearISelection
            ] = self.calcurves[:, 0, i][clearISelection] / (
                    np.arange(self.calcurves.shape[0]) * polyfitClear[0] + polyfitClear[1]
            )[clearISelection]

            retOnlySelection = ~polarizerStaged.astype(bool) & retarderStaged.astype(bool)
            retOnly = self.calcurves[:, 0, i][retOnlySelection]
            retOnlyXrun = np.arange(self.calcurves.shape[0])[retOnlySelection]
            polyfitRet = np.polyfit(retOnlyXrun, retOnly, 1)
            self.calcurves[:, 0, i][retOnlySelection] = self.calcurves[:, 0, i][retOnlySelection] / (
                np.arange(self.calcurves.shape[0]) * polyfitRet[0] + polyfitRet[1]
            )[retOnlySelection]

            lpretSelection = polarizerStaged.astype(bool) & retarderStaged.astype(bool)
            lpret = self.calcurves[:, 0, i][lpretSelection]
            lpretXrun = np.arange(self.calcurves.shape[0])[lpretSelection]
            polyfitLpRet = np.polyfit(lpretXrun, lpret, 1)
            self.calcurves[:, 0, i][lpretSelection] = self.calcurves[:, 0, i][lpretSelection] / (
                    np.arange(self.calcurves.shape[0]) * polyfitLpRet[0] + polyfitLpRet[1]
            )[lpretSelection]

            lpOnlySelection = polarizerStaged.astype(bool) & ~retarderStaged.astype(bool)
            lpOnly = self.calcurves[:, 0, i][lpOnlySelection]
            lpOnlyXrun = np.arange(self.calcurves.shape[0])[lpOnlySelection]
            polyfitLP = np.polyfit(lpOnlyXrun, lpOnly, 1)
            self.calcurves[:, 0, i][lpOnlySelection] = self.calcurves[:, 0, i][lpOnlySelection] / (
                    np.arange(self.calcurves.shape[0]) * polyfitLP[0] + polyfitLP[1]
            )[lpOnlySelection]

        self.calcurves = np.nan_to_num(self.calcurves)

        # Now create the input Stokes Vectors using the Telescope Matrix, plus cal unit parameters.
        # The cal train is Sky -> Telescope -> Polarizer -> Retarder -> Spectrograph
        inputStokes = np.zeros((self.calcurves.shape[0], 4))
        for i in range(self.calcurves.shape[0]):
            initStokes = np.array([1, 0, 0, 0])
            tmtx = self.get_telescope_matrix([azPol[i], elPol[i], taPol[i]])
            initStokes = tmtx @ initStokes
            if bool(polarizerStaged[i]):
                # Mult by 2 since we normalized our intensities earlier...
                initStokes = 2*spex.linearAnalyzerPolarizer(
                    polarizerAngle[i] * np.pi/180,
                    px=1,
                    py=0.005 # Estimate...
                ) @ initStokes
            if bool(retarderStaged[i]):
                initStokes = spex.linearRetarder(
                    retarderAngle[i] * np.pi/180,
                    self.calRetardance * np.pi/180
                ) @ initStokes
            inputStokes[i, :] = initStokes

        self.inputStokes = np.nan_to_num(inputStokes)
        self.txmat = np.zeros((self.nSubSlits, 4, 4))
        self.txchi = np.zeros(self.nSubSlits)
        self.txmat00 = np.zeros(self.nSubSlits)
        self.txmatinv = np.zeros((self.nSubSlits, 4, 4))

        for i in range(self.nSubSlits):
            errors, xmat = spex.matrix_inversion(
                inputStokes,
                self.calcurves[:, :, i]
            )
            self.txmat00[i] = xmat[0, 0]
            xmat /= xmat[0, 0]
            self.txmat[i] = xmat
            self.txmatinv[i] = np.linalg.inv(xmat)
            efficiencies = np.sqrt(np.sum(xmat**2, axis=1))
        #
        #     # Measurement of retardance from +-QU measurements
        #     # +Q
            lpPosQSelection = (
                (polarizerStaged.astype(bool) &
                 ~retarderStaged.astype(bool)) &
                ((np.abs(polarizerAngle) < 1) |
                 (np.abs(polarizerAngle - 180) < 1))
            )
            posQVec = self.calcurves[:, :, i][np.repeat(lpPosQSelection[:, np.newaxis], 4, axis=1)]
            posQVec = np.nanmean(
                posQVec.reshape(int(posQVec.shape[0]/4), 4),
                axis=0
            )
            stokesPosQ = self.txmatinv[i] @ posQVec
            stokesPosQ = stokesPosQ / stokesPosQ[0]
            posQsqdiff = np.sum((stokesPosQ - np.array([1, 1, 0, 0]))**2)
        #     # -Q
            lpNegQSelection = (
                (polarizerStaged.astype(bool) &
                 ~retarderStaged.astype(bool)) &
                ((np.abs(polarizerAngle - 90) < 1) |
                 (np.abs(polarizerAngle - 270) < 1))
            )
            negQVec = self.calcurves[:, :, i][np.repeat(lpNegQSelection[:, np.newaxis], 4, axis=1)]
            negQVec = np.nanmean(
                negQVec.reshape(int(negQVec.shape[0] / 4), 4),
                axis=0
            )
            stokesNegQ = self.txmatinv[i] @ negQVec
            stokesNegQ = stokesNegQ / stokesNegQ[0]
            negQsqdiff = np.sum((stokesNegQ - np.array([1, -1, 0, 0])) ** 2)
        #     # +U
            lpPosUSelection = (
                (polarizerStaged.astype(bool) &
                 ~retarderStaged.astype(bool)) &
                ((np.abs(polarizerAngle - 45) < 1) |
                 (np.abs(polarizerAngle - 225) < 1))
            )
            posUVec = self.calcurves[:, :, i][np.repeat(lpPosUSelection[:, np.newaxis], 4, axis=1)]
            posUVec = np.nanmean(
                posUVec.reshape(int(posUVec.shape[0] / 4), 4),
                axis=0
            )
            stokesPosU = self.txmatinv[i] @ posUVec
            stokesPosU = stokesPosU / stokesPosU[0]
            posUsqdiff = np.sum((stokesPosU - np.array([1, 0, 1, 0])) ** 2)
        #     # -U
            lpNegUSelection = (
                (polarizerStaged.astype(bool) &
                 ~retarderStaged.astype(bool)) &
                ((np.abs(polarizerAngle - 135) < 1) |
                 (np.abs(polarizerAngle - 315) < 1))
            )
            negUVec = self.calcurves[:, :, i][np.repeat(lpNegUSelection[:, np.newaxis], 4, axis=1)]
            negUVec = np.nanmean(
                negUVec.reshape(int(negUVec.shape[0] / 4), 4),
                axis=0
            )
            stokesNegU = self.txmatinv[i] @ negUVec
            stokesNegU = stokesNegU / stokesNegU[0]
            negUsqdiff = np.sum((stokesNegU - np.array([1, 0, -1, 0])) ** 2)

            self.txchi[i] = posQsqdiff + negQsqdiff + posUsqdiff + negUsqdiff

            if self.verbose:
                print("TX Matrix:")
                print(xmat)
                print("Inverse:")
                print(self.txmatinv[i])
                print("Efficiencies:")
                print(
                    "Q: "+str(round(efficiencies[1], 4)),
                    "U: "+str(round(efficiencies[2], 4)),
                    "V: "+str(round(efficiencies[3], 4))
                )
                print("Average Deviation of cal vectors: ", np.sqrt(self.txchi[i])/4)
                print("===========================\n\n")

            # Check physicality & Efficiencies:
            if np.nanmax(efficiencies[1:]) > 0.866:
                name = ['Q ','U ','V ']
                warnings.warn(
                    str(name[efficiencies[1:].argmax()]) +
                    "is likely too high with a value of " +
                    str(efficiencies[1:].max())
                )
            muellerCheck = spex.check_mueller_physicality(xmat)
            if not muellerCheck[0]:
                print(
                    ("WARNING: TX Matrix for section {0} of {1}\nIs an unphysical " +
                    "Mueller matrix with output minimum I:\n{2},\n" +
                    "and output minimum I^2 - (Q^2 + U^2 + V^2):\n{3}").format(
                        i, self.nSubSlits, muellerCheck[1], muellerCheck[2]
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
        polcalFig = plt.figure("Polcal Results", figsize=(8, 4))
        # Create 3 columns, 4 rows.
        # Column 1: Calcurves IQUV
        # Column 2: Input Stokes Vectors IQUV
        # Column 3: Output Stokes Vectors IQUV
        gs = polcalFig.add_gridspec(ncols=3, nrows=4)

        outStokes = np.array([self.txmat @ self.inputStokes[j, :] for j in range(self.inputStokes.shape[0])])
        names = ['I', 'Q', 'U', 'V']
        for i in range(4):
            ax_ccurve = polcalFig.add_subplot(gs[i, 0])
            ax_incurve = polcalFig.add_subplot(gs[i, 1])
            ax_outcurve = polcalFig.add_subplot(gs[i, 2])
            # Column Titles
            if i == 0:
                ax_ccurve.set_title("POLCAL CURVES")
                ax_incurve.set_title("INPUT VECTORS")
                ax_outcurve.set_title("FIT VECTORS")
            # Plot statements. Default is fine.
            for j in range(self.nSubSlits):
                ax_ccurve.plot(self.calcurves[:, i, j])
                ax_outcurve.plot(outStokes[:, j, i])
            ax_incurve.plot(self.inputStokes[:, i])
            # Clip to x range of [0, end]
            ax_ccurve.set_xlim(0, self.calcurves.shape[0])
            ax_incurve.set_xlim(0, self.calcurves.shape[0])
            ax_outcurve.set_xlim(0, self.calcurves.shape[0])
            # Clip to common y range defined by max/min of all 3 columns
            ymax = np.array(
                [self.calcurves[:, i, :].max(), outStokes[:, :, i].max(), self.inputStokes[:, i].max()]
            ).max()
            ymin = np.array(
                [self.calcurves[:, i, :].min(), outStokes[:, :, i].min(), self.inputStokes[:, i].min()]
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
        polcalFig.tight_layout()
        if self.saveFigs:
            filename = os.path.join(self.finalDir, "polcal_curves.png")
            polcalFig.savefig(filename, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)

        return


    def reduce_spinor_maps(self, index: str) -> None:
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
        for i in range(len(self.scienceFiles)):
            with fits.open(self.scienceFiles[i]) as hdul:
                total_slit_positions += len(hdul) - 1
        if self.verbose:
            print("{0} Slit Positions Observed in Sequence".format(total_slit_positions))
        # Check for existence of output file:
        with fits.open(self.scienceFiles[0]) as hdul:
            date, time = hdul[1].header['DATE-OBS'].split("T")
            date = date.replace("-", "")
            time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
            outname = self.reducedFilePattern.format(
                date,
                time,
                total_slit_positions
            )
            outfile = os.path.join(self.finalDir, outname)
            if os.path.exists(outfile):
                remakeFile = input("File: {0}\nExists. (R)emake or (C)ontinue?  ".format(outname))
                if ("c" in remakeFile.lower()) or (remakeFile.lower() == ""):
                    plt.pause(2)
                    plt.close("all")
                    return
                elif ("r" in remakeFile.lower()) and self.verbose:
                    print("Remaking file with user-specified corrections. This may take some time.")

        reducedData = np.zeros((
            total_slit_positions,
            4,
            self.beamEdges[0, 1] - self.beamEdges[0, 0],
            self.slitEdges[1] - self.slitEdges[0]
        ))
        completeI2QUVCrosstalk = np.zeros((
            total_slit_positions,
            3, 2, self.beamEdges[0, 1] - self.beamEdges[0, 0]
        ))
        completeInternalCrosstalks = np.zeros((
            total_slit_positions,
            3,
            self.beamEdges[0, 1] - self.beamEdges[0, 0]
        ))
        shift = self.beam1Yshift if self.beam1Yshift else 0

        # Interpolate telescope inverse matrix to entire slit from nsubslits
        xinvInterp = scinterp.CubicSpline(
            np.linspace(0, self.beamEdges[0, 1]-self.beamEdges[0, 0], self.nSubSlits),
            self.txmatinv,
            axis=0,
        )(np.arange(0, self.beamEdges[0, 1]-self.beamEdges[0, 0]))

        stepIndex = 0
        # Setting these up for later
        masterHairlineCenters = (0, 0)
        masterSpectralLineCenters = (0, 0)
        with tqdm.tqdm(
            total=total_slit_positions,
            desc="Reducing Science Map"
        ) as pbar:
            for file in self.scienceFiles:
                science_hdu = fits.open(file)
                for i in range(1, len(science_hdu)):
                    iquv = self.demodulate_spinor(
                        (science_hdu[i].data - self.solarDark)/self.lampGain/self.combinedGainTable
                    )
                    scienceBeams = np.zeros(
                        (2, 4, self.beamEdges[0, 1] - self.beamEdges[0, 0], self.slitEdges[1] - self.slitEdges[0])
                    )
                    scienceBeams[0, :, :, :] = iquv[
                        :,
                        self.beamEdges[0, 0]:self.beamEdges[0, 1],
                        self.slitEdges[0]:self.slitEdges[1]
                    ]
                    scienceBeams[1, :, :, :] = np.flip(
                        scind.shift(
                            iquv[
                                :,
                                self.beamEdges[1, 0]:self.beamEdges[1, 1],
                                self.slitEdges[0]:self.slitEdges[1]
                            ], (0, shift, self.beam1Xshift)
                        ), axis=1
                    )

                    # Reference beam for hairline/spectral line deskew shouldn't have full gain
                    # correction done, due to hairline residuals. It *should* be safe to use the
                    # lamp gain, as the hairlines in that should've been cleaned up.
                    alignmentBeam = (np.mean(science_hdu[i].data, axis=0) - self.solarDark)/self.lampGain
                    hairlineSkews, hairlineCenters = self.subpixel_hairline_align(
                        alignmentBeam
                    )
                    # Perform hairline deskew
                    for beam in range(scienceBeams.shape[0]):
                        for hairProf in range(scienceBeams.shape[3]):
                            scienceBeams[beam, :, :, hairProf] = scind.shift(
                                scienceBeams[beam, :, :, hairProf],
                                (0, -hairlineSkews[beam, hairProf]),
                                mode='nearest'
                            )
                    # Perform bulk hairline alignment on deskewed beams
                    scienceBeams[1] = scind.shift(
                        scienceBeams[1], (0, np.diff(hairlineCenters)[0], 0),
                        mode='nearest'
                    )

                    # Perform spectral deskew
                    scienceBeams, spectralCenters = self.subpixel_spectral_align(scienceBeams, hairlineCenters)

                    # Perform bulk spectral alignment on deskewed beams
                    scienceBeams[1] = scind.shift(
                        scienceBeams[1], (0, 0, np.diff(spectralCenters)[0]),
                        mode='nearest'
                    )

                    # Common positions to register observation to.
                    if stepIndex == 0:
                        masterHairlineCenters = hairlineCenters
                        masterSpectralLineCenters = spectralCenters
                    # Perform master registration to 0th slit image.
                    scienceBeams = scind.shift(
                        scienceBeams, (
                            0, 0,
                            -(hairlineCenters[0] - masterHairlineCenters[0]),
                            (spectralCenters[0] - masterSpectralLineCenters[0])
                        ),
                        mode='nearest'
                    )

                    combined_beams = np.zeros(scienceBeams.shape[1:])
                    combined_beams[0] = np.nanmean(scienceBeams[:, 0, :, :], axis=0)
                    combined_beams[1:] = (
                        scienceBeams[0, 1:, :, :] - scienceBeams[1, 1:, :, :]
                    )/2
                    tmtx = self.get_telescope_matrix(
                        [science_hdu[i].header['DST_AZ'],
                         science_hdu[i].header['DST_EL'],
                         science_hdu[i].header['DST_TBL']]
                    )
                    inv_tmtx = np.linalg.inv(tmtx)
                    for j in range(combined_beams.shape[1]):
                        combined_beams[:, j, :] = inv_tmtx @ xinvInterp[j, :, :] @ combined_beams[:, j, :]

                    # Get parallactic angle for QU rotation correction
                    angular_geometry = self.spherical_coordinate_transform(
                        [science_hdu[i].header['DST_AZ'], science_hdu[i].header['DST_EL']]
                    )
                    # Sub off P0 angle
                    rotation = np.pi + angular_geometry[2] - science_hdu[i].header['DST_PEE'] * np.pi/180
                    crot = np.cos(-2*rotation)
                    srot = np.sin(-2*rotation)

                    # Make a copy, as the Q/U components are transformed from the originals.
                    qtmp = combined_beams[1, :, :].copy()
                    utmp = combined_beams[2, :, :].copy()
                    combined_beams[1, :, :] = crot*qtmp + srot*utmp
                    combined_beams[2, :, :] = -srot*qtmp + crot*utmp

                    combined_beams, i2quvCrosstalk, internal_crosstalks = self.solve_spinor_crosstalks(
                        combined_beams
                    )

                    # Reverse the wavelength axis if required.
                    combined_beams = combined_beams[:, :, ::self.flipWaveIdx]

                    reducedData[stepIndex] = combined_beams
                    completeI2QUVCrosstalk[stepIndex] = i2quvCrosstalk
                    completeInternalCrosstalks[stepIndex] = internal_crosstalks

                    # Choose lines for analysis. Use same method of choice as hsgCal, where user sets
                    # approx. min/max, the code changes the bounds, and
                    if stepIndex == 0:
                        mean_profile = np.nanmean(combined_beams[0], axis=0)
                        approxWavelengthArray = self.tweak_wavelength_calibration(mean_profile)
                        print("Select spectral ranges (xmin, xmax) for overview maps. Close window when done.")
                        # Approximate indices of line cores
                        coarseIndices = spex.select_spans_singlepanel(
                            mean_profile, xarr=approxWavelengthArray, figName="Select Lines for Analysis"
                        )
                        # Location of minimum in the range
                        minIndices = [
                            spex.find_nearest(
                                mean_profile[coarseIndices[x][0]:coarseIndices[x][1]],
                                mean_profile[coarseIndices[x][0]:coarseIndices[x][1]].min()
                            ) + coarseIndices[x][0] for x in range(coarseIndices.shape[0])
                        ]
                        # Location of exact line core
                        lineCores = [
                            spex.find_line_core(mean_profile[x-5:x+7]) + x - 5 for x in minIndices
                        ]
                        # Find start and end indices that put line cores at the center of the window.
                        mapIndices = np.zeros(coarseIndices.shape)
                        for j in range(coarseIndices.shape[0]):
                            averageDelta = np.mean(np.abs(coarseIndices[j, :] - lineCores[j]))
                            mapIndices[j, 0] = int(round(lineCores[j] - averageDelta, 0))
                            mapIndices[j, 1] = int(round(lineCores[j] + averageDelta, 0) + 1)

                    if self.plot:
                        plt.ion()
                        # Set up overview maps to blit new data into.
                        # Need maps for the slit images (IQUV) that are replaced at each step,
                        # As well as IQUV maps of the full field for each line selected.
                        # These latter will be filled as the map is processed.
                        if stepIndex == 0:
                            fieldImages = np.zeros((
                                len(lineCores),  # Number of lines
                                4,  # Stokes-IQUV values
                                combined_beams.shape[1],
                                total_slit_positions
                            ))
                        for j in range(len(lineCores)):
                            fieldImages[j, 0, :, stepIndex] = combined_beams[0, :, int(round(lineCores[j], 0))]
                            for k in range(1, 4):
                                fieldImages[j, k, :, stepIndex] = scinteg.trapezoid(
                                    np.nan_to_num(
                                        combined_beams[k, :, int(mapIndices[j, 0]):int(mapIndices[j, 1])]/
                                        combined_beams[0, :, int(mapIndices[j, 0]):int(mapIndices[j, 1])]
                                    ),
                                    axis=-1
                                )
                        if stepIndex == 0:
                            slit_plate_scale = self.dstPlateScale * self.dstCollimator / self.slitCameraLens
                            camera_dy = slit_plate_scale * (self.spectrographCollimator / self.cameraLens) * (
                                        self.pixel_size / 1000)
                            map_dx = science_hdu[1].header['HSG_STEP']


                            plot_params = self.set_up_live_plot(
                                fieldImages, combined_beams, internal_crosstalks, camera_dy, map_dx
                            )
                        self.update_live_plot(
                            *plot_params, fieldImages, combined_beams, internal_crosstalks, stepIndex
                        )
                    stepIndex += 1
                    pbar.update(1)
                science_hdu.close()

        # Save final plots if applicable
        if self.plot & self.saveFigs:
            for fig in range(len(plot_params[0])):
                filename = os.path.join(self.finalDir, "field_image_{0}.png".format(fig))
                plot_params[0][fig].savefig(filename, bbox_inches="tight")

        mean_profile = np.nanmean(reducedData[:, 0, :, :], axis=(0, 1))
        approxWavelengthArray = self.tweak_wavelength_calibration(mean_profile)
        # Swap axes to make X/Y contigent with data X/Y
        reducedData = np.swapaxes(reducedData, 0, 2)

        reduced_filename = self.package_scan(reducedData, approxWavelengthArray)
        crosstalk_filename = self.package_crosstalks(completeI2QUVCrosstalk, completeInternalCrosstalks, index)
        parameter_maps, referenceWavelengths, tweakedIndices, meanProfile, wavelengthArray = self.spinor_analysis(
            reducedData, mapIndices
        )
        param_filename = self.package_analysis(
            parameter_maps,
            referenceWavelengths,
            tweakedIndices,
            meanProfile,
            wavelengthArray,
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


    def subpixel_hairline_align(self, slitImage: np.ndarray) -> tuple[np.ndarray, tuple]:
        """
        Performs subpixel hairline alignment of two beams into a single image and returns the necessary translations.
        Returns shift map for hairline alignment.

        Parameters
        ----------
        slitImage : numpy.ndarray
            Minimally-processed image of the spectrum. Beams will be cut out and aligned from this. Shape (ny, nx)

        Returns
        -------
        hairlineSkews : numpy.ndarray
            Array of shape (2, nx) containing subpixel shifts for hairline registration
        hairlineCenter : tuple
            Centers of the registered hairline. Should assist in getting an evenly-registered image across all slit pos.
        """
        beam0 = slitImage[self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]]
        beam1 = np.flip(
            scind.shift(
                slitImage[self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]],
                (self.beam1Yshift, self.beam1Xshift)
            ), axis=0
        )
        dualBeams = np.stack([beam0, beam1], axis=0)
        deskewedDualBeams = dualBeams.copy()
        deskewHairline = self.hairlines[0, 0]
        hairlineMinimum = int(deskewHairline - 14)
        hairlineDelta = 28
        hairlineMaximum = int(deskewHairline + 14)
        if hairlineMinimum < 0:
            hairlineMinimum = 0
            hairlineDelta = int(2*deskewHairline)
            hairlineMaximum = hairlineDelta
        elif hairlineMaximum >= dualBeams.shape[1]:
            hairlineMaximum = int(dualBeams.shape[1] - 1)
            hairlineDelta = int(2 * (hairlineMaximum - deskewHairline))
            hairlineMinimum = hairlineMaximum - hairlineDelta

        medfiltHairlineImage = scind.median_filter(
            dualBeams[:, int(hairlineMinimum):int(hairlineMaximum), :],
            size=(1, 2, 25)
        )
        hairlineSkews = np.zeros((2, medfiltHairlineImage.shape[2]))
        for i in range(dualBeams.shape[0]):
            hairlineSkews[i, :] = spex.spectral_skew(
                np.rot90(medfiltHairlineImage[i, :, :]), order=1, slit_reference=0.5
            )
            for j in range(hairlineSkews.shape[1]):
                deskewedDualBeams[i, :, j] = scind.shift(
                    dualBeams[i, :, j], -hairlineSkews[i, j],
                    mode='nearest'
                )
        # Find bulk hairline center for full alignment
        hairlineCenter = (
            spex.find_line_core(
                np.nanmedian(deskewedDualBeams[0, hairlineMinimum:hairlineMaximum, :], axis=1)
            ) + hairlineMinimum,
            spex.find_line_core(
                np.nanmedian(deskewedDualBeams[1, hairlineMinimum:hairlineMaximum, :], axis=1)
            ) + hairlineMinimum
        )

        return hairlineSkews, hairlineCenter

    def subpixel_spectral_align(self, cutoutBeams: np.ndarray, hairlineCenter: tuple) -> tuple[np.ndarray, float]:
        """
        Performs iterative deskew and align along the spectral axis.
        Returns the aligned beam and spectral line center of the aligned beam.
        Parameters
        ----------
        cutoutBeams : numpy.ndarray
            Cut-out and vertically-aligned
        hairlineCenter : tuple
            Position of hairline in each beam for masking

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
            upperHairlineCenter = (
                hairlineCenter[0] + np.diff(self.hairlines, axis=1)[0],
                hairlineCenter[1] + np.diff(self.hairlines, axis=1)[1]
            )
        x1, x2 = 20, 21
        for spiter in range(5):
            order = 2  if spiter < 2 else 2
            for beam in range(2):
                spectralImage = cutoutBeams[
                    beam, 0, :, int(self.spinorLineCores[0] - x1):int(self.spinorLineCores[0] + x2)
                ].copy()
                # Deskew function is written to cope with NaNs.
                # It is NOT written to deal with a hairline.
                # Mask hairlines
                hairMin = int(hairlineCenter[beam] - 4)
                hairMax = int(hairlineCenter[beam] + 5)
                hairMin = 0 if hairMin < 0 else hairMin
                hairMax = int(spectralImage.shape[0] - 1) if hairMax > spectralImage.shape[0] - 1 else hairMax
                spectralImage[hairMin:hairMax, :] = np.nan
                # FLIR may not have another set of hairlines.
                if self.hairlines.shape[1] > 1:
                    hairMin = int(upperHairlineCenter[beam] - 4)
                    hairMax = int(upperHairlineCenter[beam] + 5)
                    hairMin = 0 if hairMin < 0 else hairMin
                    hairMax = int(spectralImage.shape[0] - 1) if hairMax > spectralImage.shape[0] - 1 else hairMax
                    spectralImage[hairMin:hairMax, :] = np.nan
                spectralSkews = spex.spectral_skew(
                    spectralImage, order=order, slit_reference=0.5
                )
                for prof in range(cutoutBeams.shape[2]):
                    cutoutBeams[beam, :, prof, :] = scind.shift(
                        cutoutBeams[beam, :, prof, :], (0, spectralSkews[prof]), mode='nearest'
                    )
            x1 -= 3
            x2 -= 3
        # Find bulk spectral line center for full alignment
        spectralCenter = (
            spex.find_line_core(
                np.nanmedian(
                    cutoutBeams[0, 0, :, int(self.spinorLineCores[0] - 10): int(self.spinorLineCores[0] + 10)],
                    axis=0
                )
            ) + int(self.spinorLineCores[0] - 10),
            spex.find_line_core(
                np.nanmedian(
                    cutoutBeams[1, 0, :, int(self.spinorLineCores[0] - 10): int(self.spinorLineCores[0] + 10)],
                    axis=0
                )
            ) + int(self.spinorLineCores[0] - 10),
        )
        return cutoutBeams, spectralCenter



    def solve_spinor_crosstalks(self, iquvCube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            iquvCube[3] += np.repeat(internalCrosstalk[2, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[2]
        2.) Undo V -> U
            iquvCube[2] += np.repeat(internalCrosstalk[1, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[3]
        3.) Undo V -> Q
            iquvCube[1] += np.repeat(internalCrosstalk[0, :, np.newaxis], iquvCube.shape[2], axis=-1) * iquvCube[3]
        4.) Undo I -> QUV
            iquvCube[1:] += (np.repeat(crosstalkI2QUV[:, 0, :, np.newaxis], iquvCube.shape[2], axis=-1) *
                    np.arange(iquvCube.shape[2] +
                    np.repeat(crosstalkI2QUV[:, 1, :, np.newaxis], iquvCube.shape[2], axis=-1)) *
                np.repeat(iquvCube[0][np.newaxis, :, :], 3, axis=0)

        Parameters
        ----------
        iquvCube : numpy.ndarray
            Mostly-corrected 3D array of a combined full-Stokes slit image. Should have shape (4, ny, nlambda)

        Returns
        -------
        iquvCube : numpy.ndarray
            Shape (4, ny, nlamba), crosstalk-isolated datacube.
        crosstalkI2QUV : numpy.ndarray
            Has shape (3, 2, ny), contains I->QUV crosstalk coefficients.
        internalCrosstalk : numpy.ndarray
            Has shape (3, ny), contains V->Q, V->U, U->V crosstalks
        """

        crosstalkI2QUV = np.zeros((3, 2, iquvCube.shape[1]))
        internalCrosstalk = np.zeros((3, iquvCube.shape[1]))

        if self.crosstalkContinuum is not None:
            # Shape 3xNY
            i2quv = np.mean(
                iquvCube[1:, :, self.crosstalkContinuum[0]:self.crosstalkContinuum[1]] /
                np.repeat(
                    iquvCube[0, :, self.crosstalkContinuum[0]:self.crosstalkContinuum[1]][np.newaxis, :, :],
                    3, axis=0
                ), axis=2
            )
            iquvCube[1:] = iquvCube[1:] - np.repeat(
                i2quv[:, :, np.newaxis], iquvCube.shape[2], axis=2
            ) * np.repeat(iquvCube[0][np.newaxis, :, :], 3, axis=0)
            crosstalkI2QUV[:, 1, :] = i2quv
        else:
            for j in range(iquvCube.shape[1]):
                # I->QUV crosstalk correction
                for k in range(1, 4):
                    iquvCube[k, j, :], crosstalkI2QUV[k - 1, :, j] = self.i2quv_crosstalk(
                        iquvCube[0, j, :],
                        iquvCube[k, j, :]
                    )
        # V->QU crosstalk correction
        if self.v2q:
            bulkV2QCrosstalk = self.internal_crosstalk_2d(
                iquvCube[1, :, :], iquvCube[3, :, :]
            )
            iquvCube[1, :, :] = iquvCube[1, :, :] - bulkV2QCrosstalk * iquvCube[3, :, :]
            for j in range(iquvCube.shape[1]):
                iquvCube[1, j, :], internalCrosstalk[0, j] = self.v2qu_crosstalk(
                    iquvCube[3, j, :],
                    iquvCube[1, j, :]
                )
            internalCrosstalk[0] += bulkV2QCrosstalk
        if self.v2u:
            bulkV2UCrosstalk = self.internal_crosstalk_2d(
                iquvCube[2, :, :], iquvCube[3, :, :]
            )
            iquvCube[2, :, :] = iquvCube[2, :, :] - bulkV2UCrosstalk * iquvCube[3, :, :]
            for j in range(iquvCube.shape[1]):
                iquvCube[2, j, :], internalCrosstalk[1, j] = self.v2qu_crosstalk(
                    iquvCube[3, j, :],
                    iquvCube[2, j, :]
                )
            internalCrosstalk[1] += bulkV2UCrosstalk
        if self.u2v:
            bulkU2VCrosstalk = self.internal_crosstalk_2d(
                iquvCube[3, :, :], iquvCube[2, :, :]
            )
            iquvCube[3, :, :] = iquvCube[3, :, :] - bulkU2VCrosstalk * iquvCube[2, :, :]
            for j in range(iquvCube.shape[1]):
                iquvCube[3, j, :], internalCrosstalk[2, j] = self.v2qu_crosstalk(
                    iquvCube[2, j, :],
                    iquvCube[3, j, :]
                )
            internalCrosstalk[2] += bulkU2VCrosstalk

        return iquvCube, crosstalkI2QUV, internalCrosstalk


    def package_crosstalks(self, i2quvCrosstalks: np.ndarray, internalCrosstalks: np.ndarray, index: int) -> str:
        """
        Places Stokes-vector crosstalk parameters in a file for interested end users.

        Parameters
        ----------
        i2quvCrosstalks : numpy.ndarray
            Array of shape (nx, 3, 2, ny) of I->QUV crosstalk parameters.
            The option exists to use a 1D fit for crosstalk, hence the 2-axis, where the 2 entries are "m" and "b"
            in y=mx+b
        internalCrosstalks : numpy.ndarray
            Array of shape (nx, 3, ny) of internal crosstalk. (:, 0, :) is V->Q, (:, 1, :) is V->U, (:, 2, :) is U->V
        index : int
            For formatting the output filename

        Returns
        -------
        crosstalkFile : str
            Name of file where crosstalk parameters are stored.

        """
        ext0 = fits.PrimaryHDU()
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File Creation Date and Time")
        ext0.header['ORIGIN'] = "NMSU/SSOC"
        if self.crosstalkContinuum is not None:
            ext0.header['I2QUV'] = ("CONST", "0-D I2QUV Crosstalk")
        else:
            ext0.header['I2QUV'] = ("1DFIT", "1-D I2QUV Crosstalk")
        ext0.header['V2Q'] = (int(self.v2q), "1 if V->Q Calculated, else 0")
        ext0.header['V2U'] = (int(self.v2u), "1 if V->U Calculated, else 0")
        ext0.header['U2V'] = (int(self.u2v), "1 if U->V Calculated, else 0")
        ext0.header['COMMENT'] = "Crosstalks applied in order:"
        ext0.header['COMMENT'] = "I->QUV"
        ext0.header['COMMENT'] = "V->Q"
        ext0.header['COMMENT'] = "V->U"
        ext0.header['COMMENT'] = "U->V"

        i2quvExt = fits.ImageHDU(i2quvCrosstalks)
        i2quvExt.header['EXTNAME'] = "I2QUV"
        i2quvExt.header[""] = "<QUV> = <QUV> - (coef[0]*[0, 1, ... nlambda] + coef[1]) * I"

        v2qExt = fits.ImageHDU(internalCrosstalks[:, 0, :])
        v2qExt.header['EXTNAME'] = "V2Q"
        v2qExt.header[""] = "Q = Q - coef*V"

        v2uExt = fits.ImageHDU(internalCrosstalks[:, 1, :])
        v2uExt.header['EXTNAME'] = "V2U"
        v2uExt.header[""] = "U = U - coef*V"

        u2vExt = fits.ImageHDU(internalCrosstalks[:, 2, :])
        u2vExt.header['EXTNAME'] = "U2V"
        u2vExt.header[""] = "V = V - coef*U"

        hdul = fits.HDUList([ext0, i2quvExt, v2qExt, v2uExt, u2vExt])
        filename = "{0}_MAP_{1}_CROSSTALKS.fits".format(self.camera, index)
        crosstalkFile = os.path.join(self.finalDir, filename)
        hdul.writeto(crosstalkFile, overwrite=True)
        return crosstalkFile


    def set_up_live_plot(
            self, fieldImages: np.ndarray, slitImages: np.ndarray, internalCrosstalks: np.ndarray,
            dy: float, dx:float
    ) -> tuple:
        """
        Initializes live plotting statements for monitoring progress of reductions

        Parameters
        ----------
        fieldImages : numpy.ndarray
            Array of dummy field images for line cores. Shape nlines, 4, ny, nx where:
                1.) nlines is the number of spectral lines selected by the user to keep an eye on
                2.) 4 is from the IQUV stokes parameters
                3.) ny is the length of the slit
                4.) nx is the number of slit positions in the scan
        slitImages : numpy.ndarray
            Array of IQUV slit images. Shape 4, ny, nlambda where:
                5.) nlambda is the wavelength axis
        internalCrosstalks : numpy.ndarray
            3 x ny array of V<->QU crosstalk values for monitoring
        dy : float
            Approximate resolution scale in y for sizing the map extent
        dx : float
            Approximate resolution scale in x for sizing the map extent

        Returns
        -------
        fieldFigList : list
            List of matplotlib figures, with an entry for each line of interest
        fieldI : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-I subplot
        fieldQ : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-Q subplot
        fieldU : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-U subplot
        fieldV : list
            List of matplotlib.image.AxesImage with an entry for each line of interest Stokes-V subplot
        slitFig : matplotlib.pyplot.figure
            Matplotlib figure containing the slit IQUV image. The entire image will be blitted each time
        slitI : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-I image
        slitQ : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-Q image
        slitU : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-U image
        slitV : matplotlib.image.AxesImage
            Matplotlib axes class containing the slit Stokes-V image
        """
        # Close all figures to reset plotting
        plt.close("all")

        # Required for live plotting
        plt.ion()
        plt.pause(0.005)

        slitAspectRatio = slitImages.shape[2] / slitImages.shape[1]

        # Set up the spectral data first, since it's only one window
        slitFig = plt.figure("Reduced Slit Images", figsize=(5, 5/slitAspectRatio))
        slitGS = slitFig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
        slitAxI = slitFig.add_subplot(slitGS[0, 0])
        slitI = slitAxI.imshow(slitImages[0], cmap='gray', origin='lower')
        slitAxI.text(10, 10, "I", color='C1')
        slitAxQ = slitFig.add_subplot(slitGS[0, 1])
        slitQ = slitAxQ.imshow(slitImages[1], cmap='gray', origin='lower')
        slitAxQ.text(10, 10, "Q", color='C1')
        slitAxU = slitFig.add_subplot(slitGS[1, 0])
        slitU = slitAxU.imshow(slitImages[2], cmap='gray', origin='lower')
        slitAxU.text(10, 10, "U", color='C1')
        slitAxV = slitFig.add_subplot(slitGS[1, 1])
        slitV = slitAxV.imshow(slitImages[3], cmap='gray', origin='lower')
        slitAxV.text(10, 10, "V", color='C1')

        # Now the multiple windows for the multiple lines of interest
        fieldAspectRatio = (dx * fieldImages.shape[3]) / (dy * fieldImages.shape[2])

        fieldFigList = []
        fieldGS = []
        fieldI = []
        fieldQ = []
        fieldU = []
        fieldV = []
        fieldIAx = []
        fieldQax = []
        fieldUAx = []
        fieldVAx = []
        for j in range(fieldImages.shape[0]):
            fieldFigList.append(
                plt.figure("Line " + str(j), figsize=(5, 5/fieldAspectRatio+1))
            )
            fieldGS.append(
                fieldFigList[j].add_gridspec(2, 2, hspace=0.1, wspace=0.1)
            )
            fieldIAx.append(
                fieldFigList[j].add_subplot(fieldGS[j][0, 0])
            )
            fieldI.append(
                fieldIAx[j].imshow(
                    fieldImages[j, 0], origin='lower', cmap='gray',
                    extent=[0, dx*fieldImages.shape[3], 0, dy*fieldImages.shape[2]]
                )
            )
            fieldQax.append(
                fieldFigList[j].add_subplot(fieldGS[j][0, 1])
            )
            fieldQ.append(
                fieldQax[j].imshow(
                    fieldImages[j, 1], origin='lower', cmap='gray',
                    extent=[0, dx*fieldImages.shape[3], 0, dy*fieldImages.shape[2]]
                )
            )
            fieldUAx.append(
                fieldFigList[j].add_subplot(fieldGS[j][1, 0])
            )
            fieldU.append(
                fieldUAx[j].imshow(
                    fieldImages[j, 2], origin='lower', cmap='gray',
                    extent=[0, dx*fieldImages.shape[3], 0, dy*fieldImages.shape[2]]
                )
            )
            fieldVAx.append(
                fieldFigList[j].add_subplot(fieldGS[j][1, 1])
            )
            fieldV.append(
                fieldVAx[j].imshow(
                    fieldImages[j, 2], origin='lower', cmap='gray',
                    extent=[0, dx*fieldImages.shape[3], 0, dy*fieldImages.shape[2]]
                )
            )

            # Beautification; Turn off some x/y tick labels, set titles, axes labels, etc...
            # Turn off tick labels for all except the first column in y, and the last row in x
            fieldIAx[j].set_xticklabels([])
            fieldIAx[j].set_ylabel("Extent [arcsec]")
            fieldIAx[j].set_title("Line Core  Stokes-I")

            fieldQax[j].set_yticklabels([])
            fieldQax[j].set_xticklabels([])
            fieldQax[j].set_title("Integrated Stokes-Q")

            fieldUAx[j].set_ylabel("Extent [arcsec]")
            fieldUAx[j].set_xlabel("Extent [arcsec]")
            fieldUAx[j].set_title("Integrated Stokes-U")

            fieldVAx[j].set_yticklabels([])
            fieldVAx[j].set_xlabel("Extent [arcsec]")
            fieldVAx[j].set_title("Integrated Stokes-V")

        if not any((self.v2q, self.v2q, self.v2u)):

            plt.show(block=False)
            plt.pause(0.05)

            return (fieldFigList, fieldI, fieldQ, fieldU, fieldV, slitFig, slitI, slitQ, slitU, slitV,
                    None, None, None, None)
        else:
            crosstalkFig = plt.figure("Internal Crosstalks Along Slit", figsize=(8, 5))
            v2qAx = crosstalkFig.add_subplot(131)
            v2uAx = crosstalkFig.add_subplot(132)
            u2vAx = crosstalkFig.add_subplot(133)
            v2q = v2qAx.plot(
                internalCrosstalks[0, :], np.arange(internalCrosstalks.shape[1]),
                color='C1'
            )
            v2qAx.set_xlim(-1.05, 1.05)
            v2qAx.set_ylim(0, internalCrosstalks.shape[1])
            v2qAx.set_title("V->Q Crosstalk")
            v2qAx.set_ylabel("Position Along Slit")

            v2u = v2uAx.plot(
                internalCrosstalks[1, :], np.arange(internalCrosstalks.shape[1]),
                color='C1'
            )
            v2uAx.set_xlim(-1.05, 1.05)
            v2uAx.set_ylim(0, internalCrosstalks.shape[1])
            v2uAx.set_title("V->U Crosstalk")
            v2uAx.set_xlabel("Crosstalk Value")

            u2v = u2vAx.plot(
                internalCrosstalks[2, :], np.arange(internalCrosstalks.shape[1]),
                color="C1"
            )
            u2vAx.set_xlim(-1.05, 1.05)
            u2vAx.set_ylim(0, internalCrosstalks.shape[1])
            u2vAx.set_title("U->V Crosstalk [residual]")

            plt.show(block=False)
            plt.pause(0.05)

            return (
                fieldFigList, fieldI, fieldQ, fieldU, fieldV,
                slitFig, slitI, slitQ, slitU, slitV,
                crosstalkFig, v2q, v2u, u2v)




    def update_live_plot(
            self,
            fieldFigList: list, fieldI: list, fieldQ: list, fieldU: list, fieldV: list,
            slitFig: matplotlib.pyplot.figure, slitI: matplotlib.image.AxesImage, slitQ: matplotlib.image.AxesImage,
            slitU: matplotlib.image.AxesImage, slitV: matplotlib.image.AxesImage,
            crosstalkFig: matplotlib.pyplot.figure, v2q: matplotlib.image.AxesImage,
            v2u: matplotlib.image.AxesImage, u2v: matplotlib.image.AxesImage,
            fieldImages: np.ndarray, slitImages: np.ndarray, internalCrosstalks: np.ndarray,
            step: int
    ) -> None:
        """
        Updates the plots created in self.set_up_live_plot.

        Parameters
        ----------
        fieldFigList : list
        fieldI : list
        fieldQ : list
        fieldU : list
        fieldV : list
        slitFig : matplotlib.pyplot.figure
        slitI : matplotlib.image.AxesImage
        slitQ : matplotlib.image.AxesImage
        slitU : matplotlib.image.AxesImage
        slitV : matplotlib.image.AxesImage
        crosstalkFig : matplotlib.pyplot.figure
        v2q : matplotlib.image.AxesImage
        v2u : matplotlib.image.AxesImage
        u2v : matplotlib.image.AxesImage
        fieldImages : numpy.ndarray
        slitImages : numpy.ndarray
        internalCrosstalks : numpy.ndarray
        step : int
            Step of the reduction process we're on for normalization purposes.

        Returns
        -------

        """

        slitI.set_array(slitImages[0])
        slitI.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slitImages[0]) - 3 * np.std(slitImages[0]),
                vmax=np.mean(slitImages[0]) + 3 * np.std(slitImages[0])
            )
        )
        slitQ.set_array(slitImages[1])
        slitQ.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slitImages[1]) - 3 * np.std(slitImages[1]),
                vmax=np.mean(slitImages[1]) + 3 * np.std(slitImages[1])
            )
        )
        slitU.set_array(slitImages[2])
        slitU.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slitImages[2]) - 3 * np.std(slitImages[2]),
                vmax=np.mean(slitImages[2]) + 3 * np.std(slitImages[2])
            )
        )
        slitV.set_array(slitImages[3])
        slitV.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slitImages[3]) - 3 * np.std(slitImages[3]),
                vmax=np.mean(slitImages[3]) + 3 * np.std(slitImages[3])
            )
        )

        slitFig.canvas.draw()
        slitFig.canvas.flush_events()

        for j in range(fieldImages.shape[0]):
            fieldI[j].set_array(fieldImages[j, 0])
            fieldI[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(fieldImages[j, 0, :, :step]) - 3 * np.std(fieldImages[j, 0, :, :step]),
                    vmax=np.mean(fieldImages[j, 0, :, :step]) + 3 * np.std(fieldImages[j, 0, :, :step])
                )
            )
            fieldQ[j].set_array(fieldImages[j, 1])
            fieldQ[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(fieldImages[j, 1, :, :step]) - 3 * np.std(fieldImages[j, 1, :, :step]),
                    vmax=np.mean(fieldImages[j, 1, :, :step]) + 3 * np.std(fieldImages[j, 1, :, :step])
                )
            )
            fieldU[j].set_array(fieldImages[j, 2])
            fieldU[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(fieldImages[j, 2, :, :step]) - 3 * np.std(fieldImages[j, 2, :, :step]),
                    vmax=np.mean(fieldImages[j, 2, :, :step]) + 3 * np.std(fieldImages[j, 2, :, :step])
                )
            )
            fieldV[j].set_array(fieldImages[j, 3])
            fieldV[j].set_norm(
                matplotlib.colors.Normalize(
                    vmin=np.mean(fieldImages[j, 3, :, :step]) - 3 * np.std(fieldImages[j, 3, :, :step]),
                    vmax=np.mean(fieldImages[j, 3, :, :step]) + 3 * np.std(fieldImages[j, 3, :, :step])
                )
            )
            fieldFigList[j].canvas.draw()
            fieldFigList[j].canvas.flush_events()

        if crosstalkFig is not None:
            v2q[0].set_data(internalCrosstalks[0], np.arange(internalCrosstalks.shape[1]))
            v2u[0].set_data(internalCrosstalks[1], np.arange(internalCrosstalks.shape[1]))
            u2v[0].set_data(internalCrosstalks[2], np.arange(internalCrosstalks.shape[1]))
            crosstalkFig.canvas.draw()
            crosstalkFig.canvas.flush_events()
        return


    def package_scan(self, datacube: np.ndarray, wavelength_array: np.ndarray) -> str:
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

        slit_plate_scale = self.dstPlateScale * self.dstCollimator/self.slitCameraLens
        camera_dy = slit_plate_scale * (self.spectrographCollimator/self.cameraLens) * (self.pixel_size / 1000)

        with fits.open(self.scienceFiles[0]) as hdul:
            exptime = hdul[1].header['EXPTIME']
            xposure = int(hdul[1].header['SUMS'] * exptime)
            nsumexp = hdul[1].header['SUMS']
            slitwidth = hdul[1].header['HSG_SLW']
            stepsize = hdul[1].header['HSG_STEP']
            reqmapsize = hdul[1].header['HSG_MAP']
            actmapsize = stepsize * (datacube.shape[0] - 1)
            gratingangle = hdul[1].header['HSG_GRAT']
            rsun = hdul[1].header['DST_SDIM'] / 2
            cameraName = hdul[0].header['CAMERA']

        step_startobs = []
        solarX = []
        solarY = []
        rotan = []
        llvl = []
        scin = []
        slitpos = []

        for file in self.scienceFiles:
            with fits.open(file) as hdul:
                for hdu in hdul[1:]:
                    step_startobs.append(hdu.header['DATE-OBS'])
                    rotan.append(hdu.header['DST_GDRN'] - 13.3)
                    llvl.append(hdu.header['DST_LLVL'])
                    scin.append(hdu.header['DST_SEE'])
                    slitpos.append(hdu.header['HSG_SLP'])
                    centerCoord = SkyCoord(
                        hdu.header['DST_SLNG']*u.deg, hdu.header['DST_SLAT']*u.deg,
                        obstime=hdu.header['DATE-OBS'], observer='earth', frame=frames.HeliographicStonyhurst
                    ).transform_to(frames.Helioprojective)
                    solarX.append(centerCoord.Tx.value)
                    solarY.append(centerCoord.Ty.value)

        rotan = np.nanmean(rotan)

        date, time = step_startobs[0].split("T")
        date = date.replace("-", "")
        time = str(round(float(time.replace(":", "")), 0)).split(".")[0]

        outname = self.reducedFilePattern.format(
            date,
            time,
            datacube.shape[2]
        )
        outfile = os.path.join(self.finalDir, outname)

        # Need center, have to account for maps w/ even number of steps
        if len(slitpos)%2==0:
            slitPosCenter = (slitpos[int(len(slitpos) / 2) - 1] + slitpos[int(len(slitpos) / 2)]) / 2
            centerX = (solarX[int(len(solarX) / 2) - 1] + solarX[int(len(solarX) / 2)]) / 2
            centerY = (solarY[int(len(solarY) / 2) - 1] + solarY[int(len(solarY) / 2)]) / 2
        else:
            slitPosCenter = slitpos[int(len(slitpos)/2)]
            centerX = solarX[int(len(slitpos)/2)]
            centerY = solarY[int(len(slitpos)/2)]
        # SPINOR has issues with crashing partway through maps.
        # This poses an issue for determining the center point of a given map,
        # As a crash will cause a map to be off-center relative to the telescope center
        # If the requested and actual map sizes don't match, we'll have to
        # do some math to get the actual center of the map.
        # dX = cos(90 - rotan) * slitPos at halfway point
        # dY = sin(90 - rotan) * slitPos at halfway point
        if round(reqmapsize, 4) != round(actmapsize, 4):
            dx = slitPosCenter * np.cos((90 - rotan)*np.pi/180)
            centerX += dx
            dy = slitPosCenter * np.sin((90 - rotan)*np.pi/180)
            centerY -= dy # Note sign

        # Start Assembling HDUList
        # Empty 0th HDU first
        ext0 = fits.PrimaryHDU()
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File Creation Date and Time (UTC)")
        ext0.header['ORIGIN'] = 'NMSU/SSOC'
        ext0.header['TELESCOP'] = ('DST', "Dunn Solar Telescope, Sacramento Peak NM")
        ext0.header['INSTRUME'] = ("SPINOR", "SPectropolarimetry of INfrared and Optical Regions")
        ext0.header['AUTHOR'] = "sellers"
        ext0.header['CAMERA'] = cameraName
        ext0.header['DATA_LEV'] = 1.5

        if self.centralWavelength == 6302:
            ext0.header['WAVEBAND'] = "Fe I 6301.5 AA, Fe I 6302.5 AA"
        elif self.centralWavelength == 8542:
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
            round(slit_plate_scale * slitwidth/1000, 2),
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
            self.pixel_size, self.centralWavelength, self.spectral_order,
            collimator=self.spectrographCollimator, camera=self.cameraLens, slit_width=slitwidth,
        )
        ext0.header['SPEFF'] = (float(grating_params['Grating_Efficiency']), 'Approx. Total Efficiency of Grating')
        ext0.header['LITTROW'] = (float(grating_params['Littrow_Angle']), '[degrees] Littrow Angle')
        ext0.header['RESOLVPW'] = (
            round(np.nanmean(wavelength_array) / (0.001 * float(grating_params['Spectrograph_Resolution'])), 0),
            "Maximum Resolving Power of Spectrograph"
        )

        ext0.header['RSUN_ARC'] = rsun
        ext0.header['XCEN'] = (round(centerX, 2), "[arcsec], Solar-X of Map Center")
        ext0.header['YCEN'] = (round(centerY, 2), "[arcsec], Solar-Y of Map Center")
        ext0.header['FOVX'] = (round(actmapsize, 3), "[arcsec], Field-of-view of raster-x")
        ext0.header['FOVY'] = (round(datacube.shape[0] * camera_dy, 3), "[arcsec], Field-of-view of raster-y")
        ext0.header['ROT'] = (round(rotan, 3), "[degrees] Rotation from Solar-North")

        for i in range(len(prsteps)):
            ext0.header['PRSTEP' + str(int(i+1))] = (prsteps[i], prstep_comments[i])
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

        fitsHDUs = [ext0]

        # Stokes-IQUV HDU Construction
        stokes = ['I', 'Q', 'U', 'V']
        for i in range(4):
            ext = fits.ImageHDU(datacube[:, i, :, :])
            ext.header['EXTNAME'] = 'STOKES-'+stokes[i]
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
            ext.header['CRVAL1'] = (centerX, "Solar-X, arcsec")
            ext.header['CRVAL2'] = (centerY, "Solar-Y, arcsec")
            ext.header['CRVAL3'] = (wavelength_array[0], "Angstrom")
            ext.header['CRPIX1'] = np.mean(np.arange(datacube.shape[0])) + 1
            ext.header['CRPIX2'] = np.mean(np.arange(datacube.shape[2])) + 1
            ext.header['CRPIX3'] = 1
            ext.header['CROTA2'] = (rotan, "degrees")
            fitsHDUs.append(ext)

        extWvl = fits.ImageHDU(wavelength_array)
        extWvl.header['EXTNAME'] = 'lambda-coordinate'
        extWvl.header['BTYPE'] = 'lambda axis'
        extWvl.header['BUNIT'] = '[AA]'

        fitsHDUs.append(extWvl)

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
                array=np.array(solarX)
            ),
            fits.Column(
                name='TEL_SOLY',
                format='D',
                unit='ARCSEC',
                array=np.array(solarY)
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
        extMet = fits.BinTableHDU.from_columns(columns)
        extMet.header['EXTNAME'] = 'METADATA'
        fitsHDUs.append(extMet)

        fitsHDUList = fits.HDUList(fitsHDUs)
        fitsHDUList.writeto(outfile, overwrite=True)

        return outfile


    def spinor_analysis(
            self, datacube: np.ndarray, boundIndices: np.ndarray
    ) -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
        """
        Performs moment analysis, determines mean circular/linear polarization, and net circular polarization
        maps for each of the given spectral windows. See Martinez Pillet et.al., 2011 discussion of mean polarization
        For net circular polarization, see Solanki & Montavon 1993

        Parameters
        ----------
        datacube : numpy.ndarray
            Reduced FIRS data
        boundIndices : numpy.ndarray
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
        parameter_maps = np.zeros((boundIndices.shape[0], 6, datacube.shape[0], datacube.shape[2]))
        meanProfile = np.nanmean(datacube[:, 0, :, :], axis=(0, 1))
        wavelengthArray = self.tweak_wavelength_calibration(meanProfile)
        # Tweak indices to be an even range around the line core
        tweakedIndices = []
        referenceWavelengths = []
        for i in range(boundIndices.shape[0]):
            # Integer line core
            lineCore = spex.find_line_core(
                meanProfile[int(boundIndices[i][0]):int(boundIndices[i][1])]
            ) + int(boundIndices[i][0])
            referenceWavelengths.append(
                float(scinterp.interp1d(np.arange(len(wavelengthArray)), wavelengthArray)(lineCore))
            )
            # New min
            minRange = int(round(lineCore - np.abs(np.diff(boundIndices[i]))[0]/2, 0))
            maxRange = int(round(lineCore + np.abs(np.diff(boundIndices[i]))[0]/2, 0)) + 1
            tweakedIndices.append((minRange, maxRange))
        with tqdm.tqdm(
            total=parameter_maps.shape[0] * parameter_maps.shape[2] * parameter_maps.shape[3],
            desc="Constructing Derived Parameter Maps"
        ) as pbar:
            for i in range(parameter_maps.shape[0]):
                for j in range(parameter_maps.shape[2]):
                    for k in range(parameter_maps.shape[3]):
                        spectralProfile = datacube[j, 0, k, tweakedIndices[i][0]:tweakedIndices[i][1]]
                        intens, vel, wid = spex.moment_analysis(
                            wavelengthArray[tweakedIndices[i][0]:tweakedIndices[i][1]],
                            spectralProfile,
                            referenceWavelengths[i]
                        )
                        parameter_maps[i, 0:3, j, k] = np.array([intens, vel, wid])
                        # Rather than trying to calculate a continuum value, we'll take the average of the four
                        # values on the outsize of the profiles.
                        pseudoContinuum = np.nanmean(
                            spectralProfile.take([-2, -1, 0, 1])
                        )
                        # Net V
                        parameter_maps[i, 3, j, k] = spex.net_circular_polarization(
                            datacube[j, 3, k, tweakedIndices[i][0]:tweakedIndices[i][1]],
                            wavelengthArray[tweakedIndices[i][0]:tweakedIndices[i][1]]
                        )
                        # Mean V
                        parameter_maps[i, 4, j, k] = spex.mean_circular_polarization(
                            datacube[j, 3, k, tweakedIndices[i][0]:tweakedIndices[i][1]],
                            wavelengthArray[tweakedIndices[i][0]:tweakedIndices[i][1]],
                            referenceWavelengths[i],
                            pseudoContinuum
                        )
                        # Mean QU
                        parameter_maps[i, 5, j, k] = spex.mean_linear_polarization(
                            datacube[j, 1, k, tweakedIndices[i][0]:tweakedIndices[i][1]],
                            datacube[j, 2, k, tweakedIndices[i][0]:tweakedIndices[i][1]],
                            pseudoContinuum
                        )
                        pbar.update(1)

        return parameter_maps, referenceWavelengths, tweakedIndices, meanProfile, wavelengthArray


    def package_analysis(
            self, analysis_maps: np.ndarray, rwvls: list, indices: list,
            meanProfile: np.ndarray, wavelengthArray: np.ndarray, reference_file: str
    ) -> str:
        """
        Write SPINOR first-order analysis maps to FITS file.

        Parameters
        ----------
        analysis_maps : numpy.ndarray
        rwvls : list
        indices : list
        meanProfile : numpy.ndarray
        wavelengthArray : numpy.ndarray
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
            ext0.header["WAVEMIN"] = (round(wavelengthArray[indices[i][0]], 3), "Lower Bound for Analysis")
            ext0.header["WAVEMAX"] = (round(wavelengthArray[indices[i][1]], 3), "Upper Bound for Analysis")
            ext0.header["COMMENT"] = "File contains derived parameters from moment analysis and polarization analysis"
            fitsHDUs = [ext0]
            for j in range(analysis_maps.shape[1]):
                ext = fits.ImageHDU(analysis_maps[i, j, :, :])
                ext.header = hdr1.copy()
                ext.header['EXTNAME'] = extnames[j]
                ext.header["METHOD"] = (methods[j], method_comments[j])
                fitsHDUs.append(ext)

            extWvl = fits.ImageHDU(wavelengthArray)
            extWvl.header['EXTNAME'] = 'lambda-coordinate'
            extWvl.header['BTYPE'] = 'lambda axis'
            extWvl.header['BUNIT'] = '[AA]'
            extWvl.header['COMMENT'] = "Reference Wavelength Array. For use with reference profile and WAVEMIN/MAX."
            fitsHDUs.append(extWvl)

            extRef = fits.ImageHDU(meanProfile)
            extRef.header['EXTNAME'] = 'reference-profile'
            extRef.header['BTYPE'] = 'Intensity'
            extRef.header['BUNIT'] = 'Corrected DN'
            extRef.header['COMMENT'] = "Mean spectral profile. For use with WAVEMIN/MAX."
            fitsHDUs.append(extRef)

            date, time = ext0.header['STARTOBS'].split("T")
            date = date.replace("-", "")
            time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
            outname = self.parameterMapPattern.format(
                date,
                time,
                round(ext0.header['WAVEMIN'], 2),
                round(ext0.header['WAVEMAX'], 2)
            )
            outfile = os.path.join(self.finalDir, outname)
            fitsHDUList = fits.HDUList(fitsHDUs)
            fitsHDUList.writeto(outfile, overwrite=True)

        return outfile


    @staticmethod
    def i2quv_crosstalk(stokesI: np.ndarray, stokesQUV: np.ndarray) -> np.ndarray:
        """
        Corrects for Stokes-I => QUV crosstalk. In the old pipeline, this was done by
        taking the ratio of a continuum section in I, and in QUV, then subtracting
        QUV_nu = QUV_old - ratio * I.

        We're going to take a slightly different approach. Instead of a single ratio value,
        we'll use a line, mx+b, such that QUV_nu = QUV_old - (mx+b)*I.
        We'll solve for m, b such that a second line m'x+b' fit to QUV_nu has m'=b'=0

        Parameters
        ----------
        stokesI : numpy.ndarray
            1D array of Stokes-I
        stokesQUV : numpy.ndarray
            1D array of Stokes-Q, U, or V

        Returns
        -------
        correctedQUV : numpy.ndarray
            1D array containing the Stokes-I crosstalk-corrected Q, U or V profile.

        """

        def model_function(list_of_params, i, quv):
            """Fit model"""
            xrange = np.arange(len(i))
            ilinear = list_of_params[0]*xrange + list_of_params[1]
            return quv - ilinear * i

        def error_function(list_of_params, i, quv):
            """Error function"""
            quv_corr = model_function(list_of_params, i, quv)
            xrange = np.arange(len(i))
            polyfit = np.polyfit(xrange, quv_corr, 1)
            return (xrange*polyfit[0] + polyfit[1]) - np.zeros(len(i))

        fit_result = scopt.least_squares(
            error_function,
            x0=np.array([0, 0]),
            args=[stokesI[50:-50], stokesQUV[50:-50]],
            jac='3-point', tr_solver='lsmr'
        )

        ilinearParams = fit_result.x

        correctedQUV = stokesQUV - (np.arange(len(stokesI))*ilinearParams[0] + ilinearParams[1])*stokesI

        return correctedQUV, ilinearParams


    @staticmethod
    def v2qu_crosstalk(stokesV: np.ndarray, stokesQU: np.ndarray) -> np.ndarray:
        """
        Contrary to I->QUV crosstalk, we want the Q/U profiles to be dissimilar to V.
        Q in particular is HEAVILY affected by crosstalk from V.
        Take the assumption that QU = QU - aV, where a is chosen such that the error between
        QU & V is maximized
        Parameters
        ----------
        stokesV : numpy.ndarray
            Stokes-V profile
        stokesQU : numpy.ndarray
            Stokes Q or U profile

        Returns
        -------
        correctedQU : numpy.ndarray
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

            return np.dot(v, qu_corr)/(np.linalg.norm(v) * np.linalg.norm(qu_corr))

        fit_result = scopt.least_squares(
            error_function,
            x0=0,
            args=[stokesV[50:-50], stokesQU[50:-50]],
            bounds=[-0.5, 0.5]
        )

        v2qu_crosstalk = fit_result.x

        correctedQU = stokesQU - v2qu_crosstalk * stokesV

        return correctedQU, v2qu_crosstalk

    @staticmethod
    def internal_crosstalk_2d(baseImage: np.ndarray, contaminationImage: np.ndarray) -> float:
        """
        Determines a single crosstalk value for a pair of 2D images.
        Minimizes the linear correlation between:
            baseImage - crosstalkValue * contaminationImage
                and
            contaminationImage
        The v2qu_crosstalk function should be used for individual vectors (uses cosine similarity,
        which scales to 2D poorly). This should be used as an inital guess, with v2qu_crosstalk providing
        fine corrections.

        Parameters
        ----------
        baseImage : numpy.ndarray
            2D image of a spatially-resolved Stokes vector
        contaminationImage : numpy.ndarray
            2D image of a different spatially-resolved Stokes vector that is contaminating baseImage

        Returns
        -------
        crosstalkValue : float
            Value that, when baseImage - crosstalkValue*contaminationImage is considered, minimizes correlation
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
            linCorr = np.nansum(contam_corr * contam) / np.sqrt(np.nansum(contam_corr**2) * np.nansum(contam**2))

            return linCorr
        # Clean up array for correlation
        baseImage = np.abs(baseImage) - np.nanmean(np.abs(baseImage))
        contaminationImage = np.abs(contaminationImage) - np.nanmean(np.abs(contaminationImage))
        fit_result = scopt.least_squares(
            error_function,
            x0=0,
            args=[contaminationImage[50:-50, 50:-50], baseImage[50:-50, 50:-50]],
            bounds=[-1, 1]
        )

        crosstalkValue = fit_result.x

        return crosstalkValue


    def tweak_wavelength_calibration(self, referenceProfile: np.ndarray) -> np.ndarray:
        """
        Determines wavelength array from grating parameters and FTS reference

        Parameters
        ----------
        referenceProfile : numpy.ndarray
            1D array containing a reference spectral profile from SPINOR

        Returns
        -------
        wavelengthArray : numpy.ndarray
            1D array containing corresponding wavelengths.

        """

        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.centralWavelength, self.spectral_order,
            collimator=self.spectrographCollimator, camera=self.cameraLens, slit_width=self.slit_width,
        )

        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        # Same selection process as in flat fielding.
        apxWavemin = self.centralWavelength - np.nanmean(self.slitEdges) * grating_params['Spectral_Pixel'] / 1000
        apxWavemax = self.centralWavelength + np.nanmean(self.slitEdges) * grating_params['Spectral_Pixel'] / 1000
        apxWavemin -= 30 * grating_params['Spectral_Pixel'] / 1000
        apxWavemax += 30 * grating_params['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apxWavemin, apxWavemax)

        ftsCore = sorted(np.array(self.ftsLineCores))
        if self.flipWave:
            spinorLineCores = sorted(self.slitEdges[1] - np.array(self.spinorLineCores))
        else:
            spinorLineCores = sorted(np.array(self.spinorLineCores))

        ftsCoreWaves = [scinterp.CubicSpline(np.arange(len(fts_wave)), fts_wave)(lam) for lam in ftsCore]
        # Update SPINOR selected line cores by redoing core finding with wide, then narrow range
        spinorLineCores = np.array([
            spex.find_line_core(
                referenceProfile[int(lam) - 10:int(lam) + 11]
            ) + int(lam) - 10 for lam in spinorLineCores
        ])
        spinorLineCores = np.array([
            spex.find_line_core(
                referenceProfile[int(lam) - 5:int(lam) + 7]
            ) + int(lam) - 5
            for lam in spinorLineCores
        ])
        angstrom_per_pixel = np.abs(ftsCoreWaves[1] - ftsCoreWaves[0]) / np.abs(spinorLineCores[1] - spinorLineCores[0])
        zerowvl = ftsCoreWaves[0] - (angstrom_per_pixel * spinorLineCores[0])
        wavelengthArray = (np.arange(0, len(referenceProfile)) * angstrom_per_pixel) + zerowvl
        return wavelengthArray


    def spherical_coordinate_transform(self, telescopeAngles: list) -> list:
        """
        Transforms from telescope pointing to parallatic angle using the site latitude

        Parameters
        ----------
        telescopeAngles : list
            List of telescope angles. In order, these should be (telescope_azimuth, telescope_elevation)

        Returns
        -------
        coordinateAngles : list
            List of telescope angles. In order, these will be (hour_angle, declination, parallatic angle)

        """

        sinLat = np.sin(self.DSTLatitude * np.pi/180.)
        cosLat = np.cos(self.DSTLatitude * np.pi/180.)

        sinAz = np.sin(telescopeAngles[0] * np.pi/180.)
        cosAz = np.cos(telescopeAngles[0] * np.pi/180.)

        sinEl = np.sin(telescopeAngles[1] * np.pi/180.)
        cosEl = np.cos(telescopeAngles[1] * np.pi/180.)

        sinX = -cosEl * sinAz
        cosX = sinEl*cosLat - cosEl*cosAz*sinLat

        sinY = sinEl*sinLat+cosEl*cosAz*sinLat
        sinZ = cosLat*sinAz
        cosZ = sinLat*cosEl-sinEl*sinLat*cosAz

        X = np.arctan(sinX/cosX)
        Y = np.arcsin(sinY)
        Z = -np.arctan(sinZ/cosZ)

        coordinateAngles = [X, Y, Z]

        return coordinateAngles


    def get_telescope_matrix(self, telescopeGeometry: np.ndarray) -> np.ndarray:
        """
        Gets telescope matrix from IDL save (2010 matrix) or numpy save (TBD, hopefully we measure it in the future)
        file. Returns the Mueller matrix of the telescope from these measurements.

        Parameters
        ----------
        telescopeGeometry : numpy.ndarray
            3-element vector containing the coelostat azimuth, coelostat elevation, and Coude table angle

        Returns
        -------
        tmatrix : numpy.ndarray
            4x4 Mueller matrix of telescope parameters
        """

        filename, filetype = os.path.splitext(self.tMatrixFile)
        if "idl" in filetype:
            txparams = scio.readsav(self.tMatrixFile)
        else:
            txparams = scio.readsav(self.tMatrixFile)

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

        entranceWindowOrientation = txparams['tt'][1] * np.pi/180
        exitWindowOrientation = txparams['tt'][2] * np.pi/180
        refFrameOrientation = txparams['tt'][3] * np.pi/180
        entranceWindowPolarizerOffset = txparams['tt'][4]

        wvls = txparams['tt'][5::7]
        entranceWindowRetardance = scinterp.interp1d(
            wvls, txparams['tt'][6::7], kind='linear'
        )(self.centralWavelength) * np.pi/180
        exitWindowRetardance = scinterp.interp1d(
            wvls, txparams['tt'][7::7], kind='linear'
        )(self.centralWavelength) * np.pi/180
        coelostatReflectance = scinterp.interp1d(
            wvls, txparams['tt'][8::7], kind='linear'
        )(self.centralWavelength)
        coelostatRetardance = scinterp.interp1d(
            wvls, txparams['tt'][9::7], kind='linear'
        )(self.centralWavelength) * np.pi/180
        primaryReflectance = scinterp.interp1d(
            wvls, txparams['tt'][10::7], kind='linear'
        )(self.centralWavelength)
        primaryRetardance = scinterp.interp1d(
            wvls, txparams['tt'][11::7], kind='linear'
        )(self.centralWavelength) * np.pi/180

        phiElevation = (telescopeGeometry[1] + 90) * np.pi/180
        phiAzimuth = (telescopeGeometry[2] - telescopeGeometry[0] - 30.) * np.pi/180.

        # In order, the DST optical train is:
        #   1.) Entrance Window (Retarder)
        #   2.) Elevation Coelostat (Mirror)
        #   3.) Coordinate Transform Horizontal (Rotation)
        #   4.) Azimuth Coelostat (Mirror)
        #   5.) Coordinate Transform Vertical (Rotation)
        #   6.) Primary (Mirror)
        #   7.) Exit Window (Retarder)
        #   8.) Coordinate Transform Horizontal (Rotation)

        entranceWindowMueller = spex.linearRetarder(
            entranceWindowOrientation, entranceWindowRetardance
        )
        elevationMueller = spex.mirror(
            coelostatReflectance, coelostatRetardance
        )
        azelRotationMueller = spex.rotationMueller(phiElevation)
        azimuthMueller = spex.mirror(
            coelostatReflectance, coelostatRetardance
        )
        azvertRotationMueller = spex.rotationMueller(phiAzimuth)
        primaryMueller = spex.mirror(
            primaryReflectance, primaryRetardance
        )
        exitWindowMueller = spex.linearRetarder(
            exitWindowOrientation, exitWindowRetardance
        )
        refframeRotationMueller = spex.rotationMueller(
            refFrameOrientation
        )

        # There's probably a more compact way to do this,
        # but for now, we'll just go straight down the optical chain
        tmatrix = elevationMueller @ entranceWindowMueller
        tmatrix = azelRotationMueller @ tmatrix
        tmatrix = azimuthMueller @ tmatrix
        tmatrix = azvertRotationMueller @ tmatrix
        tmatrix = primaryMueller @ tmatrix
        tmatrix = exitWindowMueller @ tmatrix
        tmatrix = refframeRotationMueller @ tmatrix

        # Normalize the Mueller matrix
        tmatrix /= tmatrix[0, 0]

        return tmatrix


    @staticmethod
    def determine_spectrum_flip(
            fts_spec: np.ndarray, spinor_spex: np.ndarray, spinPixPerFTSPix: float,
            spinorCores: list, ftsCores: list
    ) -> bool:
        """
        Determine if SPINOR spectra are flipped by correlation value against interpolated
        FTS atlas spectrum. Have to interpolate FTS to SPINOR, determine offset via correlation.
        Parameters
        ----------
        fts_spec
        spinor_spex
        spinPixPerFTSPix
        spinorCores
        ftsCores

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
        spinorEdgePad = 10
        # Edge case cleaning just in case one of the selected lines is near the edge
        if (min(spinorCores) < 10) & (spinor_spex.shape[0] - 10 < max(spinorCores)):
            # Pad out to the edge of the beam
            spinorEdgePad = min([min(spinorCores), spinor_spex.shape[0] - max(spinorCores)]) - 1

        spinor_spex = spinor_spex[min(spinorCores) - spinorEdgePad:max(spinorCores) + spinorEdgePad]

        ftsEdgePad = int(spinorEdgePad / spinPixPerFTSPix)
        fts_spec = fts_spec[int(min(ftsCores) - ftsEdgePad):int(max(ftsCores) + ftsEdgePad)]

        fts_interp = scinterp.interp1d(
            np.arange(0, fts_spec.shape[0]*spinPixPerFTSPix, spinPixPerFTSPix),
            fts_spec,
            kind='linear',
            fill_value='extrapolate'
        )(np.arange(len(spinor_spex)))

        fts_interp_reversed = fts_interp[::-1]

        lin_corr = np.nansum(
            fts_interp*spinor_spex
        ) / np.sqrt(np.nansum(fts_interp**2) * np.nansum(spinor_spex**2))

        lin_corr_rev = np.nansum(
            fts_interp_reversed * spinor_spex
        ) / np.sqrt(np.nansum(fts_interp_reversed**2) * np.nansum(spinor_spex**2))

        if lin_corr_rev > lin_corr:
            return True
        else:
            return False


    @staticmethod
    def despike_image(image: np.ndarray, footprint: tuple=(5, 1), spikeRange: tuple=(0.75, 1.25)) -> np.ndarray:
        """Removes spikes in image caused by cosmic rays, hot pixels, etc. Works off median filtering image.
        Placeholder for now. Will be replaced by a more robust function in the future.

        Parameters
        ----------
        image : numpy.ndarray
            ND image array. Length of footpoint should match the number of axes
        footprint : tuple
            Footpoint used in scipy.ndimage.median_filter to create median-smoothed image
        spikeRange : tuple
            Range to clip hot pixels from. Pixels in image/median_image exceeding range will be replaced
            by corresponding pixels in median_image

        Returns
        -------
        despikedImage : numpy.ndarray
        """

        medfiltImage = scind.median_filter(image, size=footprint)
        spikeImage = image/medfiltImage
        despikedImage = image.copy()
        despikedImage[
            (spikeImage > max(spikeRange)) | (spikeImage < min(spikeRange))
        ] = medfiltImage[(spikeImage > max(spikeRange)) | (spikeImage < min(spikeRange))]
        return despikedImage

