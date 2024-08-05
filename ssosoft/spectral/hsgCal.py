import astropy.io.fits as fits
import astropy.units as u
import configparser
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as scind
import scipy.interpolate as scinterp
import tqdm
from astropy.constants import c

c_kms = c.value / 1e3
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from . import spectraTools as spex


class hsgCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    HSG (Horizontal Steerable Grating) data from the Dunn Solar Telescope.
    Note that this pipeline assumes that HSG data were taken using the AIW
    interface as the DST, and not operated through the SPINOR instrument GUI.

    ------------------------------------------------------------------------

    Use this software to process/reduce data from the HSG instrument at DST.
    To perform reductions using this package:
    1.) Install this package, along with the rest of SSOSoft.
    2.) Set the necessary instrument parameters in a configuration file.
        (Use the included rosazyla_sampleConfig.ini as a template)
    3.) Open a Python/iPython terminal and "from hsgCal import hsgCal"
    4.) Start an instance of the calibration class by using
        'h=hsgCal('<CAMERA>', '<CONFIGFILE>')'
    5.) Run the standard calibration method using
        'h.hsg_run_calibration()'

    Parameters
    ----------
    camera : str
        A string containing the camera name.
        Common values will be
            -SARNOFF_5876
            -SARNOFF_5896
            -SARNOFF_8542
            -FLIR_6302
            -SI_805_6302
            -PCO_6563
    configFile : str
        Path to the configuration file

    To my knowledge, a complete calibration should do the following:
        1.) Create average solar flat, lamp flat, and dark files
        2.) Use the solar flat to determine beam size and hairline positions, then clip the images to these values.
        3.) Deskew the dark-corrected flat.
        4.) Create a gaintable using the deskew to iteratively detrend the spectral profiles from the flat field
        5.) Perform wavelength calibration with deskewed flat field against FTS atlas, determine wavelength array.
        6.) Read in the data files, and perform the following corrections:
            a.) Subtract dark
            b.) Divide by lampgain if available
            c.) Divide by solar gain
            d.) Apply deskew shifts from flat fields.
            e.) Align hairlines with flat hairlines
        7.) Attempt a prefilter correction
        8.) Attempt a fringe correction
            Note that 7 & 8 might be easier to swap in order, since the fringes, particularly in the Sarnoff cameras
            at 8542A, affect the spectral shape pretty severely.
        9.) Write fits files for each raster.
            These files should have a master header in slot 0 with no data corresponding
            Then extension 1 has the datacube with data-specific parameters. Datacube is nx, ny, nlambda
            Extension 2 has the wavelength array with dimensions nlambda
    """

    def __init__(self, camera, configFile):
        """
        Parameters:
        -----------
        camera : str
            String containing the camera name.
        configFile : str
            Path to the configuration file
        """

        try:
            f = open(configFile, 'r')
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.configFile = configFile
        self.camera = camera.upper()

        self.avgDark = None
        self.avgFlat = None
        self.numDark = 0
        self.solarFlat = None
        self.numFlat = 0
        self.lampGain = None
        self.numLamp = 0
        self.solarGain = None
        self.coarseGain = None
        # Required for a few things, might as well store it.
        self.deskewedFlat = None
        self.prefilterCurves = None
        self.fringeTpl = None

        self.baseDir = ""
        self.reduceDir = ""
        self.finalDir = ""
        self.reducedFilePattern = ""

        self.darkList = [""]
        self.solarFlatList = [""]
        self.lampFlatList = [""]
        self.scanList = [""]

        self.darkFile = ""
        self.solarFlatFile = ""
        self.solarGainFile = ""
        self.lampGainFile = ""

        self.skewShifts = None
        self.beamEdges = []
        self.spectralEdges = []
        self.hairlines = []

        self.lineSelectionMethod = ""
        self.lineSelectionIndices = []
        self.ftsSelectionIndices = []
        self.deskewShifts = []
        self.fringeMethod = ""

        self.centralWavelength = 0
        self.approxDispersion = 0
        self.slitWidth = 0
        self.stepSpacing = 0
        self.arcsecPerPixel = 0
        self.beamThreshold = 0.5
        self.pfcDegree = 0
        self.fringeWave = 0

        self.referenceFTSspec = []
        self.referenceFTSwave = []
        self.wavelengthArray = []

        self.flipWave = False

        self.velocityMapLineIndex = None
        self.refwvls = []

    def hsg_run_calibration(self):
        """
        The main calibration method for standard HSG data.
        """
        self.hsg_configure_run()
        self.hsg_get_cal_images()  # Includes creating cal images if req'd
        self.hsg_save_cal_images()
        self.hsg_wavelength_calibration()
        self.hsg_prefilter_profile_calculation()
        if self.fringeMethod != "NONE":
            self.hsg_fringe_template()
        else:
            self.fringeTpl = np.ones(self.solarGain.shape)
        self.hsg_perform_scan_calibration(selectLine=True)  # Includes velocity determination of selected line
        return

    def hsg_configure_run(self):
        """
        Reads configuration file and sets up parameters necessary for calibration run.
        """

        def hsg_assert_file_list(flist):
            assert (len(flist) != 0), "List contains no matches."

        config = configparser.ConfigParser()
        config.read(self.configFile)

        self.baseDir = config[self.camera]['baseDir']
        self.reduceDir = config[self.camera]['reduceDir']
        self.finalDir = os.path.join(self.reduceDir, "calibratedScans")
        self.reducedFilePattern = config[self.camera]['reducedFilePattern']
        if not os.path.isdir(self.finalDir):
            print("{0}: os.mkdir: attempting to create directory:"
                  "{1}".format(__name__, self.finalDir))
            try:
                os.mkdir(self.finalDir)
            except Exception as err:
                print("An exception was raised: {0}".format(err))
                raise

        self.darkFile = os.path.join(self.reduceDir, "{0}_DARK.fits".format(self.camera))
        self.solarFlatFile = os.path.join(self.reduceDir, "{0}_SOLARFLAT.fits".format(self.camera))
        self.solarGainFile = os.path.join(self.reduceDir, "{0}_SOLARGAIN.fits".format(self.camera))
        self.lampGainFile = os.path.join(self.reduceDir, "{0}_LAMPGAIN.fits".format(self.camera))

        self.solarFlatList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['solarFlatPattern']
        )))
        try:
            hsg_assert_file_list(self.solarFlatList)
        except AssertionError as err:
            print("Error: solarFlatList: {0}".format(err))
            raise
        else:
            print("Files in solar flat list: {0}".format(len(self.solarFlatList)))

        self.darkList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['darkPattern']
        )))
        try:
            hsg_assert_file_list(self.darkList)
        except AssertionError as err:
            print("Error: darkList: {0}".format(err))
            raise
        else:
            print("Files in dark list: {0}".format(len(self.darkList)))
        if 'lampFlatPattern'.lower() in list(config[self.camera].keys()):
            self.lampFlatList = sorted(glob.glob(os.path.join(
                self.baseDir, config[self.camera]['lampFlatPattern']
            )))
            try:
                hsg_assert_file_list(self.darkList)
            except AssertionError as err:
                print("Error: lampFlatList: {0}".format(err))
                pass
            else:
                print("Files in lamp flat list: {0}".format(len(self.lampFlatList)))

        self.scanList = sorted(glob.glob(os.path.join(
            self.baseDir, config[self.camera]['dataPattern']
        )))
        try:
            hsg_assert_file_list(self.scanList)
        except AssertionError as err:
            print("Error: scanList: {0}".format(err))
            raise
        else:
            print("Files in scan list: {0}".format(len(self.scanList)))

        self.centralWavelength = float(config[self.camera]['centralWavelength'])
        self.approxDispersion = float(config[self.camera]['approximateDispersion'])
        self.slitWidth = float(config[self.camera]['slitwidth'])
        self.arcsecPerPixel = float(config[self.camera]['arcsecPerPixelY'])
        if 'beamEdgeThreshold'.lower() in list(config[self.camera].keys()):
            self.beamThreshold = float(config[self.camera]['beamEdgeThreshold'])

        if len(config[self.camera]['lineSelection'].split(",")) > 1:
            self.lineSelectionIndices = [
                int(x) for x in config[self.camera]['lineSelection'].split(',')
            ]
            self.lineSelectionMethod = "USER"
        else:
            self.lineSelectionMethod = "MANUAL"

        if 'prefilterDegree'.lower() in list(config[self.camera].keys()):
            self.pfcDegree = int(config[self.camera]['prefilterDegree'])
        self.fringeMethod = config[self.camera]['fringeCorrection'].upper()
        if 'fringeWavelength'.lower() in list(config[self.camera].keys()):
            self.fringeWave = float(config[self.camera]['fringeWavelength'])
        else:
            self.fringeWave = 'SELECT'

        if config[self.camera]['reverseWave'].upper() == "TRUE":
            self.flipWave = True
        return

    def hsg_average_image_from_list(self, fileList):
        """
        Computes an average image from a list of HSG image files.

        Parameters:
        -----------
        fileList : list
            A list of file paths to the images to be averaged

        Returns:
        --------
        numpy.ndarray
            2-Dimensional with dtype float
        """
        if len(fileList) == 0:
            return None, 0
        testImg = fits.open(fileList[0])
        testExt = testImg[1].data
        # TCOPS makes spectral images in the shape (1, NY, NX)
        avgImg = np.zeros(testExt[0].shape)
        imNum = 0
        for file in fileList:
            with fits.open(file) as hdu:
                for ext in hdu[1:]:
                    if self.flipWave:
                        avgImg += np.flip(ext.data[0], axis=1)
                    else:
                        avgImg += ext.data[0]
                    imNum += 1
        avgImg = avgImg / imNum
        return avgImg, imNum

    def hsg_get_cal_images(self):
        """
        If the average dark, solar flat, lamp gain, and solar gain files exist, reads them into the class.
        Otherwise, creates these files and saves them.
        """
        if os.path.exists(self.darkFile):
            with fits.open(self.darkFile) as hdu:
                self.avgDark = hdu[0].data
        else:
            self.avgDark, self.numDark = self.hsg_average_image_from_list(self.darkList)
        if os.path.exists(self.solarFlatFile):
            with fits.open(self.solarFlatFile) as hdu:
                self.avgFlat = hdu[0].data
        else:
            self.avgFlat, self.numFlat = self.hsg_average_image_from_list(self.solarFlatList)
        if os.path.exists(self.lampGainFile):
            with fits.open(self.lampGainFile) as hdu:
                self.lampGain = hdu[0].data
        else:
            self.lampGain, self.numLamp = self.hsg_compute_lamp_gain()
        if os.path.exists(self.solarGainFile):
            with fits.open(self.solarGainFile) as hdu:
                self.solarGain = hdu[0].data
                self.coarseGain = hdu[1].data
                self.beamEdges = [hdu[0].header['BEAM1'], hdu[0].header['BEAM2']]
                self.spectralEdges = [hdu[0].header['SLIT1'], hdu[0].header['SLIT2']]
                hairs = [key for key in list(hdu[0].header.keys()) if "HAIR" in key]
                self.hairlines = [hdu[0].header[key] for key in hairs]
                self.deskewShifts = hdu[2].data
                self.lineSelectionIndices = hdu[3].data
                self.ftsSelectionIndices = hdu[4].data
                self.referenceFTSwave = hdu[5].data
                self.referenceFTSspec = hdu[6].data
        else:
            self.hsg_compute_gain()
        return

    def hsg_save_cal_images(self):
        """
        Save Flat, Dark, Lampgain, and Gain Files. I'm choosing not to write the FITS writer function general
        enough to write both cals and data files, as the header requirements are more complicated for the data
        files, and I still want SOME information in the cal headers. Writing a function to do both would necessitate
        another function to contruct cal file specific fits headers, which, while best practice,
        is not neccessary for our use case.
        :return:
        """
        if not os.path.exists(self.darkFile):
            hdu = fits.HDUList([fits.PrimaryHDU(self.avgDark)])
            hdu[0].header['DATATYPE'] = "AVGDARK"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numDark
            hdu.writeto(self.darkFile, overwrite=True)
        if not os.path.exists(self.solarFlatFile):
            hdu = fits.HDUList([fits.PrimaryHDU(self.avgFlat)])
            hdu[0].header['DATATYPE'] = "AVGSFLAT"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numFlat
            hdu.writeto(self.solarFlatFile, overwrite=True)
        if (not os.path.exists(self.lampGainFile)) & (self.lampGain is not None):
            hdu = fits.HDUList([fits.PrimaryHDU(self.lampGain)])
            hdu[0].header['DATATYPE'] = "LAMPGAIN"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numLamp
            hdu.writeto(self.lampGainFile, overwrite=True)
        if not os.path.exists(self.solarGainFile):
            hdu = fits.HDUList([
                fits.PrimaryHDU(self.solarGain),
                fits.ImageHDU(self.coarseGain),
                fits.ImageHDU(self.deskewShifts),
                fits.ImageHDU(self.lineSelectionIndices),
                fits.ImageHDU(self.ftsSelectionIndices),
                fits.ImageHDU(self.referenceFTSwave),
                fits.ImageHDU(self.referenceFTSspec)
            ])
            hdu[0].header['DATATYPE'] = "SGAIN"
            hdu[0].header['INSTR'] = "HSG"
            hdu[0].header['CAMERA'] = self.camera
            hdu[0].header['AUTHOR'] = "sellers"
            hdu[0].header['NSUMEXP'] = self.numFlat
            hdu[0].header['BEAM1'] = self.beamEdges[0]
            hdu[0].header['BEAM2'] = self.beamEdges[1]
            hdu[0].header['SLIT1'] = self.spectralEdges[0]
            hdu[0].header['SLIT2'] = self.spectralEdges[1]
            for i in range(len(self.hairlines)):
                hdu[0].header['HAIR' + str(i + 1)] = self.hairlines[i]
            hdu[0].header['EXTNAME'] = 'FINEGAIN'
            hdu[1].header['EXTNAME'] = 'COARSEGAIN'
            hdu[2].header['EXTNAME'] = 'DESKEWSHIFTS'
            hdu[3].header['EXTNAME'] = 'GAINLINES'
            hdu[4].header['EXTNAME'] = 'FTSLINES'
            hdu[5].header['EXTNAME'] = 'FTSWAVE'
            hdu[6].header['EXTNAME'] = 'FTSSPEC'

            hdu.writeto(self.solarGainFile, overwrite=True)
        return

    def hsg_compute_lamp_gain(self):
        """
        Creates average lamp gain image (that is, dark corrected and normalized)
        """
        avgLampFlat, numLamp = self.hsg_average_image_from_list(self.lampFlatList)
        if avgLampFlat is not None:
            lampGain = avgLampFlat - self.avgDark
            lampGain = lampGain / np.nanmedian(lampGain)
            return lampGain, numLamp
        else:
            return None, 0

    def hsg_compute_gain(self):
        """
        Computes solar gain and finds the beam edges.
        WHY IS THIS DOING WORSE THAN THE TEST CASE?????
        Has to do with line selection...
        Original line selection works. Is there a crop issue?
        :return:
        """
        # Step 0: Determine beam/spectral edges & hairline positions.
        beam_edges, slit_edges, hairlines = spex.detect_beams_hairlines(
            self.avgFlat,
            threshold=self.beamThreshold
        )
        self.beamEdges = beam_edges[0]
        self.spectralEdges = slit_edges[0]
        # Position of hairlines AFTER image is cropped.
        self.hairlines = hairlines - self.beamEdges[0]

        if self.lampGain is not None:
            croppedFlat = (self.avgFlat - self.avgDark) / self.lampGain
            croppedFlat = croppedFlat[
                          self.beamEdges[0]:self.beamEdges[1],
                          self.spectralEdges[0]:self.spectralEdges[1]
                          ]
        else:
            croppedFlat = self.avgFlat - self.avgDark
            croppedFlat = croppedFlat[
                          self.beamEdges[0]:self.beamEdges[1],
                          self.spectralEdges[0]:self.spectralEdges[1]
                          ]
        # Step 1: Line Selections
        if self.lineSelectionMethod == "MANUAL":
            # Get an approximate reference spectrum from FTS
            apxWavemin = self.centralWavelength - np.nanmean(self.spectralEdges) * self.approxDispersion
            apxWavemax = self.centralWavelength + np.nanmean(self.spectralEdges) * self.approxDispersion
            # Fudge: Pad out FTS selection, since the central wavelength and dispersion are VERY approximate
            apxWavemin = apxWavemin - 50 * self.approxDispersion
            apxWavemax = apxWavemax + 50 * self.approxDispersion
            fts_wave, fts_spec = spex.fts_window(apxWavemin, apxWavemax)
            centralProfile = np.nanmedian(
                croppedFlat[
                    int(croppedFlat.shape[0] / 2 - 30):int(croppedFlat.shape[0] / 2 + 30), :
                ],
                axis=0
            )
            print("Top: HSG Spectrum (uncorrected). Bottom: FTS Reference Spectrum.")
            print("Select the same two spectral lines on each plot.")
            hsgLines, ftsLines = spex.select_lines_doublepanel(
                centralProfile,
                fts_spec,
                4
            )
            hsgLineCores = [
                int(spex.find_line_core(centralProfile[x - 5:x + 5]) + x - 5) for x in hsgLines
            ]
            ftslinecores = [
                spex.find_line_core(fts_spec[x - 20:x + 9]) + x - 20 for x in ftsLines
            ]
            # Step 1.5: Save reference profile and line centers for later
            print(hsgLines, hsgLineCores)
            self.lineSelectionIndices = hsgLineCores
            self.ftsSelectionIndices = ftslinecores
            self.referenceFTSwave = fts_wave
            self.referenceFTSspec = fts_spec

        # Step 2: Iterate gain table per selected line
        # Note that I've found an issue in the lpff method of line finding. At the bottom of the slit,
        # it tends to produce spurious results. I think it's because at the bottom of the slit, the profile
        # tends to be towards the start of the slice with a lot after it.
        # To compensate, we'll go with -10/+4 for the slice instead of even on either side.
        # Since this is just for HSG, I'm okay (if hesitant) hardcoding it. But since we took the fiducial slice
        # from the center of the array, and profiles not in the center are leftwards, I think we'll be okay...
        if not self.flipWave:
            gain, coarse, deskew = spex.create_gaintables(
                croppedFlat,
                [self.lineSelectionIndices[0] - 10, self.lineSelectionIndices[0] + 5],
                hairline_positions=self.hairlines
            )
        if self.flipWave:
            gain, coarse, deskew = spex.create_gaintables(
                croppedFlat,
                [self.lineSelectionIndices[0] - 5, self.lineSelectionIndices[0] + 10],
                hairline_positions=self.hairlines
            )
        self.solarGain = gain
        self.coarseGain = coarse
        self.deskewShifts = deskew
        # gains = []
        # coarse_gains = []
        # deskews = []
        # gainmean = []
        # for line in self.lineSelectionIndices:
        #     g, cg, desk = spex.create_gaintables(
        #         croppedFlat,
        #         [line - 7, line + 7],
        #         hairline_positions=self.hairlines
        #     )
        #     gains.append(g)
        #     coarse_gains.append(cg)
        #     deskews.append(desk)
        #     gainmean.append(np.nanmean(g))
        # # Step 3: Find the gain table with the mean nearest 1 (i.e., best table)
        # gainidx = spex.find_nearest(gainmean, 1)
        # self.solarGain = gains[gainidx]
        # self.coarseGain = coarse_gains[gainidx]
        # self.deskewShifts = deskews[gainidx]
        return

    def hsg_prefilter_profile_calculation(self):
        """
        Determines prefilter curves along the slit
        """
        if self.pfcDegree == 0:
            self.prefilterCurves = np.ones(self.deskewedFlat.shape)
        else:
            flat_image = (self.avgFlat - self.avgDark) / self.lampGain
            flat_image = flat_image[self.beamEdges[0]:self.beamEdges[1], self.spectralEdges[0]:self.spectralEdges[1]]
            flat_image /= self.solarGain
            self.prefilterCurves = spex.prefilter_correction(
                flat_image,
                self.wavelengthArray,
                self.referenceFTSspec,
                self.referenceFTSwave,
                polynomial_order=self.pfcDegree
            )
        return

    def hsg_fringe_template(self):
        """
        Sets up parameters used for Fourier fringe correction.
        Writing this one only after the rest of the pipeline has been tested.
        :return:
        """
        if self.fringeWave == 'SELECT':
            mean_profile = np.nanmean(
                self.deskewedFlat[
                    int(self.deskewedFlat.shape[0] / 2 - 30):int(self.deskewedFlat.shape[0] / 2 + 30), :
                ], axis=0)
            self.fringeWave = spex.select_fringe_freq(
                self.wavelengthArray,
                mean_profile,
                1 / 0.5
            )
        self.fringeTpl = spex.fourier_fringe_correction(
            self.deskewedFlat,
            self.fringeWave,
            (12, 4),
            self.wavelengthArray[1] - self.wavelengthArray[0]
        )
        self.fringeTpl /= np.nanmedian(self.fringeTpl)

        return

    def hsg_wavelength_calibration(self):
        """
        Performs wavelength calibration from gain-corrected, deskewed, average flat.
        :return:
        """
        croppedFlat = self.avgFlat - self.avgDark
        if self.lampGain is not None:
            croppedFlat = croppedFlat / self.lampGain
        croppedFlat = croppedFlat[self.beamEdges[0]:self.beamEdges[1], self.spectralEdges[0]:self.spectralEdges[1]]
        croppedFlat = croppedFlat / self.solarGain
        deskewedFlat = np.zeros(croppedFlat.shape)
        for i in range(deskewedFlat.shape[0]):
            deskewedFlat[i, :] = scind.shift(
                croppedFlat[i, :],
                self.deskewShifts[i],
                mode='nearest'
            )
        self.deskewedFlat = deskewedFlat
        meanProfile = np.nanmean(
            deskewedFlat[int(deskewedFlat.shape[0] / 2 - 30):int(deskewedFlat.shape[0] / 2 + 30), :],
            axis=0
        )

        deskewedCenters = np.array(self.lineSelectionIndices) - int(self.deskewShifts[int(deskewedFlat.shape[0] / 2)])
        trueCenters = [
            spex.find_line_core(meanProfile[x - 7:x + 7]) + x + 7 for x in deskewedCenters
        ]
        ftsRefWvls = [
            scinterp.interp1d(
                np.arange(len(self.referenceFTSwave)),
                self.referenceFTSwave,
                kind='linear'
            )(x) for x in self.ftsSelectionIndices
        ]
        angstrom_per_pixel = np.abs(ftsRefWvls[1] - ftsRefWvls[0]) / np.abs(trueCenters[1] - trueCenters[0])
        zerowvl = ftsRefWvls[0] - (angstrom_per_pixel * trueCenters[0])
        self.wavelengthArray = (np.arange(0, len(meanProfile)) * angstrom_per_pixel) + zerowvl
        return

    def hsg_perform_scan_calibration(self, selectLine=True):
        """
        Perfoms calibration of science data, triggers the
        :param selectLine: bool
            If true, prompts the user to select lines for use in velocity products.
            Otherwise, only corrects products and saves with no additional data
        :return:
        """
        for i in tqdm.tqdm(range(len(self.scanList)), desc="Correcting Scans"):
            with (fits.open(self.scanList[i]) as schdu):
                # Corrected data cubes are set up as ny, nx, nlambda
                # This is consistent with the FIRS L-1.5 data products.
                correctedDataCube = np.zeros(
                    (
                        self.beamEdges[1] - self.beamEdges[0],
                        len(schdu) - 1,
                        self.spectralEdges[1] - self.spectralEdges[0]
                    )
                )
                # Determine slit step size
                # This is not noted in the obs logs, but IS noted in the HSG header
                # Which stores the entire AIW script.
                # We'll do a quick parse of the script and grab the line with "step" in it.
                if "COMMENT" in list(schdu[0].header.keys()):
                    aiwScript = list(schdu[0].header['COMMENT'])
                    # Lines with "#" are comment lines
                    aiwSteps = [x for x in aiwScript if "step" in x and "#" not in x]
                    # Relies on the somewhat naive assumption that there's only one step command in the script
                    # And that the step size is the last thing in that line.
                    # This should be okay -- sometimes the effort needed to generalize a thing is greater
                    # than the effort needed to write a hack that will work 99.9999999% of the time.
                    self.stepSpacing = float(aiwSteps[0].split(" ")[-1])
                # If HSG was run by GUI
                else:
                    self.stepSpacing = float(schdu[1].header['HSG_STEP'])
                timestamps = []
                dst_slat = []
                dst_slng = []
                dst_gdran = []
                dst_sdim = []
                dst_llvl = []
                dst_see = []
                exptime = []
                for j in range(1, len(schdu)):
                    timestamps.append(schdu[j].header['DATE-OBS'])
                    dst_slat.append(schdu[j].header['DST_SLAT'])
                    dst_slng.append(schdu[j].header['DST_SLNG'])
                    dst_gdran.append(schdu[j].header['DST_GDRN'])
                    dst_sdim.append(schdu[j].header['DST_SDIM'])
                    dst_llvl.append(schdu[j].header['DST_LLVL'])
                    dst_see.append(schdu[j].header['DST_SEE'])
                    exptime.append(schdu[j].header['EXPTIME'])
                    rasterStep = schdu[j].data[0]
                    if self.flipWave:
                        rasterStep = np.flip(rasterStep, axis=1)
                    rasterStep = rasterStep - self.avgDark
                    if self.lampGain is not None:
                        rasterStep /= self.lampGain
                    rasterStep = rasterStep[
                                 self.beamEdges[0]:self.beamEdges[1],
                                 self.spectralEdges[0]:self.spectralEdges[1]
                                 ]
                    rasterStep /= self.solarGain
                    rasterStep /= self.prefilterCurves
                    # Extra step here: applying the deskew from the gain table creation step.
                    # There should be no wavelength variation along the slit now.
                    deskraster = np.zeros(rasterStep.shape)
                    for k in range(rasterStep.shape[0]):
                        deskraster[k, :] = scind.shift(
                            rasterStep[k, :],
                            self.deskewShifts[k],
                            mode='nearest'
                        )
                    # New extra step: determining bulk shift between deskewed flat mean profile
                    # and the scan deskewed profile. We'll reuse common code for it.
                    bulkShift = spex.iterate_shifts(
                        np.nanmean(self.deskewedFlat[
                                   int(self.deskewedFlat.shape[0] / 2 - 30):int(self.deskewedFlat.shape[0] / 2 + 30),
                                   10:-10
                                   ], axis=0),
                        np.nanmean(deskraster[
                                   int(deskraster.shape[0] / 2 - 30):int(deskraster.shape[0] / 2 + 30), 10:-10
                                   ], axis=0)
                    )
                    correctedDataCube[:, j - 1, :] = scind.shift(
                        np.nan_to_num(deskraster),
                        (0, bulkShift),
                        mode='nearest'
                    ) / self.fringeTpl
                metaparams = np.rec.fromarrays(
                    [
                        np.array(timestamps, dtype='datetime64[ms]'),
                        np.array(exptime),
                        np.array(dst_slat),
                        np.array(dst_slng),
                        np.array(dst_gdran) - 13.3,  # 13.3 degree offset thanks to old ESG
                        np.array(dst_sdim),
                        np.array(dst_llvl),
                        np.array(dst_see)
                    ],
                    names=[
                        'TIMESTAMPS', 'EXPTIME',
                        'DST_SLAT', 'DST_SLNG', 'DST_GDRN',
                        'DST_SDIM', 'DST_LLVL', 'DST_SEE'
                    ]
                )
                if selectLine:
                    imaps, vmaps, wmaps = self.hsg_create_moment_maps(correctedDataCube)
                    self.packageScan(
                        correctedDataCube,
                        metaparams,
                        imapList=imaps,
                        vmapList=vmaps,
                        wmapList=wmaps,
                        rwvlList=self.refwvls
                    )
                else:
                    self.packageScan(correctedDataCube, metaparams)
        return

    def hsg_create_moment_maps(self, spectralCube):
        """
        Generates integrated intensity, Doppler velocity, and Doppler width maps from user-selected spectral lines.
        These parameters are found using moment analysis.
        If self.velocityMapLineIndex is None, will prompt the user to select lines from the mean of the gain-corrected
        flat field.
        :param spectralCube: numpy.ndarray
            Reduced datacube of the shape (ny, nx, nlambda)
        :return imaps: list
            List of intensity maps for each selected range
        :return vmaps: list
            List of LOS velocity maps for each selected range
        :return wmaps: list
            List of Doppler width maps for each selected range
        """
        if self.velocityMapLineIndex is None:
            flat = self.deskewedFlat / self.fringeTpl
            meanProfile = np.nanmean(
                flat[int(flat.shape[0] / 2 - 30):int(flat.shape[0] / 2 + 30), :],
                axis=0
            )
            # meanProfile = np.nanmean(
            #     spectralCube[int(spectralCube.shape[0]/2 - 30):int(spectralCube.shape[0]/2 + 30), :, :],
            #     axis=(0, 1)
            # )
            # User-selected ranges
            vmapCoarseIndices = spex.select_spans_singlepanel(meanProfile, xarr=self.wavelengthArray)
            # Minimum location of each line in range
            vmapMidIndices = [
                spex.find_nearest(
                    meanProfile[vmapCoarseIndices[x][0]:vmapCoarseIndices[x][1]],
                    meanProfile[vmapCoarseIndices[x][0]:vmapCoarseIndices[x][1]].min()
                ) + vmapCoarseIndices[x][0] for x in range(vmapCoarseIndices.shape[0])
            ]
            # Exact core location
            vmapCores = [
                spex.find_line_core(meanProfile[x - 7:x + 7]) + x - 7 for x in vmapMidIndices
            ]
            # And their wavelengths...
            refwvls = [
                float(scinterp.interp1d(
                    np.arange(len(self.wavelengthArray)),
                    self.wavelengthArray,
                    kind='linear'
                )(x)) for x in vmapCores
            ]

            # List comprehension, my beloved
            # Now, find start and end indices that put refwvls at the center of the window.
            vmapIndices = np.zeros(vmapCoarseIndices.shape)
            for i in range(vmapCoarseIndices.shape[0]):
                averageDelta = np.nanmean(np.abs(vmapCoarseIndices[i, :] - int(vmapCores[i])))
                vmapIndices[i, 0] = int(vmapCores[i] - averageDelta)
                vmapIndices[i, 1] = int(vmapCores[i] + averageDelta)
            self.velocityMapLineIndex = vmapIndices.astype(int)
            self.refwvls = refwvls
        imaps = []
        vmaps = []
        wmaps = []
        for i in tqdm.tqdm(range(self.velocityMapLineIndex.shape[0]), desc='Creating L1.5 Maps'):
            imap = np.zeros((spectralCube.shape[0], spectralCube.shape[1]))
            vmap = np.zeros(imap.shape)
            wmap = np.zeros(imap.shape)
            with tqdm.tqdm(total=vmap.shape[0] * vmap.shape[1]) as pbar:
                pbar.set_description("Performing Moment Analyses...")
                for j in range(imap.shape[0]):
                    for k in range(imap.shape[1]):
                        intens, vel, wid = spex.moment_analysis(
                            self.wavelengthArray[self.velocityMapLineIndex[i][0]:self.velocityMapLineIndex[i][1]],
                            spectralCube[j, k, self.velocityMapLineIndex[i][0]:self.velocityMapLineIndex[i][1]],
                            self.refwvls[i]
                        )
                        imap[j, k] = intens
                        vmap[j, k] = vel
                        wmap[j, k] = wid
                        pbar.update(1)
            imaps.append(imap)
            vmaps.append(vmap)
            wmaps.append(wmap)
        return imaps, vmaps, wmaps

    def hsg_create_velocity_maps(self, spectralCube):
        """
        Generates velocity maps of the spectral cube.
        If self.velocityMapLineIndex is None, it prompts the user to select lines from a
        gain-corrected flat field. These values are used as the rest frame velocity.
        It then sets the user-generated values to a class attribute to be used on the
        rest of the main calibration loop.
        :param spectralCube: numpy.ndarray
            Reduced datacube of the shape (ny, nx, nlambda)
        :return vmaps: list
            List of velocity maps in km/s for each line chosen.
        """
        if not self.velocityMapLineIndex:
            meanProfile = np.nanmean(
                spectralCube[int(spectralCube.shape[0] / 2 - 30):int(spectralCube.shape[0] / 2 + 30), :, :],
                axis=(0, 1)
            )
            vmapCoarseIndices = spex.select_lines_singlepanel_unbound_xarr(meanProfile, xarr=self.wavelengthArray)
            vmapFineIndices = [
                spex.find_line_core(meanProfile[x - 7:x + 7]) + x + 7 for x in vmapCoarseIndices
            ]
            self.velocityMapLineIndex = vmapFineIndices
        vmaps = []
        refwvls = []
        for index in self.velocityMapLineIndex:
            refwvl = scinterp.interp1d(
                np.arange(len(self.wavelengthArray)),
                self.wavelengthArray,
                kind='linear'
            )(index)
            refwvls.append(float(refwvl))
            vmap = np.zeros((spectralCube.shape[0], spectralCube.shape[1]))
            for i in range(spectralCube.shape[0]):
                for j in range(spectralCube.shape[1]):
                    cwvl = spex.find_line_core(
                        spectralCube[i, j, int(index - 10):int(index + 10)],
                        wvl=self.wavelengthArray[int(index - 10):int(index + 10)]
                    )
                    vmap[i, j] = c_kms * (cwvl - refwvl) / refwvl
            vmaps.append(vmap)
        return vmaps, refwvls

    def packageScan(
            self,
            correctedScan, metadata,
            imapList=None, vmapList=None, wmapList=None, rwvlList=None):
        """
        Packages corrected scans, along with the level-1.5 velocity products
        :param correctedScan: numpy.ndarray
            Dark, gain corrected scan for save.
        :param metadata: nump.rec.array
            Numpy recarray of import header attributes
        :param imapList: list of numpy.ndarray
            List of 2D numpy.ndarrays corresponding to intensity maps
        :param vmapList: list of numpy.ndarray
            List of 2D numpy.ndarrays corresponding to velocity maps
        :param wmapList: list of numpy.ndarray
            List of 2D numpy.ndarrays corresponding to Doppler width maps
        :param rwvlList: list of float
            List of wavelength values used as reference wavelengths for corresponding vmap.
        :return:
        """
        t0 = metadata['TIMESTAMPS'][0]
        t1 = metadata['TIMESTAMPS'][-1] + np.timedelta64(int(metadata['EXPTIME'][-1]), "ms")
        date, time = metadata['TIMESTAMPS'][0].astype('datetime64[s]').astype(str).split("T")
        date = date.replace("-", "")
        time = time.replace(":", "")
        outname = self.reducedFilePattern.format(
            date,
            time,
            correctedScan.shape[1]
        )
        outfile = os.path.join(self.finalDir, outname)

        prsteps = [
            'DARK-SUBTRACTION',
            'FLATFIELDING',
            'WAVELENGTH-CALIBRATION'
        ]
        prstep_comments = [
            'hsgPy/SSOSoft',
            'hsgPy/SSOSoft',
            'hsgPy/SSOSoft, FTS atlas'
        ]
        if self.fringeMethod != "NONE":
            prsteps.append("FRINGE-CORRECTION")
            prstep_comments.append("hsgPy/SSOSoft, Fourier Filter")
        if self.pfcDegree != 0:
            prsteps.append("PREFILTER-CORRECTION")
            prstep_comments.append("hsgPy/SSOSoft, FTS Comparison")
        if vmapList:
            prsteps.append("MOMENT-ANALYSIS")
            prstep_comments.append("hsgPy/SSOSoft, Moment Analysis")

        centerCoord = SkyCoord(
            metadata['DST_SLNG'][0] * u.deg, metadata['DST_SLAT'][0] * u.deg,
            obstime=t0.astype(str), observer='earth', frame=frames.HeliographicStonyhurst
        ).transform_to(frames.Helioprojective)

        fovx = correctedScan.shape[1] * self.stepSpacing
        fovy = correctedScan.shape[0] * self.arcsecPerPixel

        ext0 = fits.PrimaryHDU(correctedScan)
        ext0.header['EXTNAME'] = 'SPECTRAL-DATA'
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File creation date and time (UTC)")
        # Base origin keywords
        ext0.header['ORIGIN'] = ("NMSU/SSOC", "Institution that created this FITS file")
        ext0.header['TELESCOP'] = ('DST', "Telescope used to acquire this data")
        ext0.header['INSTRUME'] = ("HSG", "Instrument used to acquire this data")
        ext0.header['CAMERA'] = self.camera
        if vmapList:
            ext0.header['DATA_LEV'] = (1.5, "Includes derived parameters")
        else:
            ext0.header['DATA_LEV'] = 1
        # Timing
        ext0.header['STARTOBS'] = t0.astype(str)
        ext0.header['ENDOBS'] = t1.astype(str)
        ext0.header['EXPTIME'] = metadata['EXPTIME'][0]
        ext0.header['BTYPE'] = 'Intensity'
        ext0.header['BUNIT'] = 'Corrected DN'
        # Coordinates (Set 1)
        ext0.header['XCEN'] = (round(centerCoord.Tx.value, 3), "Center X-Coordinate (arcsec)")
        ext0.header['YCEN'] = (round(centerCoord.Ty.value, 3), "Center Y-Coordinate (arcsec)")
        ext0.header['FOVX'] = (fovx, "Field-of-view in X (arcsec)")
        ext0.header['FOVY'] = (fovy, "Field-of-view in Y (arcsec)")
        ext0.header['ROT'] = (round(metadata['DST_GDRN'][0], 3), "Relative to Solar-North (deg)")
        ext0.header['D_SUN'] = (metadata['DST_SDIM'][0], "Solar Disk Diameter (arcsec)")
        # Coordinates (Set 2)
        ext0.header['CDELT1'] = (self.stepSpacing, "arcsec")
        ext0.header['CDELT2'] = (self.arcsecPerPixel, "arcsec")
        ext0.header['CDELT3'] = (round(self.wavelengthArray[1] - self.wavelengthArray[0], 3), "Angstrom")
        ext0.header['CTYPE1'] = 'HPLN-TAN'
        ext0.header['CTYPE2'] = 'HPLT-TAN'
        ext0.header['CTYPE3'] = 'WAVE'
        ext0.header['CUNIT1'] = 'arcsec'
        ext0.header['CUNIT2'] = 'arcsec'
        ext0.header['CUNIT3'] = 'Angstrom'
        ext0.header['CRVAL1'] = round(centerCoord.Tx.value, 3)
        ext0.header['CRVAL2'] = round(centerCoord.Ty.value, 3)
        ext0.header['CRVAL3'] = round(self.wavelengthArray[0], 3)
        ext0.header['CRPIX1'] = correctedScan.shape[0] / 2
        ext0.header['CRPIX2'] = correctedScan.shape[1] / 2
        ext0.header['CRPIX3'] = 1
        ext0.header['CROTA2'] = round(metadata['DST_GDRN'][0], 3)
        for i in range(len(self.hairlines)):
            ext0.header['HAIRLN' + str(i + 1)] = self.hairlines[i]
        # Processing history
        for i in range(len(prsteps)):
            ext0.header['PRSTEP' + str(i + 1)] = (prsteps[i], prstep_comments[i])

        fitsHDUs = [ext0]

        fitsHeaderKeys = list(ext0.header.keys())[8:]

        # Write i-map extensions
        for i in range(len(imapList)):
            ext = fits.ImageHDU(imapList[i])
            for key in fitsHeaderKeys:
                ext.header[key] = (ext0.header[key], ext0.header.comments[key])
            ext.header['EXTNAME'] = 'IMAP-' + str(i + 1)
            ext.header['BTYPE'] = 'Integrated Intensity'
            ext.header['BUNIT'] = 'int(DN)'
            ext.header['REFWVL'] = rwvlList[i]
            ext.header['WAVE1'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 0]],
                "Lower-bound wavelength for moment analysis"
            )
            ext.header['WAVE2'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 1]],
                "Upper-bound wavelength for moment analysis"
            )
            ext.header['METHOD'] = 'Moment Analysis'
            del ext.header['CDELT3']
            del ext.header['CTYPE3']
            del ext.header['CUNIT3']
            del ext.header['CRVAL3']
            del ext.header['CRPIX3']
            fitsHDUs.append(ext)

        # Write v-map extensions
        for i in range(len(vmapList)):
            ext = fits.ImageHDU(vmapList[i])
            for key in fitsHeaderKeys:
                ext.header[key] = (ext0.header[key], ext0.header.comments[key])
            ext.header['EXTNAME'] = 'VMAP-' + str(i + 1)
            ext.header['BTYPE'] = 'LOS Velocity'
            ext.header['BUNIT'] = 'km/s'
            ext.header['REFWVL'] = rwvlList[i]
            ext.header['WAVE1'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 0]],
                "Lower-bound wavelength for moment analysis"
            )
            ext.header['WAVE2'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 1]],
                "Upper-bound wavelength for moment analysis"
            )
            ext.header['METHOD'] = 'Moment Analysis'
            del ext.header['CDELT3']
            del ext.header['CTYPE3']
            del ext.header['CUNIT3']
            del ext.header['CRVAL3']
            del ext.header['CRPIX3']
            fitsHDUs.append(ext)

        # Write w-map extensions
        for i in range(len(wmapList)):
            ext = fits.ImageHDU(wmapList[i])
            for key in fitsHeaderKeys:
                ext.header[key] = (ext0.header[key], ext0.header.comments[key])
            ext.header['EXTNAME'] = 'WMAP-' + str(i + 1)
            ext.header['BTYPE'] = 'Velocity Width'
            ext.header['BUNIT'] = 'km/s'
            ext.header['REFWVL'] = rwvlList[i]
            ext.header['WAVE1'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 0]],
                "Lower-bound wavelength for moment analysis"
            )
            ext.header['WAVE2'] = (
                self.wavelengthArray[self.velocityMapLineIndex[i, 1]],
                "Upper-bound wavelength for moment analysis"
            )
            ext.header['METHOD'] = 'Moment Analysis'
            del ext.header['CDELT3']
            del ext.header['CTYPE3']
            del ext.header['CUNIT3']
            del ext.header['CRVAL3']
            del ext.header['CRPIX3']
            fitsHDUs.append(ext)

        # Finally, metadata extension
        # Get start of each exposure expressed as seconds since midnight
        timedeltas = (metadata['TIMESTAMPS'] - metadata['TIMESTAMPS'][0].astype("datetime64[D]"))
        timedeltas = timedeltas.astype('timedelta64[ms]').astype(float) / 1000
        columns = [
            fits.Column(
                name='WAVELNGTH',
                format='D',
                unit='ANGSTROM',
                array=self.wavelengthArray
            ),
            fits.Column(
                name='T_ELAPSED',
                format='D',
                unit='SECONDS',
                array=timedeltas,
                time_ref_pos=metadata['TIMESTAMPS'][0].astype('datetime64[D]').astype(str)
            ),
            fits.Column(
                name='EXPTIMES',
                format='D',
                unit='MS',
                array=metadata['EXPTIME']
            ),
            fits.Column(
                name='STONYLAT',
                format='D',
                unit='DEG',
                array=metadata['DST_SLAT']
            ),
            fits.Column(
                name='STONYLNG',
                format='D',
                unit='DEG',
                array=metadata['DST_SLNG']
            ),
            fits.Column(
                name='CROTA2',
                format='D',
                unit='DEG',
                array=metadata['DST_GDRN']
            ),
            fits.Column(
                name='LIGHTLVL',
                format='D',
                unit='UNITLESS',
                array=metadata['DST_LLVL']
            ),
            fits.Column(
                name='SCINT',
                format='D',
                unit='ARCSEC',
                array=metadata['DST_SEE']
            )
        ]

        ext = fits.BinTableHDU.from_columns(columns)
        ext.header['EXTNAME'] = 'METADATA'
        fitsHDUs.append(ext)

        fitsHDUList = fits.HDUList(fitsHDUs)
        try:
            fitsHDUList.writeto(outfile, overwrite=True)
        except Exception:
            print("Could not write FITS file: ""{0}".format(outfile))
            print("Continuing...")

        return
