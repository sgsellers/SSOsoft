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
import scipy.io as scio
import scipy.optimize as scopt
import tqdm
from astropy.constants import c
import warnings

from .spectraTools import linearAnalyzerPolarizer

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

    def __init__(self, camera, configFile):
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

        self.avgDark = None
        self.solarFlat = None
        self.lampGain = None
        self.numDark = 0
        self.numSflat = 0
        self.numLflat = 0
        self.solarGain = None
        self.coarseGain = None
        self.deskewedFlat = None
        self.polcalVecs = None
        self.tMatrix = None
        self.flipWave = False
        self.crosstalkContinuum = None

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
        self.ilimt = [0.5, 1.5]
        # It's about a quarter-wave plate.
        self.calRetardance = 90

        # Basic SPINOR Feed Optics
        self.slitCameraLens = 780 # mm, f.l. of usual HSG feed lens
        self.dstPlateScale = 3.76 # asec/mm
        self.dstCollimator = 1559 # mm, f.l., of DST Port 4 Collimator mirror

        return


    def spinor_run_calibration(self):
        """Main SPINOR calibration module"""

        self.spinor_configure_run()
        self.spinor_get_cal_images()
        self.spinor_save_cal_images()
        self.spinor_wavelength_calibration()
        self.spinor_perform_scan_calibration(selectLine=self.selectLines)

        return

    def spinor_assert_file_list(flist):
        assert (len(flist) != 0), "List contains no matches."

    def spinor_configure_run(self):
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

    def spinor_parse_configfile(self):
        """
        Parses configureation file and sets up the class structure for reductions
        """

        return

    def spinor_organize_directory(self):
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
        if self.polcalFile is not None:
            polcalFilesizes = np.array([os.path.getsize(pf) for pf in polcalFiles])
            self.polcalFile = polcalFiles[polcalFilesizes.argmax()] if len(polcalFilesizes) != 0 else None
        lineGridFilesizes = np.array([os.path.getsize(lg) for lg in lineGrids])
        self.lineGridFile = lineGrids[lineGridFilesizes.argmax()] if len(lineGridFilesizes) != 0 else None
        targetFilesizes = np.array([os.path.getsize(tg) for tg in targetFiles])
        self.targetFile = targetFiles[targetFilesizes.argmax()] if len(targetFilesizes) != 0 else None

        # Allow user to override and choose flat files to use
        if self.solarFlatFile is not None:
            self.solarFlatFilelist = [self.solarFlatFile for x in scienceFiles]
        else:
            solarFlatStartTimes = np.array(
                [
                    fits.open(x)[1].header['DATE-OBS'] for x in solarFlats
                ],
                dtype='datetime64[ms]'
            )
            scienceStartTimes = np.array(
                [
                    fits.open(x)[1].header['DATE-OBS'] for x in scienceFiles
                ],
                dtype='datetime64[ms]'
            )
            self.solarFlarFileList = [
                solarFlats[spex.find_nearest(solarFlatStartTimes, x)] for x in scienceStartTimes
            ]
        if self.lampFlatFile is not None:
            lampFlatFilesizes = np.array([os.path.getsize(lf) for lf in lampFlats])
            self.lampFlatFile = lampFlats[lampFlatFilesizes.argmax()] if len(lampFlatFilesizes) != 0 else None

        return

    def spinor_average_dark_from_hdul(self, hdulist):
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
                    averageDark += np.nanmean(hdu.data, axis=0)
                    darkctr += 1
        averageDark /= darkctr
        return averageDark

    def spinor_average_flat_from_hdul(self, hdulist):
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
                    averageFlat += np.nanmean(hdu.data, axis=0)
                    flatctr += 1
        averageFlat /= flatctr
        return averageFlat


    def spinor_get_cal_images(self):
        """
        Creates average dark, flat, lampflat images. Calculates gain tables.
        """

        self.solarDark = self.spinor_average_dark_from_hdul(self.solarFlatFile)
        self.solarFlat = self.spinor_average_flat_from_hdul(self.solarFlatFile)

        if self.lampFlatFile is not None:
            lampDark = self.spinor_average_dark_from_hdul(self.lampFlatFile)
            lampFlat = self.spinor_average_flat_from_hdul(self.lampFlatFile)

            self.lampGain = (lampFlat - lampDark) / np.nanmedian(lampFlat - lampDark)
        else:
            # If there's no lamp flat file for a given day,
            # Create a dummy array of ones so we can divide by the lampgain without fear
            self.lampGain = np.ones(self.solarDark.shape)

        processed_flat = (self.solarFlat - self.solarDark)/self.lampGain

        self.beamEdges, self.slitEdges, self.hairlines = spex.detect_beams_hairlines(
            self.solarFlat - self.solarDark,
            threshold=self.beamThreshold,
            hairline_width=self.hairlineWidth,
            expected_hairlines=self.nhair, # Possible that FLIR cams only have one hairline
            expected_beams=2, # If there's only one beam, use hsgCal
            expected_slits=1 # ...we're not getting a multislit unit for SPINOR.
        )
        print(self.beamEdges)
        print(self.hairlines)
        self.slitEdges = self.slitEdges[0]

        # Determine which beam is smaller, clip larger to same size
        smaller_beam = np.argmin(np.diff(self.beamEdges, axis=1))
        larger_beam = np.argmax(np.diff(self.beamEdges, axis=1))
        # If 4 hairlines are detected, clip the beams by the hairline positions.
        # This avoids overclipping the beam, and makes the beam alignment easier.
        # And if the beams are the same size, we can skip this next step.
        if (len(self.hairlines) == 4) and (smaller_beam != larger_beam):
            print("1")
            # Since the upper beam is flipped vertically relative to the lower,
            # It does matter which beam is smaller. Must pair inner & outer hairlines.
            if smaller_beam == 0:
                print("2")
                self.beamEdges[larger_beam, 0] = int(round(
                    self.hairlines[2] -
                    (self.beamEdges[smaller_beam, 1] - self.hairlines[1]),
                    0
                ))
                self.beamEdges[larger_beam, 1] = int(round(
                    self.hairlines[3] +
                    (self.hairlines[0] - self.beamEdges[smaller_beam, 0]),
                    0
                ))
            else:
                print("3")
                self.beamEdges[larger_beam, 0] = int(round(
                    self.hairlines[0] -
                    (self.beamEdges[smaller_beam, 1] - self.hairlines[3]),
                    0
                ))
                self.beamEdges[larger_beam, 1] = int(round(
                    self.hairlines[1] +
                    (self.hairlines[2] - self.beamEdges[smaller_beam, 0]),
                    0
                )) + 1
        elif (len(self.hairlines) != 4) and (smaller_beam != larger_beam):
            print("4")
            self.beamEdges[larger_beam, 0] = int(
                np.nanmean(
                    self.beamEdges[larger_beam, :]
                ) - np.diff(self.beamEdges[smaller_beam, :]) / 2
            )
            self.beamEdges[larger_beam, 1] = int(
                np.nanmean(
                    self.beamEdges[larger_beam, :]
                ) + np.diff(self.beamEdges[smaller_beam, :]) / 2
            )

        self.hairlines = self.hairlines.reshape(2, int(self.nhair/2))

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
            beam0meanprof,
            beam1meanprof,
            mode='full'
        ).argmax() - beam0meanprof.shape[0]

        # Rather than doing a scipy.ndimage.shift, the better way to do it would
        # be to shift the self.beamEdges for beam 1.
        # This will require some extra work, as there's a possibility that
        # the beamEdges are at the end of the frame.
        # We compensate by changing the beam edges up to the edge of the
        # frame, and storing the remaining shift for later
        excess_shift = self.beamEdges[1, 1] - yshift - self.solarFlat.shape[0]
        if excess_shift > 0:
            # Y-Shift takes beam out of bounds. Update beamEdges to be at the edge of frame,
            # Store the excess shift for scipy.ndimage.shift at a later date.
            self.beam1Yshift = -excess_shift
            self.beamEdges[1] += -yshift - excess_shift
        else:
            # Y-shift does not take beam out of bound. Update beamEdges, and move on.
            self.beam1Yshift = None
            self.beamEdges[1] += -yshift

        # Redefine beam1 with new edges. Do not flip, otherwise, everything has to flip
        # Rather, we'll do the corrections on the upside-down beam, and only flip it the beam
        # after doing the corrections. Same thing with the y-shift, if applicable.
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
            collimator=self.collimator, camera=self.camera, slit_width=self.slit_width,
        )

        beam0LampGainCorrected = (
            beam0 - self.solarDark[self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]]
        ) / self.lampGain[self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]]

        beam1LampGainCorrected = (
            beam1 - self.solarDark[self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]]
        ) / self.lampGain[self.beamEdges[1, 0]:self.beamEdges[1, 1], self.slitEdges[0]:self.slitEdges[1]]

        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        apxWavemin = self.centralWavelength - np.nanmean(self.slitEdges) * grating_params['Spectral_Pixel']/1000
        apxWavemax = self.centralWavelength + np.nanmean(self.slitEdges) * grating_params['Spectral_Pixel']/1000
        apxWavemin -= 30*grating_params['Spectral_Pixel']/1000
        apxWavemax += 30*grating_params['Spectral_Pixel']/1000
        fts_wave, fts_spec = spex.fts_window(apxWavemin, apxWavemax)
        avg_profile = np.nanmedian(
            beam0LampGainCorrected[
                int(beam0LampGainCorrected.shape[0]/2 - 30):int(beam0LampGainCorrected.shape[0]/2 + 30), :
            ],
            axis=0
        )

        print("Top: SPINOR Spectrum (uncorrected). Bottom: FTS Reference Spectrum")
        print("Select the same two spectral lines on each plot.")
        spinorLines, ftsLines = spex.select_lines_doublepanel(
            avg_profile,
            fts_spec,
            4
        )
        spinorLineCores = [
            int(spex.find_line_core(avg_profile[x-5:x+5]) + x - 5) for x in spinorLines
        ]
        ftsLineCores = [
            spex.find_line_core(fts_spec[x-20:x+9]) + x - 20 for x in ftsLines
        ]

        self.spinorLineCores = spinorLineCores
        self.ftsLineCores = ftsLineCores

        # Determine whether the spectrum is flipped by comparing the correlation
        # Value between the resampled FTS spectrum and the average spectrum, both
        # flipped and default.

        spinorPixPerFTSPix = np.abs(np.diff(spinorLineCores)) / np.abs(np.diff(ftsLineCores))

        self.determine_spectrum_flip(fts_spec, avg_profile, spinorPixPerFTSPix)
        # Rather than building in logic every time we need to flip/not flip a spectrum,
        # We'll define a flip index, and slice by it every time. So if we flip, we'll be
        # indexing [::-1]. Otherwise, we'll index [::1], i.e., doing nothing to the array
        if self.flipWave:
            self.flipWaveIdx = -1
        else:
            self.flipWaveIdx = 1

        beam0GainTable, beam0CoarseGainTable, beam0Skews = spex.create_gaintables(
            beam0LampGainCorrected,
            [spinorLineCores[0] - 10, spinorLineCores[0] + 5],
            hairline_positions=self.hairlines[0] - self.beamEdges[0, 0],
            neighborhood=12,
            hairline_width=2
        )

        beam1GainTable, beam1CoarseGainTable, beam1Skews = spex.create_gaintables(
            beam1LampGainCorrected,
            [spinorLineCores[0] - 10 - self.beam1Xshift, spinorLineCores[0] + 10 - self.beam1Xshift],
            hairline_positions=self.hairlines[1] - self.beamEdges[1, 0],
            neighborhood=12,
            hairline_width=2
        )

        print("Beam0:", beam0LampGainCorrected.shape)
        print("Beam1:", beam1LampGainCorrected.shape)

        self.beam0Skews = beam0Skews
        self.beam1Skews = beam1Skews

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


    def spinor_polcal(self):
        """
        Performs polarization calibration on SPINOR data.
        Returns
        -------

        """

        polfile = fits.open(self.polcalFile)
        self.polcalDarkCurrent = self.spinor_average_dark_from_hdul(polfile)
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

        self.polcalStokesBeams = np.zeros(
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
                data = self.demodulate_spinor(
                    (hdu.data - self.polcalDarkCurrent)/self.lampGain/self.combinedGainTable
                )
                # Cut out beams, flip n shift the upper beam
                self.polcalStokesBeams[ctr, 0, :, :, :] = data[
                    :, self.beamEdges[0, 0]:self.beamEdges[0, 1], self.slitEdges[0]:self.slitEdges[1]
                ]
                self.polcalStokesBeams[ctr, 1, :, :, :] = np.flip(
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
            self.polcalStokesBeams.shape[0],
            self.polcalStokesBeams.shape[2],
            self.polcalStokesBeams.shape[3],
            self.polcalStokesBeams.shape[4]
        ))
        merged_beams[:, 0, :, :] = np.nanmean(self.polcalStokesBeams[:, :, 0, :, :], axis=1)
        merged_beams[:, 1:, :, :] = (self.polcalStokesBeams[:, 0, 1:, :, :] - self.polcalStokesBeams[:, 1, 1:, :, :])/2.
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
                self.polcalStokesBeams.shape[0],
                self.polcalStokesBeams.shape[2],
                self.nSubSlits
            )
        )
        subarrays = np.array_split(merged_beams, self.nSubSlits, axis=2)
        submasks = np.array_split(merged_beam_mask, self.nSubSlits, axis=2)
        print(subarrays[0].shape, submasks[0].shape)
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

        self.inputStokes = inputStokes
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
                warnings.warn(
                    "TX Matrix is an unphysical Mueller matrix with output minimum I: " +
                    str(muellerCheck[1]) + ", and output minimum I^2 - (Q^2 + U^2 + V^2): " +
                    str(muellerCheck[2])
                )

            # if self.plot:
            #     # Do plotting
            #     fig = plt.figure("Subslit #"+str(i)+" of "+str(self.nSubSlits), figsize=(4, 4))
            #     stokesfit = np.array([xmat @ inputStokes[j, :] for j in range(inputStokes.shape[0])])
            #     ax_i = fig.add_subplot(411)
            #     ax_q = fig.add_subplot(412)
            #     ax_u = fig.add_subplot(413)
            #     ax_v = fig.add_subplot(414)
            #     axes = [ax_i, ax_q, ax_u, ax_v]
            #     names = ['I', 'Q', 'U', 'V']
            #     for j in range(4):
            #         axes[j].plot(self.calcurves[:, j, i], label='OBSERVED')
            #         axes[j].plot(inputStokes[:, j], label='INPUT', linestyle=':')
            #         axes[j].plot(stokesfit[:, j], label='FIT')
            #         axes[j].set_title("STOKES-"+names[j])
            #         if j == 3:
            #             axes[j].legend()
            #     fig.tight_layout()
            #     if self.saveFigs:
            #         fig.savefig(
            #             os.path.join(self.calDirectory, "subslit"+str(i)+"_calcurves.png"),
            #             bbox_inches='tight'
            #         )
            #     plt.show(block=False)

        return


    def reduce_spinor_maps(self):
        """
        Performs reduction of SPINOR science data maps.
        Applies Dark Current, Lamp Gain, Solar Gain Corrections
        Applies inverse spectrograph correction matric and telescope correction matrix to data.
        Corrects QU for parallactic angle.
        Performs simple I->QUV crosstalk estimation, and (optionally) V<-->QU crosstalk estimation.

        Returns
        -------

        """

        science_hdu = fits.open(self.scienceFile)
        # fuq yea science beam
        science_beams = np.zeros((
            len(science_hdu) - 1,
            2,
            4,
            self.beamEdges[0, 1] - self.beamEdges[0, 0],
            self.slitEdges[1] - self.slitEdges[0]
        ))
        reducedData = np.zeros((
            len(science_hdu) - 1,
            4,
            self.beamEdges[0, 1] - self.beamEdges[0, 0],
            self.slitEdges[1] - self.slitEdges[0]
        ))
        shift = self.beam1Yshift if self.beam1Yshift else 0

        # Interpolate telescope inverse matrix to entire slit from nsubslits
        xinvInterp = scinterp.CubicSpline(
            np.linspace(0, self.beamEdges[0, 1]-self.beamEdges[0, 0], self.nSubSlits),
            self.txmatinv,
            axis=0,
        )(np.arange(0, self.beamEdges[0, 1]-self.beamEdges[0, 0]))

        for i in range(1, len(science_hdu)):
            print(i)
            print("Demod")
            iquv = self.demodulate_spinor(
                (science_hdu[i].data - self.solarDark)/self.lampGain/self.combinedGainTable
            )
            science_beams[i-1, 0, :, :, :] = iquv[
                :,
                self.beamEdges[0, 0]:self.beamEdges[0, 1],
                self.slitEdges[0]:self.slitEdges[1]
            ]
            science_beams[i-1, 1, :, :, :] = np.flip(
                scind.shift(
                    iquv[
                        :,
                        self.beamEdges[1, 0]:self.beamEdges[1, 1],
                        self.slitEdges[0]:self.slitEdges[1]
                    ], (0, shift, self.beam1Xshift)
                ), axis=1
            )

            # Now, to subpixel align the two beams, we have to align the hairlines
            # Then the spectral lines. We'll have to pop a widget up for the spectral lines.
            # For the hairlines, we should use the *raw* image, as the gain correction washes
            # the hairlines out. So define a beam cutout for the 0th mod state

            tmp_beams = np.zeros((
                2,
                science_beams.shape[-2],
                science_beams.shape[-1]
            ))
            tmp_beams[0] = (
                    science_hdu[i].data[0] - self.solarDark
            )[
                self.beamEdges[0, 0]:self.beamEdges[0, 1],
                self.slitEdges[0]:self.slitEdges[1]
            ]
            tmp_beams[1] = np.flip(
                scind.shift(
                    science_hdu[i].data[0][
                        self.beamEdges[1, 0]:self.beamEdges[1, 1],
                        self.slitEdges[0]:self.slitEdges[1]
                    ], (shift, self.beam1Xshift)
                ), axis=0
            )

            hairlines = self.hairlines.copy()
            hairlines[1] = self.beamEdges[1, 1] - hairlines[1] + shift

            hairlines = hairlines.flatten() - self.beamEdges[0, 0]

            hairline_skews = np.zeros((self.nhair, self.slitEdges[1] - self.slitEdges[0]))
            print("Hair deskew")
            for j in range(self.nhair):
                hairline_skews[j, :] = spex.spectral_skew(
                    np.rot90(
                        tmp_beams[int(j / 2), int(hairlines[j] - 5):int(hairlines[j] + 7), :]
                    ), order=1
                )
            avg_hairlines_skews = np.zeros((2, self.slitEdges[1] - self.slitEdges[0]))
            avg_hairlines_skews[0] = np.nanmean(
                hairline_skews[0:int(self.nhair/2), :], axis=0
            )
            avg_hairlines_skews[1] = np.nanmean(
                hairline_skews[int(self.nhair/2):, :], axis=0
            )
            for j in range(avg_hairlines_skews.shape[1]):
                science_beams[i-1, 0, :, :, j] = scind.shift(
                    science_beams[i-1, 0, :, :, j], (0, avg_hairlines_skews[0, j]),
                    mode='nearest'
                )
                science_beams[i-1, 1, :, :, j] = scind.shift(
                    science_beams[i-1, 1, :, :, j], (0, avg_hairlines_skews[1, j]),
                    mode='nearest'
                )
            print("Spec skew")
            # Reuse spectral lines from gain table creation to deskew...
            x1 = 10
            x2 = 11
            for k in range(5):
                order = 1 if k < 2 else 2
                spectral_skews = np.zeros((2, 2, science_beams.shape[-2]))
                spectral_skews[0, 0] = spex.spectral_skew(
                    science_beams[i-1, 0, 0, :, int(self.spinorLineCores[0] - x1):int(self.spinorLineCores[0]+x2)],
                    slit_reference=0.5, order=order
                )
                spectral_skews[1, 0] = spex.spectral_skew(
                    science_beams[i-1, 1, 0, :, int(self.spinorLineCores[0] - x1):int(self.spinorLineCores[0] + x2)],
                    slit_reference=0.5, order=order
                )
                spectral_skews[0, 1] = spex.spectral_skew(
                    science_beams[i - 1, 0, 0, :, int(self.spinorLineCores[1] - x1):int(self.spinorLineCores[1] + x2)],
                    slit_reference=0.5, order=order
                )
                spectral_skews[1, 1] = spex.spectral_skew(
                    science_beams[i - 1, 1, 0, :, int(self.spinorLineCores[1] - x1):int(self.spinorLineCores[1] + x2)],
                    slit_reference=0.5, order=order
                )
                spectral_skews = np.nanmean(spectral_skews, axis=1)
                for j in range(spectral_skews.shape[1]):
                    science_beams[i-1, 0, :, j, :] = scind.shift(
                        science_beams[i-1, 0, :, j, :], (0, spectral_skews[0, j])
                    )
                    science_beams[i-1, 1, :, j, :] = scind.shift(
                        science_beams[i-1, 1, :, j, :], (0, spectral_skews[1, j])
                    )
                x1 -= 1
                x2 -= 1

            print("Comb beam")
            combined_beams = np.zeros(science_beams.shape[2:])
            combined_beams[0] = np.nanmean(science_beams[i-1, :, 0, :, :], axis=0)
            combined_beams[1:] = (science_beams[i-1, 0, 1:, :, :] - science_beams[i-1, 1, 1:, :, :])/2
            tmtx = self.get_telescope_matrix(
                [science_hdu[i].header['DST_AZ'], science_hdu[i].header['DST_EL'], science_hdu[i].header['DST_TBL']]
            )
            inv_tmtx = np.linalg.inv(tmtx)
            for j in range(combined_beams.shape[1]):
                combined_beams[:, j, :] = inv_tmtx @ xinvInterp[j, :, :] @ combined_beams[:, j, :]

            # Get parallactic angle for QU rotation correction
            angular_geometry = self.spherical_coordinate_transform(
                [science_hdu[i].header['DST_AZ'], science_hdu[i].header['DST_EL']]
            )
            # Sub off P0 angle
            rotation = np.pi +angular_geometry[2] - science_hdu[i].header['DST_PEE'] * np.pi/180
            crot = np.cos(-2*rotation)
            srot = np.sin(-2*rotation)

            # Make a copy, as the Q/U components are transformed from the originals.
            qtmp = combined_beams[1, :, :].copy()
            utmp = combined_beams[2, :, :].copy()
            combined_beams[1, :, :] = crot*qtmp + srot*utmp
            combined_beams[2, :, :] = -srot*qtmp + crot*utmp

            if self.crosstalkContinuum is not None:
                # Shape 3xNY
                i2quv = np.mean(
                    combined_beams[1:, :, self.crosstalkContinuum[0]:self.crosstalkContinuum[1]] /
                    np.repeat(
                        combined_beams[0, :, self.crosstalkContinuum[0]:self.crosstalkContinuum[1]][np.newaxis, :, :],
                        3, axis=0
                    ), axis=2
                )
                combined_beams[1:] = combined_beams[1:] - np.repeat(
                    i2quv[:, :, np.newaxis], combined_beams.shape[2], axis=2
                )
            else:
                for j in range(combined_beams.shape[1]):
                    # I->QUV crosstalk correction
                    for k in range(1, 4):
                        combined_beams[k, j, : ] = self.i2quv_crosstalk(
                            combined_beams[0, j, :],
                            combined_beams[k, j, :]
                        )
                    # V->QU crosstalk correction
                    if self.v2qu:
                        for k in range(1, 3):
                            combined_beams[k, j, :] = self.v2qu_crosstalk(
                                combined_beams[3, j, :],
                                combined_beams[k, j, :]
                            )
                    if self.u2v:
                        combined_beams[3, j, :] = self.v2qu_crosstalk(
                            combined_beams[2, j, :],
                            combined_beams[3, j, :]
                        )

            # Reverse the wavelength axis if required.
            combined_beams = combined_beams[:, :, ::self.flipWaveIdx]

            reducedData[i - 1] = combined_beams

            # Choose lines for analysis. Use same method of choice as hsgCal, where user sets
            # approx. min/max, the code changes the bounds, and
            if i == 1:
                mean_profile = np.nanmean(combined_beams[0], axis=0)
                approxWavelengthArray = self.spinor_wavelength_calibration(mean_profile)
                print("Select spectral ranges (xmin, xmax) for overview maps. Close window when done.")
                # Approximate indices of line cores
                coarseIndices = spex.select_spans_singlepanel(mean_profile, xarr=approxWavelengthArray)
                # Location of minimum in the range
                minIndices = [
                    spex.find_nearest(
                        mean_profile[coarseIndices[x][0]:coarseIndices[x][1]],
                        mean_profile[coarseIndices[x][0]:coarseIndices[x][1]].min()
                    ) + coarseIndices[x][0] for x in range(coarseIndices.shape[0])
                ]
                print(minIndices)
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
                    print(mapIndices)

            if self.plot:
                plt.ion()
                # Set up overview maps to blit new data into.
                # Need maps for the slit images (IQUV) that are replaced at each step,
                # As well as IQUV maps of the full field for each line selected.
                # These latter will be filled as the map is processed.
                if i == 1:
                    fieldImages = np.zeros((
                        len(lineCores), # Number of lines
                        4, # Stokes-IQUV values
                        combined_beams.shape[1],
                        len(science_hdu[1:])
                    ))
                    figNum = 0
                    slitFigure = plt.figure("Reduced Slit Images")
                    slitGS = slitFigure.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
                    slitAxI = slitFigure.add_subplot(slitGS[0, 0])
                    slitI = slitAxI.imshow(combined_beams[0], cmap='gray', origin='lower')
                    slitAxI.text(10, 10, "I", color='C1')
                    slitAxQ = slitFigure.add_subplot(slitGS[0, 1])
                    slitQ = slitAxQ.imshow(combined_beams[1], cmap='gray', origin='lower')
                    slitAxQ.text(10, 10, "Q", color='C1')
                    slitAxU = slitFigure.add_subplot(slitGS[1, 0])
                    slitU = slitAxU.imshow(combined_beams[2], cmap='gray', origin='lower')
                    slitAxU.text(10, 10, "U", color='C1')
                    slitAxV = slitFigure.add_subplot(slitGS[1, 1])
                    slitV = slitAxV.imshow(combined_beams[3], cmap='gray', origin='lower')
                    slitAxV.text(10, 10, "V", color='C1')
                    slitFigure.tight_layout()

                    figNum += 1

                    lineFigure = []
                    lineGS = []
                    lineI = []
                    lineQ = []
                    lineU = []
                    lineV = []
                    lineIax = []
                    lineQax = []
                    lineUax = []
                    lineVax = []
                    for j in range(len(lineCores)):
                        lineFigure.append(
                            plt.figure("Line "+str(j))
                        )
                        lineGS.append(
                            lineFigure[j].add_gridspec(2, 2, hspace=0.1, wspace=0.1)
                        )
                        lineIax.append(
                            lineFigure[j].add_subplot(lineGS[j][0, 0])
                        )
                        lineI.append(
                            lineIax[j].imshow(
                                fieldImages[j, 0], origin='lower', cmap='gray'
                            )
                        )
                        lineQax.append(
                            lineFigure[j].add_subplot(lineGS[j][0, 1])
                        )
                        lineQ.append(
                            lineQax[j].imshow(
                                fieldImages[j, 1], origin='lower', cmap='gray'
                            )
                        )
                        lineUax.append(
                            lineFigure[j].add_subplot(lineGS[j][1, 0])
                        )
                        lineU.append(
                            lineUax[j].imshow(
                                fieldImages[j, 2], origin='lower', cmap='gray'
                            )
                        )
                        lineVax.append(
                            lineFigure[j].add_subplot(lineGS[j][1, 1])
                        )
                        lineV.append(
                            lineVax[j].imshow(
                                fieldImages[j, 2], origin='lower', cmap='gray'
                            )
                        )
                        lineFigure[j].tight_layout()

                        figNum += 1

                slitI.set_array(combined_beams[0])
                slitI.set_norm(
                    matplotlib.colors.Normalize(
                        vmin=np.mean(combined_beams[0])-3*np.std(combined_beams[0]),
                        vmax=np.mean(combined_beams[0])+3*np.std(combined_beams[0])
                    )
                )
                slitQ.set_array(combined_beams[1])
                slitQ.set_norm(
                    matplotlib.colors.Normalize(
                        vmin=np.mean(combined_beams[1]) - 3 * np.std(combined_beams[1]),
                        vmax=np.mean(combined_beams[1]) + 3 * np.std(combined_beams[1])
                    )
                )
                slitU.set_array(combined_beams[2])
                slitU.set_norm(
                    matplotlib.colors.Normalize(
                        vmin=np.mean(combined_beams[2]) - 3 * np.std(combined_beams[2]),
                        vmax=np.mean(combined_beams[2]) + 3 * np.std(combined_beams[2])
                    )
                )
                slitV.set_array(combined_beams[3])
                slitV.set_norm(
                    matplotlib.colors.Normalize(
                        vmin=np.mean(combined_beams[3]) - 3 * np.std(combined_beams[3]),
                        vmax=np.mean(combined_beams[3]) + 3 * np.std(combined_beams[3])
                    )
                )

                slitFigure.canvas.draw()
                slitFigure.canvas.flush_events()

                for j in range(len(lineCores)):
                    fieldImages[j, 0, :, i-1] = combined_beams[0, :, int(round(lineCores[j], 0))]
                    fieldImages[j, 1:, :, i-1] = np.sum(
                        np.abs(
                            combined_beams[1:, :, int(mapIndices[j, 0]):int(mapIndices[j ,1])]
                        ), axis=2
                    )
                    lineI[j].set_array(fieldImages[j, 0])
                    lineI[j].set_norm(
                        matplotlib.colors.Normalize(
                            vmin=np.mean(fieldImages[j, 0, :, :i]) - 3 * np.std(fieldImages[j, 0, :, :i]),
                            vmax=np.mean(fieldImages[j, 0, :, :i]) + 3 * np.std(fieldImages[j, 0, :, :i])
                        )
                    )
                    lineQ[j].set_array(fieldImages[j, 1])
                    lineQ[j].set_norm(
                        matplotlib.colors.Normalize(
                            vmin=np.mean(fieldImages[j, 1, :, :i]) - 3 * np.std(fieldImages[j, 1, :, :i]),
                            vmax=np.mean(fieldImages[j, 1, :, :i]) + 3 * np.std(fieldImages[j, 1, :, :i])
                        )
                    )
                    lineU[j].set_array(fieldImages[j, 1])
                    lineU[j].set_norm(
                        matplotlib.colors.Normalize(
                            vmin=np.mean(fieldImages[j, 2, :, :i]) - 3 * np.std(fieldImages[j, 2, :, :i]),
                            vmax=np.mean(fieldImages[j, 2, :, :i]) + 3 * np.std(fieldImages[j, 2, :, :i])
                        )
                    )
                    lineV[j].set_array(fieldImages[j, 1])
                    lineV[j].set_norm(
                        matplotlib.colors.Normalize(
                            vmin=np.mean(fieldImages[j, 3, :, :i]) - 3 * np.std(fieldImages[j, 3, :, :i]),
                            vmax=np.mean(fieldImages[j, 3, :, :i]) + 3 * np.std(fieldImages[j, 3, :, :i])
                        )
                    )
                    lineFigure[j].canvas.draw()
                    lineFigure[j].canvas.flush_events()

        mean_profile = np.nanmean(reducedData[:, 0, :, :], axis=(0, 1))
        approxWavelengthArray = self.spinor_wavelength_calibration(mean_profile)
        
        if self.write:
            self.package_scan(reducedData, approxWavelengthArray, science_hdu)
            science_hdu.close()
            return
        else:
            return reducedData, approxWavelengthArray


    def package_scan(self, datacube, wavelength_array, hdul):
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
            4D reduced stokes data in shape nx, 4, ny, nlambda
        wavelength_array : numpy.ndarray
            1D array containing the wavelengths corrsponding to nlambda in datacube
        hdul : astropy.fits.HDUList

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

        if self.v2qu:
            prsteps.append(
                'V->QU CROSSTALK'
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
        camera_dy = slit_plate_scale * (self.camera/self.collimator) * (self.pixel_size / 1000)

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
        time = time.replace(":", "")

        outname = self.reducedFilePattern.format(
            date,
            time,
            datacube.shape[0]
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
            collimator=self.collimator, camera=self.camera, slit_width=self.slit_width,
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
        ext0.header['FOVY'] = (round(datacube.shape[2] * camera_dy, 3), "[arcsec], Field-of-view of raster-y")
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

        return


    def spinor_analysis(self, datacube, wavelengths, indices, rwvls):
        """
        Performs moment analysis, determines mean circular/linear polarization, and net circular polarization
        maps for each of the given spectral windows. See Martinez Pillet et.al., 2011 discussion of mean polarization
        For net circular polarization, see Solanki & Montavon 1993

        Parameters
        ----------
        datacube
        wavelengths
        indices
        rwvls

        Returns
        -------

        """
        f=1
        return


    def i2quv_crosstalk(self, stokesI, stokesQUV):
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

        return correctedQUV


    def v2qu_crosstalk(self, stokesV, stokesQU):
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
            bounds=[-1, 1]
        )

        v2qu_crosstalk = fit_result.x

        correctedQU = stokesQU - v2qu_crosstalk * stokesV

        return correctedQU


    def spinor_wavelength_calibration(self, referenceProfile):
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
            collimator=self.collimator, camera=self.camera, slit_width=self.slit_width,
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


    def spherical_coordinate_transform(self, telescopeAngles):
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


    def get_telescope_matrix(self, telescopeGeometry):
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


    def determine_spectrum_flip(self, fts_spec, spinor_spex, spinPixPerFTSPix):
        """
        Determine if SPINOR spectra are flipped by correlation value against interpolated
        FTS atlas spectrum. Have to interpolate FTS to SPINOR, determine offset via correlation.
        Parameters
        ----------
        fts_spec
        spinor_spex
        spinPixPerFTSPix

        Returns
        -------

        """

        spinor_spex /= spinor_spex.max()

        fts_interp = scinterp.interp1d(
            np.arange(len(fts_spec)),
            fts_spec,
            kind='linear',
            fill_value='extrapolate'
        )(np.arange(0, len(fts_spec), 1/spinPixPerFTSPix))

        fts_interp_reversed = fts_interp[::-1]

        fts_shift = np.convolve(
            spinor_spex[::-1] - (fts_interp.max() + fts_interp.min())/2,
            fts_interp - (fts_interp.max() + fts_interp.min())/2,
            mode='full'
        ).argmax() - len(spinor_spex)

        fts_reverse_shift = np.convolve(
            spinor_spex[::-1] - (fts_interp_reversed.max() + fts_interp_reversed.min())/2,
            fts_interp_reversed - (fts_interp_reversed.max() + fts_interp_reversed.min())/2,
            mode='full'
        ).argmax() - len(spinor_spex)

        shifted = scind.shift(
            fts_interp,
            -fts_shift
        )[:len(spinor_spex)]
        shifted[shifted == 0] = 1

        shifted_reversed = scind.shift(
            fts_interp_reversed,
            -fts_reverse_shift
        )[:len(spinor_spex)]
        shifted_reversed[shifted_reversed == 0] = 1

        lin_corr = np.nansum(
            shifted*spinor_spex
        ) / np.sqrt(np.nansum(shifted**2) * np.nansum(spinor_spex**2))

        lin_corr_rev = np.nansum(
            shifted_reversed * spinor_spex
        ) / np.sqrt(np.nansum(shifted_reversed**2) * np.nansum(spinor_spex**2))

        if lin_corr_rev > lin_corr:
            self.flipWave = True
        return


    def demodulate_spinor(self, poldata):
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
        for i in range(stokes.shape[0]):
            for j in range(poldata.shape[0]):
                stokes[i] += self.polDemod[i, j] * poldata[j, :, :]
        return stokes
