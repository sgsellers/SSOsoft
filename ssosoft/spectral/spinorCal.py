import astropy.io.fits as fits
import astropy.units as u
import configparser
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as scind
import scipy.interpolate as scinterp
import scipy.io as scio
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

        # Expressed as a fraction of mean(I). For polcals
        self.ilimt = [0.5, 1.5]
        # It's about a half-wave plate.
        self.calRetardance = 90

        return


    def spinor_run_calibration(self):
        """Main SPINOR calibration module"""

        self.spinor_configure_run()
        self.spinor_get_cal_images()
        self.spinor_save_cal_images()
        self.spinor_wavelength_calibration()
        self.spinor_perform_scan_calibration(selectLine=self.selectLines)

        return

    def spinor_configure_run(self):
        """Reads configuration file and sets up parameters for calibration sequence"""

        def spinor_assert_file_list(flist):
            assert (len(flist) != 0), "List contains no matches."

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
        lampDark = self.spinor_average_dark_from_hdul(self.lampFlatFile)
        lampFlat = self.spinor_average_flat_from_hdul(self.lampFlatFile)

        self.lampGain = (lampFlat - lampDark) / np.nanmedian(lampFlat - lampDark)

        self.solarDark = self.spinor_average_dark_from_hdul(self.solarFlatFile)
        self.solarFlat = self.spinor_average_flat_from_hdul(self.solarFlatFile)

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
                self.beamEdges[larger_beam, 0] = int(
                    self.hairlines[2] -
                    (self.beamEdges[smaller_beam, 1] - self.hairlines[1])
                )
                self.beamEdges[larger_beam, 1] = int(
                    self.hairlines[3] +
                    (self.hairlines[0] - self.beamEdges[smaller_beam, 0])
                )
            else:
                print("3")
                self.beamEdges[larger_beam, 0] = int(
                    self.hairlines[0] -
                    (self.beamEdges[smaller_beam, 1] - self.hairlines[3])
                )
                self.beamEdges[larger_beam, 1] = int(
                    self.hairlines[1] +
                    (self.hairlines[2] - self.beamEdges[smaller_beam, 0])
                )
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
        # X shift will be taken care of during skew calculation and gain creation
        yshift = np.correlate(
            (np.nanmean(beam0, axis=0) - np.nanmean(beam0)),
            (np.nanmean(beam1, axis=0) - np.nanmean(beam1)),
            mode='full'
        ).argmax() - beam0.shape[1]

        # beam1_aligned, alignment_shifts = spex.image_align(beam1, beam0)
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
            hairline_positions=self.hairlines[0],
            neighborhood=12,
            hairline_width=2
        )

        beam1GainTable, beam1CoarseGainTable, beam1Skews = spex.create_gaintables(
            beam1LampGainCorrected,
            [spinorLineCores[0] - 10, spinorLineCores[0] + 10],
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
                        ], (0, shift, 0)
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

            # Measurement of retardance from +-QU measurements
            # +Q
            lpPosQSelection = lpOnlySelection & (np.abs(polarizerAngle) < 1 | np.abs(polarizerAngle - 180) < 1)
            posQVec = np.nanmean(
                self.calcurves[:, :, i][np.repeat(lpPosQSelection[:, np.newaxis], 4, axis=1)], axis=0
            )
            stokesPosQ = self.txmatinv[i] @ posQVec
            stokesPosQ = stokesPosQ / stokesPosQ[0]
            posQsqdiff = (stokesPosQ - np.array([1, 1, 0, 0])**2)
            # -Q
            lpNegQSelection = lpOnlySelection & (np.abs(polarizerAngle - 90) < 1 | np.abs(polarizerAngle - 270) < 1)
            negQVec = np.nanmean(
                self.calcurves[:, :, i][np.repeat(lpNegQSelection[:, np.newaxis], 4, axis=1)], axis=0
            )
            stokesNegQ = self.txmatinv[i] @ negQVec
            stokesNegQ = stokesNegQ / stokesNegQ[0]
            negQsqdiff = (stokesNegQ - np.array([1, -1, 0, 0]) ** 2)
            # +U
            lpPosUSelection = lpOnlySelection & (np.abs(polarizerAngle - 45) < 1 | np.abs(polarizerAngle - 225) < 1)
            posUVec = np.nanmean(
                self.calcurves[:, :, i][np.repeat(lpPosUSelection[:, np.newaxis], 4, axis=1)], axis=0
            )
            stokesPosU = self.txmatinv[i] @ posUVec
            stokesPosU = stokesPosU / stokesPosU[0]
            posUsqdiff = (stokesPosU - np.array([1, 0, 1, 0]) ** 2)
            # -U
            lpNegUSelection = lpOnlySelection & (np.abs(polarizerAngle - 135) < 1 | np.abs(polarizerAngle - 135) < 1)
            negUVec = np.nanmean(
                self.calcurves[:, :, i][np.repeat(lpNegUSelection[:, np.newaxis], 4, axis=1)], axis=0
            )
            stokesNegU = self.txmatinv[i] @ negUVec
            stokesNegU = stokesNegU / stokesNegU[0]
            negUsqdiff = (stokesNegU - np.array([1, 0, -1, 0]) ** 2)

            self.txchi[i] = posQsqdiff + negQsqdiff + posUsqdiff + negUsqdiff

            if self.verbose:
                print("TX Matrix:")
                print(xmat)
                print("Inverse:")
                print(self.txmatinv[i])
                print("Efficiencies:")
                print("Q: "+efficiencies[1], "U: "+efficiencies[2], "V: "+efficiencies[3])
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

            if self.verbose:
                # Do plotting
                fig = plt.figure("Subslit #"+str(i)+" of "+str(self.nSubSlits), figsize=(4, 4))
                stokesfit = np.array([xmat @ inputStokes[j, :] for j in range(inputStokes.shape[0])])
                ax_i = fig.add_subplot(411)
                ax_q = fig.add_subplot(412)
                ax_u = fig.add_subplot(413)
                ax_v = fig.add_subplot(414)
                axes = [ax_i, ax_q, ax_u, ax_v]
                names = ['I', 'Q', 'U', 'V']
                for j in range(4):
                    axes[j].plot(self.calcurves[:, j, i], label='OBSERVED')
                    axes[j].plot(inputStokes[:, j], label='INPUT', linestyle=':')
                    axes[j].plot(stokesfit[:, j], label='FIT')
                    axes[j].set_title("STOKES-"+names[j])
                    if j == 3:
                        axes[j].legend()
                if self.saveFigs:
                    fig.savefig(
                        os.path.join(self.calDirectory, "subslit"+str(i)+"_calcurves.png"),
                        bbox_inches='tight'
                    )
                plt.show(block=False)

        return


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

        # print(entranceWindowRetardance)
        # print(entranceWindowOrientation)
        # print(exitWindowRetardance)
        # print(exitWindowOrientation)
        # print(refFrameOrientation)
        # print(coelostatReflectance)
        # print(coelostatRetardance)
        # print(primaryReflectance)
        # print(primaryRetardance)
        # print(entranceWindowPolarizerOffset)

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
        # print(entranceWindowMueller)
        elevationMueller = spex.mirror(
            coelostatReflectance, coelostatRetardance
        )
        # print(elevationMueller)
        azelRotationMueller = spex.rotationMueller(phiElevation)
        # print(azelRotationMueller)
        azimuthMueller = spex.mirror(
            coelostatReflectance, coelostatRetardance
        )
        # print(azimuthMueller)
        azvertRotationMueller = spex.rotationMueller(phiAzimuth)
        # print(azvertRotationMueller)
        primaryMueller = spex.mirror(
            primaryReflectance, primaryRetardance
        )
        # print(primaryMueller)
        exitWindowMueller = spex.linearRetarder(
            exitWindowOrientation, exitWindowRetardance
        )
        # print(exitWindowMueller)
        refframeRotationMueller = spex.rotationMueller(
            refFrameOrientation
        )
        # print(refframeRotationMueller)

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