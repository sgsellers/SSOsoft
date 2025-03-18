import configparser
import glob
import logging, logging.config
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
from . import polarimetryTools as pol
# We can piggyback off the static methods in spinorCal
from . import spinorCal


class FirsCal:
    """
    The Sunspot Solar Observatory Consortium's software for reducing
    FIRS (Facility InfraRed Spectropolarimeter) data from the
    Dunn Solar Telescope.

    -----------------------------------------------------------------

    Use this package to process data from the FIRS instrument at the DST.
    To perform reductions using this package:
    1.) Install SSOSoft
    2.) Set the necessary instrument parameters in a configuration file.
        The included spectroConfig.ini has a template with all mandatory
        and optional keywords.
    3.) Open a python/iPython session and "from ssosoft.spectral import firsCal"
    4.) Start an instance of the calibration class by using:
        firs = firsCal("<CONFIGFILE>")
    5.) Run the standard calibration procedure using:
        fits.fits_run_calibration()

    Parameters
    ----------
    config_file : str
        Path to configuration file
    """

    def __init__(self, config_file: str) -> None:
        """

        Parameters
        ----------
        config_file : str
            Path to configuration file
        """

        try:
            f = open(config_file, "r")
            f.close()
        except Exception as err:
            print("Exception: {0}".format(err))
            raise

        self.config_file = config_file

        self.solar_dark = None
        self.solar_flat = None
        self.lamp_gain = None
        self.combined_gain_table = None
        self.combined_coarse_gain_table = None
        self.polcal_vecs = None
        self.t_matrix = None
        self.flip_wave = False
        self.beam_rotation = []

        # Locations
        self.indir = ""
        self.final_dir = ""
        self.reduced_file_pattern = None
        self.parameter_map_pattern = None
        self.obssum_info = [] # Will be recarray of obs series information
        self.polcal_file_list = []
        self.solar_flat_file_list = []
        self.lamp_flat_file_list = []
        self.solar_dark_file_list = []
        self.lamp_dark_file_list = []
        self.science_series_list = []
        self.science_file_list = []
        self.line_grid_file_list = []
        self.target_file_list = []

        # Cal file:
        self.t_matrix_file = None

        # For saving the reduced calibration files:
        self.solar_gain_reduced = None  # we'll include dark currents in these files
        self.lamp_gain_reduced = ""
        self.tx_matrix_reduced = ""

        # Setting up variables to be filled:
        self.beam_edges = None
        self.slit_edges = None
        self.hairlines = None
        self.full_hairlines = None
        self.rotated_beam_sizes = None
        self.beam_shifts = None
        self.firs_line_cores = None
        self.fts_line_cores = None
        self.flip_wave_idx = 1

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
        self.nslits = 1
        self.beam_threshold = 0.5
        self.hairline_width = 3
        self.grating_rules = 31.6  # lpmm
        self.blaze_angle = 63.5
        self.grating_angle = 63.5 - 3.75 # Default 10830/15648 value
        # Usually, FIRS is operated at 10830, but the code should be able to handle 15648
        self.central_wavelength = 10830
        # 52nd order for 10830, 36th order for 15648
        self.spectral_order = 52
        self.pixel_size = 20 # um -- for the Virgo 1k, probably.
        self.slit_width = 40  # um -- needs to be changed if slit unit is swapped out.
        self.n_subslits = 10
        self.verbose = False
        self.v2q = True
        self.v2u = True
        self.q2v = False
        self.u2v = False
        self.despike = False
        self.despike_footprint = (1, 5, 1)
        self.plot = False
        self.save_figs = False
        self.crosstalk_continuum = None
        self.defringe = "flat" # Construct a fringe template from nearest solar flat series.
        # Future releases will include PCA-based defringe.

        # Basic FIRS Feed Optics
        self.slit_camera_lens = 780  # mm, f.l. of usual HSG feed lens
        self.telescope_plate_scale = 3.76  # asec/mm
        self.dst_collimator = 1559  # mm, f.l., of DST Port 4 Collimator mirror
        # For FIRS, there's an off-axis parabola that both acts as the spectrograph collimator
        # and the re-imaging lens after the grating. For the IR arm, the beam is then re-collimated
        # By a 400 mm lens, then re-imaged onto the SEIR Virgo 1k array by a 400 mm lens,
        # i.e., these two lenses cancel each other out.
        # I have never gotten a straight answer and the Zemax files are long lost.
        # I ~think~ Virgo has 24 um pixels.
        self.spectrograph_collimator = 1524  # mm, f.l., post-slit collimator
        self.camera_lens = 1524  # mm, same as collimator for FIRS
        # (there are beam-extending lenses, but they're symmetric)

        # Default polarization modulation from LCVRs (2025-03)
        # I: ++++
        # Q: --++
        # U: +-+-
        # V: +--+
        self.pol_demod = np.array([
            [1, 1, 1, 1],
            [-1, -1, 1, 1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ], dtype=int)
        self.pol_norm = np.array([0.5/2., 0.866/2., 0.866/2., 0.866/2.])

        self.site_latitude = 32.786

        # Expressed as a fraction of mean(I). For polcals
        self.ilimit = [0.5, 1.5]
        # It's about a quarter-wave plate.
        self.cal_retardance = 83

        return

    def firs_run_calibration(self) -> None:
        """Main FIRS calibration module"""
        self.firs_configure_run()
        # Sometimes the observers add extra notes in the obstype, so it may not match "SCAN" exactly
        # For compatibility with numpy.where, we'll use numpy.char.find, which will return -1 where the
        # substring is not found.
        self.science_series_list = np.where(
            np.char.find(self.obssum_info['OBSTYPE'], "SCAN") != -1
        )[0]
        if self.verbose:
            print(
                "Found {0} science map series in base directory:"
                "\n{1}\nReduced files will be saved to:\n{2}".format(
                    len(self.science_series_list), self.indir, self.final_dir
            ))
        for index in self.science_series_list:
            self.__init__(self.config_file)
            self.firs_configure_run()
            self.science_series_list = np.where(
                np.char.find(self.obssum_info['OBSTYPE'], "SCAN") != -1
            )[0]
            self.firs_get_cal_images(index)
            if self.plot:
                # Need to clear out previous plot instances
                plt.pause(2)
                plt.close("all")
            self.science_file_list = sorted(glob.glob(self.science_series_list[index] + "*"))
            fringe_template = None
            if self.defringe == "flat":
                sflt_indices = np.where(["SFLT" in i for i in self.obssum_info['OBSTYPE']])[0]
                sflt_starts = self.obssum_info['STARTTIME'][sflt_indices]
                obs_start = self.obssum_info['STARTTIME'][index]
                sflat_index = sflt_indices[spex.find_nearest(sflt_starts, obs_start)]
                fringe_template = self.reduce_firs_maps(sflat_index, write=False, overview=False, fringe_template=None)

            self.reduce_firs_maps(
                index, overview=self.plot, write=True, fringe_template=fringe_template,
                v2q=self.v2q, v2u=self.v2u, q2v=self.q2v, u2v=self.u2v
            )
        return

    def firs_configure_run(self) -> None:
        """
        Reads config file and parses input directory
        """
        self.firs_parse_configfile()
        self.firs_parse_directory()

        return

    def firs_get_cal_images(self, index: int) -> None:
        """
        Loads or creates flat, dark, lamp & solar gain, polcal arrays
        """
        self.firs_get_solar_flat(index)
        self.firs_get_lamp_gain()
        self.firs_get_solar_gain(index)
        self.firs_get_polcal()

        return

    def firs_get_solar_flat(self, index: int) -> None:
        """
        Loods or creates solar flat and solar dark
        """
        if os.path.exists(self.solar_gain_reduced[index]):
            with fits.open(self.solar_gain_reduced[index]) as hdul:
                self.solar_flat = hdul['SOLAR-FLAT'].data
                self.solar_dark = hdul['SOLAR-DARK'].data
        else:
            sflt_indices = np.where(["SFLT" in i for i in self.obssum_info['OBSTYPE']])[0]
            sflt_starts = self.obssum_info['STARTTIME'][sflt_indices]
            obs_start = self.obssum_info['STARTTIME'][index]
            sflat_index = sflt_indices[spex.find_nearest(sflt_starts, obs_start)]
            self.solar_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][sflat_index])
            dark_indices = np.where(['DARK' in i for i in self.obssum_info['OBSTYPE']])[0]
            dark_index = spex.find_nearest(dark_indices, sflat_index)
            self.solar_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][dark_index])
        return

    def firs_get_solar_gain(self, index: int) -> None:
        """
        Creates or loads a solar gain file.
        """
        if os.path.exists(self.solar_gain_reduced[index]):
            with fits.open(self.solar_gain_reduced[index]) as hdul:
                self.combined_coarse_gain_table = hdul['COARSE-GAIN'].data
                self.combined_gain_table = hdul['GAIN'].data
                self.beam_edges = hdul["BEAM-EDGES"].data
                self.hairlines = hdul['HAIRLINES'].data
                self.slit_edges = hdul["SLIT-EDGES"].data
                self.beam_rotation = hdul["BEAM-ROTATION"].data
                self.beam_shifts = hdul['BEAM-SHIFTS'].data[0]
                self.firs_line_cores = [hdul[0].header['LC1'], hdul[0].header['LC2']]
                self.fts_line_cores = [hdul[0].header['FTSLC1'], hdul[0].header['FTSLC2']]
        else:
            self.firs_create_solar_gain()
            self.firs_save_gaintables(index)

        if self.plot:
            self.firs_plot_gaintables(index)

        return

    def firs_create_solar_gain(self) -> None:
        """
        Creates solar gain tables from a flat. Also determines beam/slit edges, hairline position,
        rough offsets between upper and lower beams, and relative rotation between beams.
        """

        self.beam_edges, self.slit_edges, self.hairlines = spex.detect_beams_hairlines(
            self.solar_flat - self.solar_dark,
            threshold=self.beam_threshold,
            hairline_width=self.hairline_width,
            expected_hairlines=self.nhair,
            expected_beams=2,
            expected_slits=self.nslits,
            fallback=True
        )
        if self.verbose:
            print("===========================")
            print("Lower Beam Edges in Y: ", self.beam_edges[0])
            print("Upper Beam Edges in Y: ", self.beam_edges[1])
            for slit in range(self.nslits):
                print("Slit {0} of {1} X-Range: ".format(slit+1, self.nslits), self.slit_edges[slit])
            print("There are {0} hairlines at ".format(self.nhair), self.hairlines)
            print("===========================\n\n")

        self.hairlines = self.hairlines.reshape(2, 2)

        # Per slit, per beam hairlines
        self.full_hairlines = np.zeros((self.nslits, 2, 2))

        # Grab the size of every beam and slit after rotations.
        # We'll clip every beam to this size during combine:
        self.rotated_beam_sizes = np.zeros((2, self.nslits, 2)) # Y/X shape, slit, beam

        # Shifts relative to slit 0, beam 0:
        self.beam_shifts = np.zeros((2, self.nslits, 2)) # Y/X shifts, slit, beam

        # Determine rotation of each beam from hairline slope
        # Remember -- FIRS beams are not flipped relative to each other.
        # Benefits of a Wollaston instead of the prism PBS
        self.beam_rotation = np.zeros((2, self.nslits)) # Per beam, per slit
        master_image = (self.solar_flat - self.solar_dark)[
                       self.beam_edges[0, 0]:self.beam_edges[0, 1], self.slit_edges[0, 0]:self.slit_edges[0, 1]
                       ]
        for beam, i in zip(self.beam_edges, range(2)):
            for slit, j in zip(self.slit_edges, range(self.nslits)):
                image = (self.solar_flat - self.solar_dark)[beam[0]:beam[1], slit[0]:slit[1]]
                rotations = []
                for hair, k in zip(self.hairlines[i], range(int(self.nhair/2))):
                    hair_image = np.rot90(image[hair-5:hair+7, :])
                    # While we've got it, grab the hairline center from this image
                    hairline_center = spex.find_line_core(
                        hair_image[:, int(hair_image.shape[1]/2 - 20):int(hair_image.shape[1]/2 - 20)].mean(axis=1)
                    ) + hair - 5
                    self.full_hairlines[j, i, k] = hairline_center
                    skews = spex.spectral_skew(hair_image[25:-25], order=1) # Clip the edges of the slit a bit
                    slope = (skews.max() - skews.min())/(hair_image.shape[0] - 50)
                    angle = -np.arctan(slope)*180/np.pi # Negative due to direction of rot90
                    rotations.append(angle)
                self.beam_rotation[i, j] = np.mean(rotations)
                rotated_image = scind.rotate(image, self.beam_rotation[i, j])
                self.rotated_beam_sizes[:, j, i] = rotated_image.shape
                if i == j == 0:
                    master_image = rotated_image
                else:
                    min_shape = np.minimum(master_image.shape, rotated_image.shape)
                    master_clipped = master_image[tuple(slice(s) for s in min_shape)]
                    rotate_clipped = rotated_image[tuple(slice(s) for s in min_shape)]
                    xshift = np.correlate(
                        (np.nanmean(master_clipped, axis=0) - np.nanmean(master_clipped))[10:-10],
                        (np.nanmean(rotate_clipped, axis=0) - np.nanmean(rotate_clipped))[10:-10],
                        mode='full'
                    ).argmax() - master_clipped.shape[1] - 20
                    yshift = np.correlate(
                        (np.nanmean(master_clipped, axis=1) - np.nanmean(master_clipped))[10:-10],
                        (np.nanmean(rotate_clipped, axis=1) - np.nanmean(rotate_clipped))[10:-10],
                        mode='full'
                    ).argmax() - master_clipped.shape[0] - 20
                    self.beam_shifts[:, j, i] = (yshift, xshift)

        cleaned_solar_flat = self.clean_flat(self.solar_flat.copy())
        cleaned_solar_flat -= self.solar_dark
        cleaned_solar_flat /= self.lamp_gain
        # Grab beam00, get lines for gain table creation and wavelength cal from it
        beam00 = cleaned_solar_flat[self.beam_edges[0, 0]:self.beam_edges[0, 1],
                 self.slit_edges[0, 0]:self.slit_edges[0, 1]]

        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens, slit_width=self.slit_width
        )

        avg_profile = np.nanmedian(
            beam00[
                int(beam00.shape[0]/2 - 30):int(beam00.shape[0]/2 + 30), :
            ], axis=0
        )

        self.firs_line_cores, self.fts_line_cores = self.fts_line_select(
            grating_params, avg_profile
        )

        self.combined_gain_table = np.ones(self.solar_flat.shape)
        self.combined_coarse_gain_table = np.ones(self.solar_flat.shape)

        for beam in self.beam_edges:
            for slit in self.slit_edges:
                image = cleaned_solar_flat[beam[0]:beam[1], slit[0]:slit[1]]
                gain, coarse, _ = spex.create_gaintables(
                    image,
                    [self.firs_line_cores[0] - 7, self.firs_line_cores[0] + 9],
                    neighborhood=12,
                    hairline_width=self.hairline_width / 2
                )
                self.combined_gain_table[beam[0]:beam[1], slit[0]:slit[1]] = gain
                self.combined_coarse_gain_table[beam[0]:beam[1], slit[0]:slit[1]] = coarse

        return

    def firs_save_gaintables(self, index: int) -> None:
        """
        Saves gaintables to FITS file in reduced directory.

        Parameters
        ----------
        index : int
        """
        # Check if file exists
        if os.path.exists(self.solar_gain_reduced[index]):
            if self.verbose:
                print("File exists: {0}\nSkipping File Write.".format(self.solar_gain_reduced[index]))
            return
        phdu = fits.PrimaryHDU()
        phdu.header['DATE'] = np.datetime64('now').astype(str)
        phdu.header['LC1'] = self.firs_line_cores[0]
        phdu.header['LC2'] = self.firs_line_cores[1]
        phdu.header['FTSLC1'] = self.fts_line_cores[0]
        phdu.header['FTSLC2'] = self.fts_line_cores[0]

        flat = fits.ImageHDU(self.solar_flat)
        flat.header['EXTNAME'] = 'SOLAR-FLAT'
        dark = fits.ImageHDU(self.solar_dark)
        dark.header['EXTNAME'] = 'SOLAR-DARK'
        cgain = fits.ImageHDU(self.combined_coarse_gain_table)
        cgain.header['EXTNAME'] = 'COARSE-GAIN'
        fgain = fits.ImageHDU(self.combined_gain_table)
        fgain.header['EXTNAME'] = 'GAIN'
        bedge = fits.ImageHDU(self.beam_edges)
        bedge.header['EXTNAME'] = 'BEAM-EDGES'
        hairs = fits.ImageHDU(self.hairlines)
        hairs.header['EXTNAME'] = 'HAIRLINES'
        slits = fits.ImageHDU(self.slit_edges)
        slits.header['EXTNAME'] = 'SLIT-EDGES'
        rotat = fits.ImageHDU(self.beam_rotation)
        rotat.header['EXTNAME'] = 'BEAM-ROTATION'
        shifts = fits.ImageHDU(self.beam_shifts)
        shifts.header['EXTNAME'] = 'BEAM-SHIFTS'

        hdul = fits.HDUList([phdu, flat, dark, cgain, fgain, bedge, hairs, slits, rotat, shifts])
        hdul.writeto(self.solar_gain_reduced[index], overwrite=True)

        return

    def firs_plot_gaintables(self, index: int) -> None:
        """
        Helper method to plot FIRS gaintables
        """
        aspect_ratio = self.solar_flat.shape[1]/self.solar_flat.shape[0]
        gain_fig = plt.figure("FIRS Gain Tables", figsize=(4 * 2.5, 2.5/aspect_ratio))
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
            vmin=np.nanmean(corr_flat) - 2*np.nanstd(corr_flat),
            vmax=np.nanmean(corr_flat) + 2*np.nanstd(corr_flat)
        )
        ax_coarse.imshow(
            self.combined_coarse_gain_table, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_fine.imshow(
            self.combined_gain_table, origin='lower', cmap='gray', vmin=0.5, vmax=2.5
        )
        ax_lamp.set_title('LAMP GAIN')
        ax_flat.set_title('SOLAR FLAT')
        ax_coarse.set_title('COARSE GAIN')
        ax_fine.set_title('FINE GAIN')
        for beam in self.beam_edges.flatten():
            ax_lamp.axhline(beam, c='C1', linewidth=1)
            ax_flat.axhline(beam, c='C1', linewidth=1)
            ax_coarse.axhline(beam, c='C1', linewidth=1)
            ax_fine.axhline(beam, c='C1', linewidth=1)
        for edge in self.slit_edges.flatten():
            ax_lamp.axvline(edge, c='C1', linewidth=1)
            ax_flat.axvline(edge, c='C1', linewidth=1)
            ax_coarse.axvline(edge, c='C1', linewidth=1)
            ax_fine.axvline(edge, c='C1', linewidth=1)
        for hair in self.hairlines.flatten():
            ax_lamp.axhline(hair, c='C2', linewidth=1)
            ax_flat.axhline(hair, c='C2', linewidth=1)
            ax_coarse.axhline(hair, c='C2', linewidth=1)
            ax_fine.axhline(hair, c='C2', linewidth=1)
        gain_fig.tight_layout()
        if self.save_figs:
            filename = os.path.join(self.final_dir, "gain_tables_{0}.png".format(index))
            gain_fig.savefig(filename, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)

        return

    def firs_get_lamp_gain(self) -> None:
        """
        Creates or loads the FIRS lamp gain.
        If there's no reduced lamp gain file, creates one from Level-0.
        If there's no dark file available for the lamp exposure time, kludges one from the solar dark.
        If there's no lamp flat series on that day, creates a dummy array of ones.
        """
        # Restore previously-created lamp flat
        if os.path.exists(self.lamp_gain_reduced):
            with fits.open(self.lamp_gain_reduced) as hdul:
                self.lamp_gain = hdul[0].data
        # Create new lamp gain and save to file:
        elif any(["LFLT" in i for i in self.obssum_info["OBSTYPE"]]):
            if any(["LDRK" in i for i in self.obssum_info['OBSTYPE']]):
                lamp_dark_index = ["LDRK" in i for i in self.obssum_info['OBSTYPE']].index(True)
                lamp_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_dark_index])
                lamp_flat_index = ["LFLT" in i for i in self.obssum_info['OBSTYPE']].index(True)
                lamp_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_flat_index])
            else:
                # Need to kludge a dark file together from the solar dark.
                # Grab the last solar dark taken in the day, make an average dark image
                solar_dark_index = ['DARK' in i for i in self.obssum_info['OBSTYPE'][::-1]].index(True)
                average_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][solar_dark_index])
                dark_exptime = (self.obssum_info['EXPTIME'][solar_dark_index] *
                                self.obssum_info['COADD'][solar_dark_index])
                # Need to create an average dark rate. Then we can get a sensible estimate of the dark current
                # at the lamp flat exposure time.
                dark_rate = average_dark / dark_exptime
                lamp_flat_index = ['LFLT' in i for i in self.obssum_info['OBSTYPE']].index(True)
                lamp_exptime = (self.obssum_info['EXPTIME'][lamp_flat_index] *
                                self.obssum_info['COADD'][lamp_flat_index])
                lamp_dark = dark_rate * lamp_exptime
                lamp_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_flat_index])
            cleaned_lamp_flat = self.clean_flat(lamp_flat - lamp_dark)
            self.lamp_gain = cleaned_lamp_flat / np.nanmedian(cleaned_lamp_flat)
            hdu = fits.PrimaryHDU(self.lamp_gain)
            hdu.header['DATE'] = np.datetime64("now").astype(str)
            hdu.header['COMMENT'] = "Created from series {0}".format(self.obssum_info['OBSSERIES'][lamp_flat_index])
            fits.HDUList([hdu]).writeto(self.lamp_gain_reduced, overwrite=True)
        else:
            self.lamp_gain = np.ones(self.solar_dark.shape)
            warnings.warn("No lamp flat available. Reduced data may show strong internal fringes.")
        return

    def firs_get_polcal(self) -> None:
        """
        Creates or loads a processed FIRS polcal
        """
        if os.path.exists(self.tx_matrix_reduced):
            with fits.open(self.tx_matrix_reduced) as hdul:
                self.input_stokes = hdul['STOKES-IN'].data
                self.calcurves = hdul['CALCURVES'].data
                self.txmat = hdul['TXMAT'].data
                self.txchi = hdul['TXCHI'].data
                self.txmat00 = hdul['TX00'].data
                self.txmatinv = hdul['TXMATINV'].data
        else:
            self.firs_polcal()
            self.save_polcal()
        if self.plot:
            self.plot_polcal()

        return

    def firs_polcal(self) -> None:
        """
        Performs polarization calibration of FIRS data
        """
        # Unlike IDL pipeline, we'll use both sets of polcals.
        polcal_indices = np.where(np.char.find(self.obssum_info['OBSTYPE'], 'PCAL') != -1)[0]
        polcal_files = []
        for index in polcal_indices:
            polcal_files += sorted(glob.glob(
                os.path.join(self.indir, self.obssum_info['OBSSERIES'][index]+"*")
            ))
        polcal_stokes_beams = np.zeros((
            len(polcal_files), # pcal stage
            self.nslits, # slit
            2, # beam
            4, # Stokes Vector
            np.minimum(self.rotated_beam_sizes[0]), # Y-range
            np.minimum(self.rotated_beam_sizes[1]) # x-range
        ))
        polarizer_angle = np.zeros(len(polcal_files))
        retarder_angle = np.zeros(len(polcal_files))
        llvl = np.zeros(len(polcal_files))
        azimuth = np.zeros(len(polcal_files))
        elevation = np.zeros(len(polcal_files))
        table_angle = np.zeros(len(polcal_files))
        for file, i in zip(polcal_files, tqdm.tqdm(range(len(polcal_files)), desc='Reading Polcal Files...')):
            with fits.open(file) as hdul:
                # Grab the angles we need to form the polcal model
                polarizer_angle[i] = hdul[0].header['PT4_POL']
                retarder_angle[i] = hdul[0].header['PT4_POL']
                llvl[i] = hdul[0].header['DST_LLVL']
                azimuth[i] = hdul[0].header['DST_AZ']
                elevation[i] = hdul[0].header['DST_EL']
                table_angle[i] = hdul[0].header['DST_TBL']
                # Dark/gain-correct and demodulate data
                dmod_data = self.demodulate_firs(
                    (hdul[0].data - self.solar_dark) / self.lamp_gain / self.combined_gain_table
                )
                # Cut out beams, shift, rotate, and clip to minimum size
                for j in range(self.slit_edges.shape[0]):
                    for k in range(self.beam_edges.shape[0]):
                        stokes_image = dmod_data[
                            :,
                            self.beam_edges[k, 0]:self.beam_edges[k, 1],
                            self.slit_edges[j, 0]:self.slit_edges[j, 0]
                        ]
                        rotated = scind.rotate(
                            stokes_image,
                            self.beam_rotation[k, j],
                            axes=(1, 2)
                        )
                        shifted = scind.shift(
                            rotated, (0, *self.beam_shifts[:, j, k])
                        )
                        min_shape = (np.minimum(self.rotated_beam_sizes[0]), np.minimum(self.rotated_beam_sizes[1]))
                        clipped = shifted[:, :min_shape[0], :min_shape[1]]
                        polcal_stokes_beams[i, j, k, :, :, :] = clipped
        merged_beams = np.zeros((
            polcal_stokes_beams.shape[0], # files
            polcal_stokes_beams.shape[1], # slits
            polcal_stokes_beams.shape[3], # stokes
            polcal_stokes_beams.shape[4], # y
            polcal_stokes_beams.shape[5] # x
        ))
        merged_beams[:, :, 0, :, :] = np.nanmean(polcal_stokes_beams[:, :, :, 0, :, :], axis=2)
        merged_beams[:, :, 1:, :, :] = (
                polcal_stokes_beams[:, :, 0, 1:, :, :] - polcal_stokes_beams[:, :, 1, 1:, :, :]
        ) / 2.
        # Mask values outside ilimit*mean(I)
        merged_beam_mask = (
            (merged_beams[:, :, 0, :, :] < self.ilimit[1] * np.nanmean(merged_beams[:, :, 0, :, :])) &
            (merged_beams[:, :, 0, :, :] > self.ilimit[0] * np.nanmean(merged_beams[:, :, 0, :, :]))
        )
        merged_beam_mask = np.repeat(merged_beam_mask[:, :, np.newaxis, :, :], 4, axis=2)
        self.calcurves = np.zeros((
            merged_beams.shape[0], # polcal stages
            merged_beams.shape[1], # slits
            merged_beams.shape[2], # stokes
            self.n_subslits # subslits
        ))
        subarrays = np.array_split(merged_beams, self.n_subslits, axis=3)
        submasks = np.array_split(merged_beam_mask, self.n_subslits, axis=3)

        for i in range(self.n_subslits):
            masked_subarray = subarrays[i]
            masked_subarray[~submasks[i]] = np.nan
            # Normalize QUV curve by I
            self.calcurves[:, :, 0, i] = np.nanmean(masked_subarray[:, :, 0, :, :], axis=(3, 4))
            self.calcurves[:, :, 1:, i] = np.nanmean(
                masked_subarray[:, :, 1:, :, :] / np.repeat(
                    masked_subarray[:, :, 0, :, :][:, :, np.newaxis, :, :], 3, axis=2
                ),
                axis=(3, 4)
            )
        # Catch any edge cases where an array was all NaN
        self.calcurves = np.nan_to_num(self.calcurves)
        # Create the input Stokes vectors from telescope matrix plus pt4 cal unit params
        # Cal train is Sky -> Telescope -> Lin. Polarizer -> Retarder -> Optical Train + Spectrograph
        input_stokes = np.zeros((self.calcurves.shape[0], 4))
        for i in range(self.calcurves.shape[0]):
            init_stokes = np.array([1, 0, 0, 0])
            tmtx = pol.get_dst_matrix(
                [azimuth[i], elevation[i], table_angle[i]],
                self.central_wavelength,
                self.t_matrix_file
            )
            init_stokes = tmtx @ init_stokes
            # Mult by 2 since we normalized intensities earlier...
            init_stokes = 2 * pol.linear_analyzer_polarizer(
                polarizer_angle * np.pi/180.,
                px=1,
                py=0.005 # estimate
            ) @ init_stokes

            init_stokes = pol.linear_retarder(
                retarder_angle[i] * np.pi/180,
                self.cal_retardance * np.pi/180
            ) @ init_stokes
            input_stokes[i, :] = init_stokes
        self.input_stokes = np.nan_to_num(input_stokes)
        self.txmat = np.zeros((self.n_subslits, self.nslits, 4, 4))
        self.txchi = np.zeros((self.n_subslits, self.nslits))
        self.txmat00 = np.zeros((self.n_subslits, self.nslits))
        self.txmatinv = np.zeros((self.n_subslits, self.nslits, 4, 4))

        for i in range(self.n_subslits):
            for j in range(self.nslits):
                errors, xmat = pol.matrix_inversion(
                    input_stokes,
                    self.calcurves[:, j, :, i]
                )
                self.txmat00[i, j] = xmat[0, 0]
                xmat /= xmat[0, 0]
                self.txmat[i, j] = xmat
                self.txmatinv[i, j] = np.linalg.inv(xmat)
                efficiencies = np.sqrt(np.sum(xmat ** 2, axis=1))
                #
                # Measurement of retardance from +-QU measurements
                # Since FIRS polcals are taken with polarizer at 0/45 degrees, there's no -Q, -U
                # +Q
                lp_pos_q_selection = (
                    ((np.abs(polarizer_angle) < 1) |
                     (np.abs(polarizer_angle - 180) < 1))
                )
                pos_q_vec = self.calcurves[:, j, :, i][np.repeat(
                    lp_pos_q_selection[:, np.newaxis], 4, axis=1
                )]
                pos_q_vec = np.nanmean(
                    pos_q_vec.reshape(int(pos_q_vec.shape[0] / 4), 4),
                    axis=0
                )
                stokes_pos_q = self.txmatinv[i, j] @ pos_q_vec
                stokes_pos_q = stokes_pos_q / stokes_pos_q[0]
                pos_qsqdiff = np.sum((stokes_pos_q - np.array([1, 1, 0, 0])) ** 2)
                # +U
                lp_pos_u_selection = (
                    ((np.abs(polarizer_angle - 45) < 1) |
                     (np.abs(polarizer_angle - 225) < 1))
                )
                pos_u_vec = self.calcurves[:, j, :, i][np.repeat(
                    lp_pos_u_selection[:, np.newaxis], 4, axis=1
                )]
                pos_u_vec = np.nanmean(
                    pos_u_vec.reshape(int(pos_u_vec.shape[0] / 4), 4),
                    axis=0
                )
                stokes_pos_u = self.txmatinv[i, j] @ pos_u_vec
                stokes_pos_u = stokes_pos_u / stokes_pos_u[0]
                pos_usqdiff = np.sum((stokes_pos_u - np.array([1, 0, 1, 0])) ** 2)

                self.txchi[i, j] = pos_qsqdiff + pos_usqdiff

                if self.verbose:
                    print("TX Matrix:")
                    print(xmat)
                    print("Inverse:")
                    print(self.txmatinv[i, j])
                    print("Efficiencies:")
                    print(
                        "Q: " + str(round(efficiencies[1], 4)),
                        "U: " + str(round(efficiencies[2], 4)),
                        "V: " + str(round(efficiencies[3], 4))
                    )
                    print("Average Deviation of cal vectors: ", np.sqrt(self.txchi[i, j]) / 4)
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
                        "WARNING: TX Matrix for slit {0} of {1}, section {2} of {3}"
                        "\nTX Matrix is an unphysical Mueller matrix!"
                        "\nOutput minimum I: \n{4}"
                        "\nOutput minimum I^2 - (Q^2 + U^2 + V^2): \n{5}".format(
                            j, self.nslits,
                            i, self.n_subslits,
                            mueller_check[1],
                            mueller_check[2]
                        )
                    )
        return

    def save_polcal(self) -> None:
        """
        Writes FITS file with FIRS polcal parameters
        """
        # Only write if the file doesn't already exist.
        if os.path.exists(self.tx_matrix_reduced):
            if self.verbose:
                print("File exists: {0}\nSkipping file write.".format(self.tx_matrix_reduced))
            return
        phdu = fits.PrimaryHDU()
        phdu.header['DATE'] = np.datetime64('now').astype(str)

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

    def plot_polcal(self) -> None:
        """
        Plots FIRS polcal input and output vectors
        """

        polcal_fig = plt.figure("Polcal Results", figsize=(8,4))
        # 3 columns, 4 rows
        # Column 1: Calcurves IQUV
        # Column 2: Input Stokes Vectors IQUV
        # Column 3: TXMAT @ Input Stokes Vectors IQUV
        gs = polcal_fig.add_gridspec(ncols=3, nrows=4)
        out_stokes = np.array([self.txmat @ self.input_stokes[j, :] for j in range(self.input_stokes.shape[0])])
        names = ['I', 'Q', 'U', 'V']
        # Increment colors for each subslit, increment linestyle for each slit
        # Max quad slit unit
        linestyles = ['-', ':', '--', '-.']
        for i in range(4):
            ax_ccurve = polcal_fig.add_subplot(gs[i, 0])
            ax_incurve = polcal_fig.add_subplot(gs[i, 1])
            ax_outcurve = polcal_fig.add_subplot(gs[i, 2])
            if i == 0:
                ax_ccurve.set_title("POLCAL CURVES")
                ax_incurve.set_title("INPUT VECTORS")
                ax_outcurve.set_title("FIT VECTORS")
            for j in range(self.n_subslits):
                for k in range(self.nslits):
                    ax_ccurve.plot(self.calcurves[:, k, i, j], c='C{0}'.format(j), linestyle=linestyles[k])
                    ax_outcurve.plot(out_stokes[:, j, k, i], c='C{0}'.format(j), linestyle=linestyles[k])
            ax_incurve.plot(self.input_stokes[:, i])
            # Clip to x range of [0, end]
            ax_ccurve.set_xlim(0, self.calcurves.shape[0])
            ax_incurve.set_xlim(0, self.calcurves.shape[0])
            ax_outcurve.set_xlim(0, self.calcurves.shape[0])
            # Clip to common y range defined by max/min of all 3 columns
            ymax = np.array(
                [self.calcurves[:, :, i, :].max(), out_stokes[:, :, :, i].max(), self.input_stokes[:, i].max()]
            ).max()
            ymin = np.array(
                [self.calcurves[:, :, i, :].min(), out_stokes[:, :, :, i].min(), self.input_stokes[:, i].min()]
            ).min()
            ax_ccurve.set_ylim(ymin, ymax)
            ax_incurve.set_ylim(ymin, ymax)
            ax_outcurve.set_ylim(ymin, ymax)

            ax_ccurve.set_ylabel(names[i])
            ax_ccurve.locator_params(axis="x", tight=True, nbins=4)
            ax_ccurve.locator_params(axis="y", tight=True, nbins=4)
            # Turn off tick labels except exterior plots
            ax_incurve.set_yticklabels([])
            ax_outcurve.set_yticklabels([])
            if i != 3:
                ax_ccurve.set_xticklabels([])
                ax_incurve.set_xticklabels([])
                ax_outcurve.set_xticklabels([])
        polcal_fig.tight_layout()
        if self.save_figs:
            filename = os.path.join(self.final_dir, "polcal_curves.png")
            polcal_fig.savefig(filename, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)

        return

    def reduce_firs_maps(
            self,
            index: int, overview: bool=False, write: bool=True, fringe_template: None or np.ndarray=None,
            v2q: bool or str=False, v2u: bool or str=False, q2v: bool or str=False, u2v: bool or str=False
    ) -> np.ndarray:
        """
        Main reduction loop. Performs the following corrections:
            ~Applies dark and gain corrections.
            ~Applies inverse telescope+spectrograph matrix.
            ~Corrects QU for parallactic angle.
            ~Performs I->QUV crosstalk estimation, and optional V<-->QU estimates
            ~Subtracts a fringe template


        Parameters
        ----------
        index : int
            Index in observing summary array to reduce
        overview : bool, optional
            Whether to create overview plots and construct parameter maps
        write : bool, optional
            Whether to write files to disk
        fringe_template : None or numpy.ndarray, optional
            If provided, uses an array of shape (4, nslits, ny, nx, nlambda) to subtract off polarimetric fringes
        v2q : bool or str, optional
            If true, determines single crosstalk value V->Q. If "full", determines profile-by-profile crosstalk
        v2u : bool or str, optional
            If true, determines single crosstalk value V->Q. If "full", determines profile-by-profile crosstalk
        q2v : bool or str, optional
            If true, determines single crosstalk value V->Q. If "full", determines profile-by-profile crosstalk
        u2v : bool or str, optional
            If true, determines single crosstalk value V->Q. If "full", determines profile-by-profile crosstalk

        Returns
        -------
        reduced_data : numpy.ndarray
            Reduced stokes vectors, shape (4, nslits, ny, nx, nlambda)
        """
        # FIRS maps taken with a repeat have the same file stem;
        # firs.2.YYYYMMDD.HHMMSS.XXXX.RRRR, where XXXX is the slit position, and RRRR is the iteration of map.
        nrepeat = self.obssum_info['NREPEAT'][index]
        for n in range(nrepeat):
            filelist = sorted(glob.glob(os.path.join(
                self.indir,
                self.obssum_info['OBSSERIES'][index] + ".*.{0:04d}".format(n) # Really should use regex for this...
            )))
            if self.verbose:
                print(
                    "Reducing Map {0} of {1} with {2} slit positions:"
                    "\nSeries: {3}".format(n+1, nrepeat, len(filelist), self.obssum_info['OBSSERIES'][index])
                )
            with fits.open(filelist[0]) as hdul:
                date, time = hdul[1].header['OBS_STAR'].split("T")
                date = date.replace("-", "")
                time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
                outname = self.reduced_file_pattern.format(
                    date, time, len(filelist)
                )
                outfile = os.path.join(self.final_dir, outname)
                if os.path.exists(outfile):
                    remake_file = input("File: {0}"
                                        "\nExists. (R)emake or (C)ontinue?  ".format(outname))
                    if ("c" in remake_file.lower()) or (remake_file.lower() == ""):
                        plt.pause(2)
                        plt.close('all')
                        reduced_data = self.read_reduced_data(outfile)
                        return reduced_data
                    elif ("r" in remake_file.lower()) and self.verbose:
                        print("Remaking file with current correction configuration. This may take some time.")

            reduced_data = np.zeros((
                4, self.nslits, np.minimum(self.rotated_beam_sizes[0]),
                len(filelist),
                np.minimum(self.rotated_beam_sizes[1])
            ))
            complete_i2quv_crosstalk = np.zeros((
                3, 2, self.nslits, np.minimum(self.rotated_beam_sizes[0]), len(filelist)
            ))
            complete_internal_crosstalks = np.zeros((
                4,
                self.nslits, np.minimum(self.rotated_beam_sizes[0]), len(filelist)
            ))
            xinv_interp = scinterp.CubicSpline(
                np.linspace(0, np.minimum(self.rotated_beam_sizes[0]), self.n_subslits),
                self.txmatinv,
                axis=0
            )(np.arange(0, np.minimum(self.rotated_beam_sizes[0])))
            step_ctr = 0
            with tqdm.tqdm(total=len(filelist), desc="Reducing Science Map") as pbar:
                for file in filelist:
                    with fits.open(file) as hdul:
                        iquv = self.demodulate_firs(
                            (hdul[0].data - self.solar_dark) / self.lamp_gain / self.combined_gain_table
                        )
                        science_beams = np.zeros((
                            2, 4, self.nslits,
                            np.minimum(self.rotated_beam_sizes[0]), np.minimum(self.rotated_beam_sizes[1])
                        ))
                        # Reference image for hairline/spectral line deskew. Don't do full gain correction,
                        # as this will result in hairline residuals.
                        ibeam = (np.mean(hdul[0].data, axis=0) - self.solar_dark) / self.lamp_gain
                        alignment_beams = np.zeros((
                            2, self.nslits,
                            np.minimum(self.rotated_beam_sizes[0]), np.minimum(self.rotated_beam_sizes[1])
                        ))
                        for i in range(2): # Beams
                            for j in range(self.nslits): # Slits
                                # Cut em out, rotate n shift em, clip em
                                science_beams[i, :, j, :, :] = scind.shift(
                                    scind.rotate(
                                        iquv[:,
                                            self.beam_edges[i, 0]:self.beam_edges[i, 1],
                                            self.slit_edges[j, 0]:self.slit_edges[j, 1]
                                        ], self.beam_rotation[i, j], axis=(1, 2)
                                    ), (0, *self.beam_shifts[:, j, i])
                                )[:, :np.minimum(self.rotated_beam_sizes[0]), :np.minimum(self.rotated_beam_sizes[1])]
                                alignment_beams[i, j, :, :] = scind.shift(
                                    scind.rotate(
                                        ibeam[
                                            self.beam_edges[i, 0]:self.beam_edges[i, 1],
                                            self.slit_edges[j, 0]:self.slit_edges[j, 1]
                                        ], self.beam_rotation[i, j], axis=(0, 1)
                                    ), self.beam_shifts[:, j, i]
                                )[:np.minimum(self.rotated_beam_sizes[0]), :np.minimum(self.rotated_beam_sizes[1])]
                        if step_ctr == 0:
                            hairline_skews, hairline_centers = self.subpixel_hairline_align(
                                alignment_beams, hair_centers=None
                            )
                            master_hairline_centers = (
                                hairline_centers[0, 0], # Slit 0, Beam 0 lower hairline
                                hairline_centers[0, 0] + np.diff(self.full_hairlines[0, 0]) # Based on original spacing.
                            )
                        else:
                            hairline_skews, hairline_centers = self.subpixel_hairline_align(
                                alignment_beams, hair_centers=hairline_centers
                            )
                        # Perform hairline deskew
                        for beam in range(science_beams.shape[0]):
                            for slit in range(science_beams.shape[2]):
                                for profile in range(science_beams.shape[4]):
                                    science_beams[beam, :, slit, :, profile] = scind.shift(
                                        science_beams[beam, :, slit, :, profile],
                                        (0, hairline_skews[beam, slit, profile]),
                                        mode='nearest', order=1
                                    )
                                # Perform bulk shift to 0th beam, 0th slit
                                if not beam == slit == 0:
                                    science_beams[beam, :, slit, :, :] = scind.shift(
                                        science_beams[beam, :, slit, :, :],
                                        (0, -(hairline_centers[beam, slit] - hairline_centers[0, 0]), 0),
                                        mode='nearest', order=1
                                    )
                                # Perform bulk shift to 0th beam, 0th slit, 0th step
                                science_beams[beam, :, slit, :, :] = scind.shift(
                                    science_beams[beam, :, slit, :, :],
                                    (0, -(hairline_centers[beam, slit] - master_hairline_centers[0, 0]), 0),
                                    mode='nearest', order=1
                                )
                        # Perform spectral deskew and registration
                        science_beams, spectral_centers = self.subpixel_spectral_align(science_beams, hairline_centers)


    def subpixel_hairline_align(
            self,
            alignment_image: np.ndarray, hair_centers: None or np.ndarray=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs subpixel hairline alignment of two beams into a single image, and returns the necessary translations.

        Parameters
        ----------
        alignment_image : numpy.ndarray
            Minimally processed image cutouts. Shape (2, nslits, ny, nx)
        hair_centers : None or numpy.ndarray
            If given, should be of shape (2, nslits) containing the y-location of the lower hairline.
            If None, uses ssosoft.spectral.spectraTools.detect_beams_hairlines to get a first-order estimate.

        Returns
        -------
        hairline_skews : numpy.ndarray
            Array of shape (2, nslits, nx) containing subpixel shifts for hairline registration
        hairline_centers : numpy.ndarray
            Array of shape (2, nslits) containing the y-value of the hairline center
        """
        if hair_centers is None:
            spex.detect_beams_hairlines.num_calls = 0
            hair_centers = np.zeros((alignment_image.shape[0], alignment_image.shape[1]))
            for i in range(alignment_image.shape[0]):
                for j in range(alignment_image.shape[1]):
                    _, _, tmp_hairlines = spex.detect_beams_hairlines(
                        alignment_image[i, j, :, :], threshold=self.beam_threshold, hairline_width=self.hairline_width,
                        expected_hairlines=2, expected_slits=1, expected_beams=1, fallback=False # Too many popups.
                    )
                    hair_centers[i, j] = tmp_hairlines.min()
        hairline_skews = np.zeros((alignment_image.shape[0], alignment_image.shape[1], alignment_image.shape[3]))
        hairline_centers = np.zeros((alignment_image.shape[0], alignment_image.shape[1]))
        for i in range(alignment_image.shape[0]):
            for j in range(alignment_image.shape[1]):
                deskewed_image = np.zeros(alignment_image.shape[2:])
                hairline_minimum = hair_centers[i, j] - 6
                hairline_maximum = hair_centers[i, j] + 6
                if hairline_minimum < 0:
                    hairline_minimum = 0
                if hairline_maximum > alignment_image.shape[2]:
                    hairline_maximum = alignment_image.shape[2] - 1
                medfilt_hairline_image = scind.median_filter(
                    alignment_image[i, j, hairline_minimum:hairline_maximum, :],
                    size=(2, 25)
                )
                hairline_skews[i, j, :] = spex.spectral_skew(
                    np.rot90(medfilt_hairline_image), order=1, slit_reference=0.5
                )
                for k in range(hairline_skews.shape[2]):
                    deskewed_image[:, k] = scind.shift(
                        alignment_image[i, j, :, k], hairline_skews[i, j, k],
                        mode='nearest', order=1
                    )
                hairline_centers[i, j] = spex.find_line_core(
                    np.nanmedian(deskewed_image[hairline_minimum:hairline_maximum, 50:-50], axis=1)
                ) + hairline_minimum
        return hairline_skews, hairline_centers

    def subpixel_spectral_align(
            self, cutout_beams: np.ndarray, hairline_centers: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Performs iterative deskew and align along the spectral axis.
        Returns the aligned beam and spectral line center for master registration.

        Parameters
        ----------
        cutout_beams
        hairline_centers

        Returns
        -------

        """
        return

    @staticmethod
    def read_reduced_data(filename: str) -> np.ndarray:
        """
        Reads reduced data file Stokes vectors into memory

        Parameters
        ----------
        filename : str

        Returns
        -------
        reduced_data : numpy.ndarray
            Array of shape (4, nslits, ny, nx, nlambda)
        """
        with fits.open(filename) as hdul:
            reduced_data = np.zeros((4, *hdul['STOKES-I'].data.shape))
            for i in range(1, 5):
                reduced_data[i-1] = hdul[i].data
        return reduced_data

    def clean_flat(self, flat_image: np.ndarray) -> np.ndarray:
        """
        Cleans flat image by finding hairlines and removing them via scipy.interpolate.griddata
        Copied from ssosoft.spectral.spinorCal.SpinorCal.

        Parameters
        ----------
        flat_image : numpy.ndarray
            Dark-subtracted flat image

        Returns
        -------
        cleaned_flat_image : numpy.ndarray
            Flat with hairlines removed

        """

        _, _, hairlines = spex.detect_beams_hairlines(
            flat_image,
            threshold=self.beam_threshold, hairline_width=self.hairline_width,
            expected_hairlines=self.nhair, expected_beams=2, expected_slits=self.nslits,
            fallback=True  # Hate relying on it, but safer for now
        )
        # Reset recursive counter since we'll need to use the function again later
        spex.detect_beams_hairlines.num_calls = 0
        for line in hairlines:
            # range + 1 to compensate for casting a float to an int, plus an extra 2-wide pad for edge effects
            flat_image[int(line - self.hairline_width - 2):int(line + self.hairline_width + 3), :] = np.nan
        x = np.arange(0, flat_image.shape[1])
        y = np.arange(0, flat_image.shape[0])
        masked_flat = np.ma.masked_invalid(flat_image)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~masked_flat.mask]
        y1 = yy[~masked_flat.mask]
        new_flat = masked_flat[~masked_flat.mask]
        cleaned_flat_image = scinterp.griddata(
            (x1, y1),
            new_flat.ravel(),
            (xx, yy),
            method='nearest',
            fill_value=np.nanmedian(flat_image)
        )
        return cleaned_flat_image

    def average_image_from_list(self, obsseries_stem: str) -> np.ndarray:
        """
        Returns an average dark from a list of FIRS files
        Parameters
        ----------
        obsseries_stem : str
            Filename stem to grab a list of

        Returns
        -------
        average_image : numpy.ndarray
            Average image
        """
        file_list = sorted(glob.glob(os.path.join(self.indir, obsseries_stem + "*")))
        with fits.open(file_list[0]) as hdul:
            average_image = np.zeros(hdul[0].data.shape[1:])
            increment = hdul[0].data.shape[0]
            counter = 0
        for file in file_list:
            with fits.open(file) as hdul:
                average_image += np.sum(hdul[0].data, axis=0)
                counter += increment
        average_image /= counter
        return average_image


    def firs_parse_configfile(self) -> None:
        """Parses config file, sets class variables"""

        return

    def firs_parse_directory(self) -> None:
        """
        Organizes files in the Level-0 FIRS directory, and places in a numpy recarray.
        Firs stores one exposure per file,
        """
        # Check if obssum file exists first, and load it into memory if it does
        # This allows the user to override the obstype from the obssum file this
        # code creates if necessary.
        # i.e., the user can run this function once, generate the obssum, check it,
        # then, if there are still mislabelled obs series that the code didn't catch,
        # or if my catch statements caught something unintentional, the user can edit
        # the obssum file with the correct values and the code will accept the new
        # interpretation without the need to edit the original files.
        obssum_files = sorted(glob.glob(os.path.join(self.final_dir, "firs_obssum_*.txt")))
        if len(obssum_files) > 0:
            if self.verbose:
                print("Loading observing summary file: {0}".format(os.path.split(obssum_files[0])[1]))
            self.obssum_info = np.genfromtxt(
                obssum_files[0],
                names=True,
                dtype=[
                    "U30", "datetime64[ms]", "datetime64[ms]", int, int, int, int, "U4"
                ]
            )
            return

        filelist = sorted(glob.glob(
            os.path.join(self.indir, "firs.2.*")
        ))
        # Grab unique file series. FIRS IR files have the name pattern:
        #   firs.2.YYYYMMDD.HHMMSS.XXXX.YYYY
        # Where XXXX is the step number, and YYYY is the loop number.
        # So a 500-step map, acquired with the instrument set to take 2 maps back-to-back
        # will have XXXX in the range [0 -> 499] and YYYY [0 or 1]
        obs_series = []
        for file in filelist:
            obs_series.append(".".join(os.path.split(file)[1].split(".")[:-2]))
        obs_series = list(set(obs_series))
        starttime = []
        endtime = []
        exptime = []
        coadds = []
        repeats = []
        obstype = []
        nfiles = []
        for series in obs_series:
            series_list = sorted(glob.glob(
                os.path.join(self.indir, series + "*")
            ))
            nfiles.append(len(series_list))
            with fits.open(series_list[0]) as hdul:
                starttime.append(np.datetime64(hdul[0].header['OBS_STAR']))
                exptime.append(hdul[0].header['EXP_TIME'])
                coadds.append(hdul[0].header['SUMS'])
                repeats.append(hdul[0].header['NOREPEAT'])
                # Obstype is stored by observers in a comment card of the FITS file
                obstype.append(str(hdul[0].header['COMMENT']))
            with fits.open(series_list[-1]) as hdul:
                # OBS_END *is* present in the headers, but it's fixed to the same as OBS_STAR
                # We'll get an approximation by taking the frame starttime and adding the total exposure time.
                endtime.append(
                    np.datetime64(hdul[0].header['OBS_STAR']) + np.timedelta64(
                        hdul[0].header['SUMS'] * hdul[0].header['EXP_TIME'], "ms"
                    )
                )
        # Oftentimes, the observers forget to change the comment card that is the only way for FIRS
        # to tell what its file type is. Minimally, we need a flat, a dark, a polcal, and science maps.
        # Flats typically have 32 repeats, darks 16, polcals 36, science maps can be anything, but are
        # typically longer than 36. Lamp flats and lamp darks have longer exposure times.
        # We'll go back through the list and excise anything we don't need, and re-classify files as needed.
        final_obsseries = []
        final_starttime = []
        final_endtime = []
        final_exptime = []
        final_coadd = []
        final_repeats = []
        final_obstype = []
        final_nfiles = []
        for i in range(len(obs_series)):
            # Case, file tagged as cal with an unusable number of exposures. Cut from series.
            if "scan" not in obstype[i].lower() and nfiles[i] < 16:
                # Most likely a cal sequence cut short.
                continue
            # Case, file tagged as pcal with < 36 exposures. Cut.
            elif "pcal" in obstype[i].lower() and nfiles[i] < 36:
                continue
            # Case, file tagged as sflt with < 32 exposures. Cut.
            elif "sflt" in obstype[i].lower() and nfiles[i] < 32:
                continue
            # Dark series missing files is caught by first if statement.
            # Check for mislabelled files
            elif nfiles[i] == 32 and not any([obstype[i].lower() in ot for ot in ['sflt', 'scan', 'pcal', 'lflt']]):
                # If it's got 32 exposures and it isn't a sflt, lflt, scan, or pcal, it's probably a mislabeled sflt
                final_obsseries.append(obs_series[i])
                final_starttime.append(starttime[i])
                final_endtime.append(endtime[i])
                final_exptime.append(exptime[i])
                final_coadd.append(coadds[i])
                final_repeats.append(repeats[i])
                final_obstype.append("SFLT")
                final_nfiles.append(nfiles[i])
            elif "dark" in obstype[i].lower() and nfiles[i] > 16:
                final_obsseries.append(obs_series[i])
                final_starttime.append(starttime[i])
                final_endtime.append(endtime[i])
                final_exptime.append(exptime[i])
                final_coadd.append(coadds[i])
                final_repeats.append(repeats[i])
                final_obstype.append("SCAN")
                final_nfiles.append(nfiles[i])
            elif "dark" in obstype[i].lower() and exptime[i] >= 1000:
                # Lamp dark
                final_obsseries.append(obs_series[i])
                final_starttime.append(starttime[i])
                final_endtime.append(endtime[i])
                final_exptime.append(exptime[i])
                final_coadd.append(coadds[i])
                final_repeats.append(repeats[i])
                final_obstype.append("LDRK")
                final_nfiles.append(nfiles[i])
            else:
                final_obsseries.append(obs_series[i])
                final_starttime.append(starttime[i])
                final_endtime.append(endtime[i])
                final_exptime.append(exptime[i])
                final_coadd.append(coadds[i])
                final_repeats.append(repeats[i])
                final_obstype.append(obstype[i].upper())
                final_nfiles.append(nfiles[i])
        names = [
            "OBSSERIES", "STARTTIME", "ENDTIME", "EXPTIME", "COADD", "NREPEAT", "NFILES", "OBSTYPE"
        ]
        self.obssum_info = np.rec.fromarrays(
            [
                np.array(final_obsseries), np.array(final_starttime), np.array(final_endtime),
                np.array(final_exptime), np.array(final_coadd), np.array(final_repeats),
                np.array(final_nfiles), np.array(final_obstype)
            ],
            names=names
        )

        date = self.obssum_info["STARTTIME"][0].astype("datetime64[D]").astype(str).replace("-", "")
        obssum_path = os.path.join(self.final_dir, "firs_obssum_{0}.txt".format(date))
        with open(obssum_path, "w") as file:
            file.write(
                "{0:<22}\t{1:<23}\t{2:<23}\t{3:<7}\t{4:<5}\t{5:<7}\t{6:<6}\t{7:<7}\n".format(
                    *names
                )
            )
            for i in range(self.obssum_info.shape[0]):
                file.write(
                    '{0:<22}\t{1:<23}\t{2:<23}\t{3:<7}\t{4:<5}\t{5:<7}\t{6:<6}\t{7:<7}\n'.format(
                        *self.obssum_info[i]
                    )
                )
        if self.verbose:
            print("Saved observing summary file: {0}".format(os.path.split(obssum_path)[1]))
        return


    def demodulate_firs(self, poldata: np.ndarray) -> np.ndarray:
        """
        Applies demodulation, and returns 4-array of IQUV

        Parameters
        ----------
        poldata : numpy.ndarray
            3D array of polarization data. Should have shape (8, ny, nx)

        Returns
        -------
        stokes : numpy.ndarray
            3D array of stokes data, of shape (4, ny, nx)
        """
        stokes = np.zeros((4, *poldata.shape[1:]))
        # Despike...
        if self.despike:
            poldata = spinorCal.SpinorCal.despike_image(
                poldata, footprint=self.despike_footprint
            )
        # Unlike SPINOR, FIRS is redundant between 0:4, 4:8. Combine.
        poldata = (poldata[:4] + poldata[4:]) / 2.
        for i in range(stokes.shape[0]):
            for j in range(poldata.shape[0]):
                stokes[i] += self.pol_demod[i, j] * poldata[j, :, :]
            stokes[i] *= self.pol_norm[i]
        return stokes

    def fts_line_select(self, grating_params: np.rec.recarray, average_profile: np.ndarray) -> tuple[list, list]:
        """
        Pops up the line selection widget for gain table creation and wavelength grid determination

        Parameters
        ----------
        grating_params : numpy.records.recarray
            From spectraTools.grating_calculations; basic spctrograph configuration info
        average_profile : numpy.ndarray
            Average spectral profile from a cleaned and corrected flat field image

        Returns
        -------
        firs_line_cores : list
            List of Fourier phase-determined line cores from average_profile and user selected spectral lines
        fts_line_cores : list
            List of Fourier phase-determined line cores from the FTS atlas user-selected spectral lines

        """

        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        apx_wavemin = self.central_wavelength - np.nanmean(self.slit_edges[0]) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax = self.central_wavelength + np.nanmean(self.slit_edges[0]) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemin -= 30 * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax += 30 * grating_params['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apx_wavemin, apx_wavemax)

        print("Top: FIRS Spectrum. Bottom: FTS Reference Spectrum")
        print("Select the same two spectral lines on each plot.")
        firs_lines, fts_lines = spex.select_lines_doublepanel(
            average_profile,
            fts_spec,
            4
        )

        firs_line_cores = [
            int(spex.find_line_core(average_profile[x - 5:x + 5]) + x - 5) for x in firs_lines
        ]
        fts_line_cores = [
            spex.find_line_core(fts_spec[x - 20:x + 9]) + x - 20 for x in fts_lines
        ]

        return firs_line_cores, fts_line_cores




