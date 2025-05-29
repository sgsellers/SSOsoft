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
import scipy.ndimage as scind
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
        self.slit_spacing = 0
        self.beam_threshold = 0.5
        self.hairline_width = 3
        self.grating_rules = 31.6  # lpmm
        self.blaze_angle = 63.5
        self.grating_angle = 63.5 - 3.75 # Default 10830/15648 value
        # Usually, FIRS is operated at 10830, but the code should be able to handle 15648
        self.central_wavelength = 10830
        self.internal_crosstalk_line = 10827.089 # Line to use for V<->QU crosstalk determination.
        self.crosstalk_range = 2 # Angstroms, range around self.internal_crosstalk_line
        # 52nd order for 10830, 36th order for 15648
        self.spectral_order = 52
        self.pixel_size = 20 # um -- for the Virgo 1k, probably.
        self.slit_width = 40  # um -- needs to be changed if slit unit is swapped out.
        self.n_subslits = 10
        self.fringe_frequency = [-0.4, 0.4] # In angstrom, the assumed threshold for high-frequency noise to filter out
        self.verbose = False
        self.spectral_transmission = False
        self.v2q = True
        self.v2u = True
        self.q2v = False
        self.u2v = False
        self.despike = False
        self.despike_footprint = (1, 5, 1)
        self.plot = False
        self.save_figs = False
        self.crosstalk_continuum = None
        self.analysis_ranges = "default" # Default or choose. Choose pops up a widget.
        self.analysis_indices = []
        # From NIST, laboratory reference line centers for Si I, He I
        self.default_reference_wavelengths = np.array([10827.089, 10829.09115, 10830.30989])
        self.default_analysis_ranges = np.array(
            [self.default_reference_wavelengths - 1.25, self.default_reference_wavelengths + 1.25]
        )
        self.default_analysis_names = ["SiI 10827", "HeI 10829", "HeI 10830"]
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
        for index in range(len(self.science_series_list)):
            self.__init__(self.config_file)
            self.firs_configure_run()
            self.science_series_list = self.obssum_info['OBSSERIES'][np.where(
                np.char.find(self.obssum_info['OBSTYPE'], "SCAN") != -1
            )[0]]
            self.firs_get_cal_images(index)
            if self.plot:
                # Need to clear out previous plot instances
                plt.pause(2)
                plt.close("all")
            fringe_template = None
            if self.defringe == "flat":
                fringe_template = self.firs_get_fringe_template(index)

            obs_index = np.where(self.obssum_info["OBSSERIES"] == self.science_series_list[index])[0][0]
            _ = self.reduce_firs_maps(
                obs_index, overview=self.plot, write=True, fringe_template=fringe_template,
                v2q=self.v2q, v2u=self.v2u, q2v=self.q2v, u2v=self.u2v
            )
        return

    def firs_configure_run(self) -> None:
        """
        Reads config file and parses input directory
        """
        self.firs_parse_configfile()
        self.firs_parse_directory()
        self.firs_organize_directory()

        return

    def firs_organize_directory(self):
        """
        Matches cal files and sets up code expectation
        """
        solar_flat_start_times = self.obssum_info['STARTTIME'][self.obssum_info['OBSTYPE'] == 'SFLT']
        science_start_times = self.obssum_info['STARTTIME'][self.obssum_info['OBSTYPE'] == 'SCAN']
        nearest_flat_indices = [
            spex.find_nearest(solar_flat_start_times, x) for x in science_start_times
        ]
        self.solar_gain_reduced = [
            os.path.join(
                self.final_dir,
                "FIRS_{0}_SOLARGAIN.fits"
            ).format(x) for x in nearest_flat_indices
        ]
        self.lamp_gain_reduced = os.path.join(
            self.final_dir,
            "FIRS_LAMPGAIN.fits"
        )
        self.tx_matrix_reduced = os.path.join(
            self.final_dir,
            "FIRS_POLCAL.fits"
        )

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

    def firs_get_fringe_template(self, index: int):
        """
        Loads or creates fringe template from reduced flat field
        """
        solar_flat_start_times = self.obssum_info['STARTTIME'][self.obssum_info['OBSTYPE'] == 'SFLT']
        obs_index = np.where(self.science_series_list[index] == self.obssum_info["OBSSERIES"])[0][0]
        science_start_time = self.obssum_info['STARTTIME'][obs_index]
        nearest_flat_index = spex.find_nearest(solar_flat_start_times, science_start_time)
        sflat_index = np.where(self.obssum_info["STARTTIME"] == solar_flat_start_times[nearest_flat_index])[0][0]
        sflat_start = self.obssum_info['STARTTIME'][sflat_index]
        date, time = str(sflat_start).split("T")
        date = date.replace("-", "")
        time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
        fringe_template_filename = "FIRS_FRINGETEMPLATE_{0}_{1}.fits".format(
            date, time
        )
        fringe_template_path = os.path.join(self.final_dir, fringe_template_filename)
        if os.path.exists(fringe_template_path):
            with fits.open(fringe_template_path) as hdul:
                fringe_template = hdul[0].data
        else:
            reduced_flat = self.reduce_firs_maps(
                sflat_index, write=False, overview=False, fringe_template=None
            )
            fringe_template = self.construct_fringe_template_from_flat(reduced_flat)
            hdu = fits.PrimaryHDU(fringe_template)
            hdul = fits.HDUList(hdu)
            hdul.writeto(fringe_template_filename, overwrite=True)
        return fringe_template

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
            obs_index = np.where(self.obssum_info["OBSSERIES"] == self.science_series_list[index])[0][0]
            obs_start = self.obssum_info['STARTTIME'][obs_index]
            sflat_index = sflt_indices[spex.find_nearest(sflt_starts, obs_start)]
            self.solar_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][sflat_index])
            dark_indices = np.where(['DARK' in i for i in self.obssum_info['OBSTYPE']])[0]
            dark_index = spex.find_nearest(dark_indices, sflat_index)
            self.solar_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][dark_indices[dark_index]])
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
                self.beam_shifts = hdul['BEAM-SHIFTS'].data
                self.rotated_beam_sizes = hdul['BEAM-SIZES'].data
                self.full_hairlines = hdul['FULL-HAIR'].data
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
                for hair, k in zip(self.hairlines[i] - beam[0], range(int(self.nhair/2))):
                    hair_image = np.rot90(image[int(hair-5):int(hair+7), :])
                    # While we've got it, grab the hairline center from this image
                    hairline_center = spex.find_line_core(
                        hair_image[:, int(hair_image.shape[1]/2 - 20):int(hair_image.shape[1]/2 + 20)].mean(axis=1)
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
                        (np.nanmean(master_clipped, axis=0) - np.nanmean(master_clipped)),
                        (np.nanmean(rotate_clipped, axis=0) - np.nanmean(rotate_clipped)),
                        mode='full'
                    ).argmax() - master_clipped.shape[1]
                    yshift = np.correlate(
                        (np.nanmean(master_clipped, axis=1) - np.nanmean(master_clipped)),
                        (np.nanmean(rotate_clipped, axis=1) - np.nanmean(rotate_clipped)),
                        mode='full'
                    ).argmax() - master_clipped.shape[0]
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
                image = np.nan_to_num(image, nan=np.nanmedian(image))
                image[image == 0] = np.nanmedian(image)
                gain, coarse, _ = spex.create_gaintables(
                    image,
                    [self.firs_line_cores[0] - 7, self.firs_line_cores[0] + 9],
                    neighborhood=24,
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
        phdu.header['FTSLC2'] = self.fts_line_cores[1]

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
        bsizes = fits.ImageHDU(self.rotated_beam_sizes)
        bsizes.header['EXTNAME'] = 'BEAM-SIZES'
        fhair = fits.ImageHDU(self.full_hairlines)
        fhair.header['EXTNAME'] = 'FULL-HAIR'

        hdul = fits.HDUList([phdu, flat, dark, cgain, fgain, bedge, hairs, slits, rotat, shifts, bsizes, fhair])
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
                lamp_dark_index = np.where(self.obssum_info['OBSTYPE'] == 'LDRK')[0][0]
                lamp_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_dark_index])
                lamp_flat_index = np.where(self.obssum_info['OBSTYPE'] == 'LFLT')[0][0]
                lamp_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_flat_index])
            else:
                # Need to kludge a dark file together from the solar dark.
                # Grab the last solar dark taken in the day, make an average dark image
                solar_dark_index = np.where(self.obssum_info['OBSTYPE'] == 'DARK')[0][0]
                average_dark = self.average_image_from_list(self.obssum_info['OBSSERIES'][solar_dark_index])
                dark_exptime = (self.obssum_info['EXPTIME'][solar_dark_index] *
                                self.obssum_info['COADD'][solar_dark_index])
                # Need to create an average dark rate. Then we can get a sensible estimate of the dark current
                # at the lamp flat exposure time.
                dark_rate = average_dark / dark_exptime
                lamp_flat_index = np.where(self.obssum_info['OBSTYPE'] == 'LFLT')[0][0]
                lamp_exptime = (self.obssum_info['EXPTIME'][lamp_flat_index] *
                                self.obssum_info['COADD'][lamp_flat_index])
                lamp_dark = dark_rate * lamp_exptime
                lamp_flat = self.average_image_from_list(self.obssum_info['OBSSERIES'][lamp_flat_index])
            cleaned_lamp_flat = self.clean_flat(lamp_flat - lamp_dark)
            lamp_gain = cleaned_lamp_flat / np.nanmedian(cleaned_lamp_flat)
            lamp_gain[lamp_gain == 0] = 1
            self.lamp_gain = lamp_gain
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
        self.polcal_file_list = polcal_files
        polcal_stokes_beams = np.zeros((
            len(polcal_files), # pcal stage
            self.nslits, # slit
            2, # beam
            4, # Stokes Vector
            int(np.amin(self.rotated_beam_sizes[0])), # Y-range
            int(np.amin(self.rotated_beam_sizes[1])) # x-range
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
                retarder_angle[i] = hdul[0].header['PT4_RET']
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
                            self.slit_edges[j, 0]:self.slit_edges[j, 1]
                        ]
                        rotated = scind.rotate(
                            stokes_image,
                            self.beam_rotation[k, j],
                            axes=(1, 2)
                        )
                        shifted = scind.shift(
                            rotated, (0, *self.beam_shifts[:, j, k])
                        )
                        min_shape = (int(np.amin(self.rotated_beam_sizes[0])), int(np.amin(self.rotated_beam_sizes[1])))
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
            self.calcurves[:, :, 0, i] = np.nanmean(masked_subarray[:, :, 0, :, :], axis=(2, 3))
            self.calcurves[:, :, 1:, i] = np.nanmean(
                masked_subarray[:, :, 1:, :, :] / np.repeat(
                    masked_subarray[:, :, 0, :, :][:, :, np.newaxis, :, :], 3, axis=2
                ),
                axis=(3, 4)
            )
            # Normalize I curve
            self.calcurves[:, :, 0, i] /= np.repeat(
                np.nanmean(self.calcurves[:, :, 0, i], axis=0)[np.newaxis, :], self.calcurves.shape[0], axis=0
            )
        # Catch any edge cases where an array was all NaN
        self.calcurves = np.nan_to_num(self.calcurves)
        # Create the input Stokes vectors from telescope matrix plus pt4 cal unit params
        # Cal train is Sky -> Telescope -> Lin. Polarizer -> Retarder -> Optical Train + Spectrograph
        input_stokes = np.zeros((self.calcurves.shape[0], 4))
        for i in range(self.calcurves.shape[0]):
            init_stokes = np.array([1, 0, 0, 0])
            init_stokes = pol.linear_analyzer_polarizer(
                polarizer_angle[i] * np.pi/180.,
                px=1,
                py=0.005 # estimate
            ) @ init_stokes

            init_stokes = pol.linear_retarder(
                retarder_angle[i] * np.pi/180,
                self.cal_retardance * np.pi/180
            ) @ init_stokes
            input_stokes[i, :] = init_stokes / init_stokes[0]
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
                efficiencies = pol.muller_efficiencies(xmat)
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
                        "I: " + str(round(efficiencies[0], 4)),
                        "Q: " + str(round(efficiencies[1], 4)),
                        "U: " + str(round(efficiencies[2], 4)),
                        "V: " + str(round(efficiencies[3], 4)),
                        "QUV: " + str(round(efficiencies[4], 4))
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
        reduced_data = []
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
                date, time = hdul[0].header['OBS_STAR'].split("T")
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
            self.rotated_beam_sizes = self.rotated_beam_sizes.astype(int)
            reduced_data = np.zeros((
                4, self.nslits, np.amin(self.rotated_beam_sizes[0]),
                len(filelist),
                np.amin(self.rotated_beam_sizes[1])
            ))
            complete_i2quv_crosstalk = np.zeros((
                3, 2, self.nslits, np.amin(self.rotated_beam_sizes[0]), len(filelist)
            ))
            complete_internal_crosstalks = np.zeros((
                4,
                self.nslits, np.amin(self.rotated_beam_sizes[0]), len(filelist)
            ))
            xinv_interp = scinterp.CubicSpline(
                np.linspace(0, np.amin(self.rotated_beam_sizes[0]), self.n_subslits),
                self.txmatinv,
                axis=0
            )(np.arange(0, np.amin(self.rotated_beam_sizes[0])))
            step_ctr = 0
            with tqdm.tqdm(total=len(filelist), desc="Reducing Science Map") as pbar:
                for file in filelist:
                    with fits.open(file) as hdul:
                        iquv = self.demodulate_firs(
                            (hdul[0].data - self.solar_dark) / self.lamp_gain / self.combined_gain_table
                        )
                        science_beams = np.zeros((
                            2, 4, self.nslits,
                            np.amin(self.rotated_beam_sizes[0]), np.amin(self.rotated_beam_sizes[1])
                        ))
                        # Reference image for hairline/spectral line deskew. Don't do full gain correction,
                        # as this will result in hairline residuals.
                        ibeam = (np.mean(hdul[0].data, axis=0) - self.solar_dark) / self.lamp_gain
                        alignment_beams = np.zeros((
                            2, self.nslits,
                            np.amin(self.rotated_beam_sizes[0]), np.amin(self.rotated_beam_sizes[1])
                        ))
                        for i in range(2): # Beams
                            for j in range(self.nslits): # Slits
                                # Cut em out, rotate n shift em, clip em
                                science_beams[i, :, j, :, :] = scind.shift(
                                    scind.rotate(
                                        iquv[:,
                                            self.beam_edges[i, 0]:self.beam_edges[i, 1],
                                            self.slit_edges[j, 0]:self.slit_edges[j, 1]
                                        ], self.beam_rotation[i, j], axes=(1, 2)
                                    ), (0, *self.beam_shifts[:, j, i])
                                )[:, :np.amin(self.rotated_beam_sizes[0]), :np.amin(self.rotated_beam_sizes[1])]
                                alignment_beams[i, j, :, :] = scind.shift(
                                    scind.rotate(
                                        ibeam[
                                            self.beam_edges[i, 0]:self.beam_edges[i, 1],
                                            self.slit_edges[j, 0]:self.slit_edges[j, 1]
                                        ], self.beam_rotation[i, j], axes=(0, 1)
                                    ), self.beam_shifts[:, j, i]
                                )[:np.amin(self.rotated_beam_sizes[0]), :np.amin(self.rotated_beam_sizes[1])]
                        if step_ctr == 0:
                            hairline_skews, hairline_centers = self.subpixel_hairline_align(
                                alignment_beams, hair_centers=None
                            )
                            master_hairline_centers = (
                                hairline_centers[0, 0], # Slit 0, Beam 0 lower hairline
                                hairline_centers[0, 0] + np.diff(self.full_hairlines[0, 0])[0] # Based on original spacing.
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
                                    (0, -(hairline_centers[beam, slit] - master_hairline_centers[0]), 0),
                                    mode='nearest', order=1
                                )
                        # Perform spectral deskew and registration
                        science_beams, spectral_centers = self.subpixel_spectral_align(science_beams, hairline_centers)
                        if step_ctr == 0:
                            master_spectral_center = spectral_centers[0, 0]
                        # Bulk spectral alignment on deskewed beams to 0th slit, beam, step
                        for beam in range(science_beams.shape[0]):
                            for slit in range(science_beams.shape[2]):
                                science_beams[beam, :, slit, :, :] = scind.shift(
                                    science_beams[beam, :, slit, :, :],
                                    (0, 0, -(spectral_centers[beam, slit] - master_spectral_center))
                                )
                        norm_factor = np.mean(
                            science_beams[0, 0, :, 50:-50, :], axis=(1, 2)
                        ) / np.mean(science_beams[1, 0, :, 50:-50, :], axis=(1, 2))
                        for slit in range(self.nslits):
                            science_beams[1, :, slit, :, :] *= norm_factor[slit]
                        reduced_data[0, :, :,step_ctr, :] = np.nanmean(science_beams[:, 0, :, :, :], axis=0)
                        reduced_data[1:, :, :, step_ctr, :] = (
                            science_beams[0, 1:, :, :, :] - science_beams[1, 1:, :, :, :]
                        ) / 2
                        tmtx = pol.get_dst_matrix(
                            [hdul[0].header['DST_AZ'], hdul[0].header['DST_EL'], hdul[0].header['DST_TBL']],
                            self.central_wavelength,
                            180,
                            self.t_matrix_file
                        )
                        inv_tmtx = np.linalg.inv(tmtx)
                        for slit in range(self.nslits):
                            for profile in range(reduced_data.shape[2]):
                                reduced_data[:, slit, profile, step_ctr, :] = (
                                    inv_tmtx @
                                    xinv_interp[profile, slit, :, :] @
                                    reduced_data[:, slit, profile, step_ctr, :]
                                )
                        # Parallactic angle for QU rotation correction
                        angular_geometry = pol.spherical_coordinate_transform(
                            [hdul[0].header['DST_AZ'], hdul[0].header['DST_EL']],
                            self.site_latitude
                        )
                        # Sub off P0 angle
                        rotation = np.pi + angular_geometry[2] - hdul[0].header['DST_PEE'] * np.pi / 180
                        crot = np.cos(-2 * rotation)
                        srot = np.sin(-2 * rotation)
                        # Make Q/U copies since the correction to each depends on the other
                        qtmp = reduced_data[1, :, :, step_ctr, :].copy()
                        utmp = reduced_data[2, :, :, step_ctr, :].copy()
                        reduced_data[1, :, :, step_ctr, :] = crot * qtmp + srot * utmp
                        reduced_data[2, :, :, step_ctr, :] = -srot * qtmp + crot * utmp
                        # Last thing we need the file open for: grab the slit width as a proxy for step size

                        slit_width = hdul[0].header['SLITWDTH'] # mm
                        # Exit the FITS context manager and close the file
                    # Here, we have to make a choice. We need to do crosstalks, prefilter corrections, and defringe.
                    # I'm torn on what the best order of this is, but my gut feeling is:
                    #   1.) Prefilter to flatten Stokes-I
                    #   2.) I->QUV crosstalk to flatten QUV
                    #   3.) QUV Fringe removal
                    #   4.) V->QU
                    # Prefilter/spectral efficiency correction:
                    wavegrid = np.zeros((self.nslits, reduced_data.shape[-1]))
                    for slit in range(self.nslits):
                        # The wavelength calibration should really be done for each slit, just in case...
                        wavelength_array = self.tweak_wavelength_calibration(
                            np.nanmean(reduced_data[0, slit, :, :step_ctr + 1, :], axis=(0, 1))
                        )
                        wavegrid[slit, :] = wavelength_array
                        if self.spectral_transmission:
                            reduced_data[0, slit, :, step_ctr, :] = self.prefilter_correction(
                                reduced_data[0, slit, :, step_ctr, :], wavelength_array
                            )
                    # I -> QUV crosstalk correction
                    reduced_data[
                        1:, :, :, step_ctr, :
                    ], complete_i2quv_crosstalk[
                       :, :, :, :, step_ctr
                    ] = self.detrend_i_crosstalk(
                        reduced_data[1:, :, :, step_ctr, :], reduced_data[0, :, :, step_ctr, :]
                    )
                    # Next up; sub off fringe template, perform crosstalk correction, set up overviews.
                    if fringe_template is not None:
                        # Subtract fringes
                        reduced_data[1:, :, :, step_ctr, :] = self.defringe_from_template(
                            reduced_data[1:, :, :, step_ctr, :], fringe_template
                        )
                    if any((v2q, v2u, q2v, u2v)):
                        # Internal Crosstalk
                        crosstalk_results = self.detrend_internal_crosstalk(
                            reduced_data[1:, :, :, step_ctr, :], wavegrid
                        )
                        reduced_data[1:, :, :, step_ctr, :] = crosstalk_results[0]
                        complete_internal_crosstalks[:, :, :, step_ctr] = crosstalk_results[1]
                    # Grab analysis lines if they're not the default.
                    # Skip if overview set to false, i.e., reducing a flat field
                    if (step_ctr == 0) and (self.analysis_ranges == "choose") and overview:
                        mean_profile = np.nanmean(reduced_data[0, 0, :, 0, :], axis=0)
                        print(
                            "Select spectral ranges (xmin, xmax) for overview maps and initial analysis.\n"
                            "Close window when you're done."
                        )
                        coarse_indices = spex.select_spans_singlepanel(
                            mean_profile, xarr=wavegrid[0], fig_name="Select Lines for Analysis"
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
                        self.analysis_indices = np.zeros((2, self.nslits, len(line_cores)))
                        line_core_arr = np.zeros((self.nslits, len(line_cores)))
                        for slit in range(self.nslits):
                            for j in range(coarse_indices.shape[0]):
                                average_delta = np.mean(np.abs(coarse_indices[j, :] - line_cores[j]))
                                min_idx_s0 = int(round(line_cores[j] - average_delta, 0))
                                max_idx_s0 = int(round(line_cores[j] + average_delta, 0) + 1)
                                self.analysis_indices[0, slit, j] = spex.find_nearest(
                                    wavegrid[slit], wavegrid[0, min_idx_s0]
                                )
                                self.analysis_indices[1, slit, j] = spex.find_nearest(
                                    wavegrid[slit], wavegrid[0, max_idx_s0]
                                )
                                line_core_arr[slit, j] = spex.find_nearest(
                                    wavegrid[slit], wavegrid[0, line_cores[j]]
                                )
                    elif (step_ctr == 0) and (self.analysis_ranges == "default") and overview:
                        # Default analysis ranges, i.e., Si I and He I lines.
                        # Find indices of default ranges in each slit
                        self.analysis_indices = np.zeros((2, self.nslits, self.default_analysis_ranges.shape[1]))
                        line_core_arr = np.zeros((self.nslits, self.default_analysis_ranges.shape[1]))
                        for slit in range(self.nslits):
                            for line in range(self.default_analysis_ranges.shape[1]):
                                self.analysis_indices[0, slit, line] = spex.find_nearest(
                                    wavegrid[slit], self.default_analysis_ranges[0, line]
                                )
                                self.analysis_indices[1, slit, line] = spex.find_nearest(
                                    wavegrid[slit], self.default_analysis_ranges[1, line]
                                )
                                line_core_arr[slit, line] = spex.find_nearest(
                                    wavegrid[slit], self.default_reference_wavelengths[line]
                                )
                    if overview:
                        plt.ion()
                        # If step 0, set up overview maps to blit data into
                        # Unlike SPINOR, we're considering multiple slits,
                        # so we'll be updating nslits columns each step
                        if step_ctr == 0:
                            field_images = np.zeros((
                                self.analysis_indices.shape[2], # Nlines
                                4, # IQUV,
                                reduced_data.shape[2],
                                int(len(filelist) * self.nslits)
                            ))
                        for line in range(self.analysis_indices.shape[2]):
                            for slit in range(self.nslits):
                                field_images[
                                    line, 0, :, step_ctr+slit*len(filelist)
                                ] = reduced_data[0, slit, :, step_ctr, int(line_core_arr[slit, line])]
                                for k in range(1, 4):
                                    field_images[line, k, :, step_ctr+slit*len(filelist)] = scinteg.trapezoid(
                                        np.nan_to_num(np.abs(
                                            # What a mess.. clean this up!
                                            reduced_data[
                                                k, slit, :, step_ctr,
                                                int(
                                                    self.analysis_indices[0, slit, line]
                                                ): int(self.analysis_indices[1, slit, line])
                                            ] / reduced_data[
                                                0, slit, :, step_ctr,
                                                int(
                                                    self.analysis_indices[0, slit, line]
                                                ):int(self.analysis_indices[1, slit, line])
                                            ]
                                        )), axis=-1
                                    )
                        if step_ctr == 0:
                            slit_plate_scale = self.telescope_plate_scale * self.dst_collimator / self.slit_camera_lens
                            camera_dy = slit_plate_scale * self.pixel_size / 1000
                            map_dx = slit_plate_scale * slit_width
                            plot_params = self.set_up_live_plot(
                                field_images, reduced_data[:, :, :, step_ctr, :],
                                complete_internal_crosstalks[:, :, :, step_ctr],
                                camera_dy, map_dx
                            )
                        self.update_live_plot(
                            *plot_params, field_images, reduced_data[:, :, :, step_ctr, :],
                            complete_internal_crosstalks[:, :, :, step_ctr], step_ctr
                        )
                    step_ctr += 1
                    pbar.update(1)
            # Completed series, moving to next repeat, but save first
            if overview and self.save_figs:
                if self.analysis_ranges == "default":
                    names = ["SiI_10827", "HeI_10829", "HeI_10830"]
                else:
                    names = ["line{0}".format(i) for i in range(len(plot_params[0]))]
                for fig in range(len(plot_params[0])):
                    filename = os.path.join(
                        self.final_dir,
                        "field_image_{0}_map{1}_repeat{2}.png".format(names[fig], index, n)
                    )
                    plot_params[0][fig].savefig(filename, bbox_inches='tight')
            if write:
                wavelength_grids = np.zeros((self.nslits, reduced_data.shape[-1]))
                for slit in range(self.nslits):
                    mean_profile = np.nanmean(reduced_data[0, slit], axis=(0, 1))
                    wavelength_grids[slit] = self.tweak_wavelength_calibration(mean_profile)
                reduced_filename = self.package_scan(
                    reduced_data, wavelength_grids, master_hairline_centers, filelist
                )
                crosstalk_filename = self.package_crosstalks(
                    complete_i2quv_crosstalk, complete_internal_crosstalks, index, n
                )
                parameter_maps, reference_wavelengths, flat_profile, flat_waves = self.firs_analysis(
                    reduced_data, self.analysis_indices
                )
                param_filename = self.package_analysis(
                    parameter_maps,
                    reference_wavelengths,
                    self.analysis_indices,
                    flat_profile,
                    flat_waves,
                    reduced_filename
                )

                if self.verbose:
                    print("\n\n=====================")
                    print("Saved Reduced Data at: {0}".format(reduced_filename))
                    print("Saved Parameter Maps at: {0}".format(param_filename))
                    print("Saved Crosstalk Coefficients at: {0}".format(crosstalk_filename))
                    print("=====================\n\n")
                if overview:
                    plt.pause(2)
                    plt.close("all")
        return reduced_data


    def detrend_i_crosstalk(
            self, quv_data: np.ndarray, i_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs and applies I->QUV crosstalk.
        Two methods currently implemented;
            1.) Old method; continuum region defined and used to determine a single crosstalk value
            2.) New method; linear crosstalk along wavelength axis.

        Parameters
        ----------
        quv_data : numpy.ndarray
            Partially-reduced Stokes-QUV array. Shape (3, nslits, ny, nlambda)
        i_data : numpy.ndarray
            Reduced Stokes-I array. Shape (nslits, ny, nlambda)

        Returns
        -------
        quv_data : numpy.ndarray
            Shape (3, nslits, ny, nlambda), crosstalk-corrected Stokes-QUV
        crosstalk_i2quv : numpy.ndarray
            Shape (3, 2, nslits, ny), contains I->QUV crosstalk coefficients
        """
        crosstalk_i2quv = np.zeros((3, 2, self.nslits, quv_data.shape[2]))

        if self.crosstalk_continuum is not None:
            # Shape 3xNSlitsxNY
            i2quv = np.mean(
                quv_data[:, :, :, self.crosstalk_continuum[0]:self.crosstalk_continuum[1]]/
                np.repeat(
                    i_data[np.newaxis, :, :, self.crosstalk_continuum[0]:self.crosstalk_continuum[1]],
                    3, axis=0
                ), axis=-1
            )
            quv_data = quv_data - np.repeat(
                i2quv[:, :, :, np.newaxis], quv_data.shape[-1], axis=-1
            ) * np.repeat(i_data[np.newaxis, :, :, :], 3, axis=0)
            crosstalk_i2quv[:, 1, :, :] = i2quv
        else:
            for i in range(quv_data.shape[0]):
                for j in range(quv_data.shape[1]):
                    for k in range(quv_data.shape[2]):
                        quv_data[i, j, k, :], crosstalk_i2quv[i, :, j, k] = pol.i2quv_crosstalk(
                            i_data[j, k, :], quv_data[i, j, k, :]
                        )
        return quv_data, crosstalk_i2quv

    def detrend_internal_crosstalk(
            self, quv_data: np.ndarray, wavelength_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detrend internal crosstalks. Unlike SPINOR, in order to avoid spurious fringe effects,
        we should pick a window surrounding a magnetically-sensitive line to use for the crosstalk cosine
        similarity determination. Si I 10827 will be the default, but we'll write the function for generality

        Parameters
        ----------
        quv_data : numpy.ndarray
            Array of QUV data. Assumed to be roughly defringed and of shape (3, nslits, ny, nlambda)
        wavelength_grid : numpy.ndarray
            Array of wavelengths corresponding to nlambda axis of quv_data. Shape (nslits, nlambda)

        Returns
        -------
        quv_data : numpy.ndarray
        internal_crosstalk : numpy.ndarray
            Of shape (4, nslits, ny). Axis 0 corresponds to V->Q, V->U, Q->V, U->V
        """
        internal_crosstalk = np.zeros((4, self.nslits, quv_data.shape[2]))
        if self.v2q:
            for slit in range(self.nslits):
                # Baseline bulk crosstalk first
                bulk_v2q_crosstalk = pol.internal_crosstalk_2d(
                    quv_data[0, slit, :, :], quv_data[2, slit, :, :]
                )
                quv_data[0, slit, :, :] = quv_data[0, slit, :, :] - bulk_v2q_crosstalk * quv_data[2, slit, :, :]
                if self.v2q == "full":
                    # Determine crosstalks from range around self.internal_crosstalk_line
                    min_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line - self.crosstalk_range
                    ))
                    max_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line + self.crosstalk_range
                    ))
                    for y in range(quv_data.shape[2]):
                        _, ct_val = pol.v2qu_crosstalk(
                            quv_data[2, slit, y, min_idx:max_idx], quv_data[0, slit, y, min_idx:max_idx]
                        )
                        internal_crosstalk[0, slit, y] = ct_val
                        quv_data[0, slit, y, :] = quv_data[0, slit, y, :] - ct_val * quv_data[2, slit, y, :]
                internal_crosstalk[0, slit] += bulk_v2q_crosstalk
        if self.v2u:
            for slit in range(self.nslits):
                # Baseline bulk crosstalk first
                bulk_v2u_crosstalk = pol.internal_crosstalk_2d(
                    quv_data[1, slit, :, :], quv_data[2, slit, :, :]
                )
                quv_data[1, slit, :, :] = quv_data[1, slit, :, :] - bulk_v2u_crosstalk * quv_data[2, slit, :, :]
                if self.v2u == "full":
                    # Determine crosstalks from range around self.internal_crosstalk_line
                    min_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line - self.crosstalk_range
                    ))
                    max_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line + self.crosstalk_range
                    ))
                    for y in range(quv_data.shape[2]):
                        _, ct_val = pol.v2qu_crosstalk(
                            quv_data[2, slit, y, min_idx:max_idx], quv_data[1, slit, y, min_idx:max_idx]
                        )
                        internal_crosstalk[1, slit, y] = ct_val
                        quv_data[1, slit, y, :] = quv_data[1, slit, y, :] - ct_val * quv_data[2, slit, y, :]
                internal_crosstalk[1, slit] += bulk_v2u_crosstalk
        if self.q2v:
            for slit in range(self.nslits):
                # Baseline bulk crosstalk first
                bulk_q2v_crosstalk = pol.internal_crosstalk_2d(
                    quv_data[2, slit, :, :], quv_data[0, slit, :, :]
                )
                quv_data[2, slit, :, :] = quv_data[2, slit, :, :] - bulk_q2v_crosstalk * quv_data[0, slit, :, :]
                if self.q2v == "full":
                    # Determine crosstalks from range around self.internal_crosstalk_line
                    min_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line - self.crosstalk_range
                    ))
                    max_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line + self.crosstalk_range
                    ))
                    for y in range(quv_data.shape[2]):
                        _, ct_val = pol.v2qu_crosstalk(
                            quv_data[0, slit, y, min_idx:max_idx], quv_data[2, slit, y, min_idx:max_idx]
                        )
                        internal_crosstalk[3, slit, y] = ct_val
                        quv_data[2, slit, y, :] = quv_data[2, slit, y, :] - ct_val * quv_data[0, slit, y, :]
                internal_crosstalk[2, slit] += bulk_q2v_crosstalk
        if self.u2v:
            for slit in range(self.nslits):
                # Baseline bulk crosstalk first
                bulk_u2v_crosstalk = pol.internal_crosstalk_2d(
                    quv_data[2, slit, :, :], quv_data[1, slit, :, :]
                )
                quv_data[2, slit, :, :] = quv_data[2, slit, :, :] - bulk_u2v_crosstalk * quv_data[1, slit, :, :]
                if self.u2v == "full":
                    # Determine crosstalks from range around self.internal_crosstalk_line
                    min_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line - self.crosstalk_range
                    ))
                    max_idx = int(spex.find_nearest(
                        wavelength_grid[slit], self.internal_crosstalk_line + self.crosstalk_range
                    ))
                    for y in range(quv_data.shape[2]):
                        _, ct_val = pol.v2qu_crosstalk(
                            quv_data[1, slit, y, min_idx:max_idx], quv_data[2, slit, y, min_idx:max_idx]
                        )
                        internal_crosstalk[3, slit, y] = ct_val
                        quv_data[2, slit, y, :] = quv_data[2, slit, y, :] - ct_val * quv_data[1, slit, y, :]
                internal_crosstalk[3, slit] += bulk_u2v_crosstalk
        return quv_data, internal_crosstalk

    def construct_fringe_template_from_flat(self, reduced_flat: np.ndarray) -> np.ndarray:
        """
        Performs median and Fourier filtering to construct a fringe template from a solar flat

        Parameters
        ----------
        reduced_flat : numpy.ndarray
            Flat field that's undergone the same reduction process as a science map.
            Shape (4, nslits, ny, 32, nlambda) as FIRS flats are 32 steps each.

        Returns
        -------
        fringe_template : numpy.ndarray
            Flat field that's been median filtered along the slit, with high/low frequency information removed.
            Shape (3, nslits, ny, nlambda)
        """
        fringe_template = np.zeros((
            3, self.nslits, reduced_flat.shape[2], reduced_flat.shape[-1]
        ))
        mean_flat = np.mean(reduced_flat, axis=3)
        with tqdm.tqdm(total=3 * self.nslits * reduced_flat.shape[2], desc="Constructing Fringe Template") as pbar:
            # Fringe template likely varies for each slit
            for slit in range(self.nslits):
                wavelength_array = self.tweak_wavelength_calibration(
                    np.mean(mean_flat[0, slit, 50:-50, :], axis=0)
                )
                fft_frequencies = np.fft.fftfreq(
                    len(wavelength_array),
                    np.mean(wavelength_array[1:] - wavelength_array[:-1])
                )
                ft_cut1 = fft_frequencies >= max(self.fringe_frequency)
                ft_cut2 = fft_frequencies <= min(self.fringe_frequency)
                medfilt_flat = scind.median_filter(mean_flat[:, slit, :, :], size=(1, 32, 16))
                for y in range(medfilt_flat.shape[1]):
                    for stoke in range(1, 4):
                        quv_ft = np.fft.fft(medfilt_flat[stoke, y, :])
                        quv_ft[ft_cut1] = 0
                        quv_ft[ft_cut2] = 0
                        fringe_template[stoke-1, slit, y, :] = np.real(np.fft.ifft(quv_ft))
                        pbar.update(1)
        return fringe_template

    def defringe_from_template(self, data_slice: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        Removes polarimetric fringes via template
        Parameters
        ----------
        data_slice : numpy.ndarray
            Slice of QUV data for defringe. Shape 3, nslits, ny, nlambda
        template : numpy.ndarray
            Template of QUV fringes. Currently constructed from flats assuming mostly-static fringes

        Returns
        -------
        defringed_data_slice : numpy.ndarray
        """
        defringed_data_slice = np.zeros(data_slice.shape)
        for stoke in range(data_slice.shape[0]):
            for slit in range(data_slice.shape[1]):
                for y in range(data_slice.shape[2]):
                    fringe_med = np.nanmedian(template[stoke, slit, y, :50])
                    map_med = np.nanmedian(data_slice[stoke, slit, y, :50])
                    corr_factor = fringe_med - map_med
                    fringe_corr = template[stoke, slit, y, :] - corr_factor
                    defringed_data_slice[stoke, slit, y, :] = data_slice[stoke, slit, y, :] - fringe_corr
        return defringed_data_slice

    def prefilter_correction(
            self, data_slice: np.ndarray, wavelength_array: np.ndarray, degrade_to: int=50, rolling_window: int=8
    ) -> np.ndarray:
        """
        Performs correction for prefilter curvature/spectrograph efficiency differences along wavelength in Stokes-I
        This correction is performed by dividing the FIRS reference spectrum by the FTS reference spectrum.
        In an ideal world, you'd end up with the prefilter profile from this.
        This is not an ideal world, and there are usually strong residuals from this.
        Instead, we degrade this to a small number of points, median filter it, and then fit a polynomial to that
        degraded residual profile. Interpolating this back up gets a reasonable prefilter estimate.

        Parameters
        ----------
        data_slice : numpy.ndarray
            Slice of Stokes-I data to determine prefilter profiles for. Of the shape (ny, nlambda)
        wavelength_array : numpy.ndarray
            1D array of wavelength grid values
        degrade_to : int
            Number of points for the profile to fit. Default 50 points
        rolling_window : int
            Width of the window along the spectral axis to median filter by. Default 8.

        Returns
        -------
        data_slice : numpy.ndarray
            Corrected for spectrograph/prefilter curvature
        """
        fts_wave, fts_spec = spex.fts_window(wavelength_array[0], wavelength_array[-1])
        fts_spec_firs_res = np.interp(wavelength_array, fts_wave, fts_spec)
        degraded_wavelengths = np.linspace(wavelength_array[0], wavelength_array[-1], num=degrade_to)
        # Median filter along spatial direction
        medfilt_divided = scind.median_filter(data_slice, size=(25, 1)) / fts_spec_firs_res
        degraded_medfilt = scinterp.CubicSpline(
            wavelength_array, medfilt_divided, axis=-1, extrapolate=True
        )(degraded_wavelengths)
        pfc = scinterp.CubicSpline(
            degraded_wavelengths, scind.median_filter(degraded_medfilt, size=(1, rolling_window)),
            axis=-1, extrapolate=True
        )(wavelength_array)
        pfc /= np.repeat(np.nanmax(pfc, axis=1)[:, np.newaxis], pfc.shape[1], axis=1)

        return data_slice/pfc

    def tweak_wavelength_calibration(self, reference_profile: np.ndarray) -> np.ndarray:
        """
        Determines wavelength array from grating parameters and FTS reference

        Parameters
        ----------
        reference_profile : numpy.ndarray
            1D array containing reference spectral profile

        Returns
        -------
        wavelength_array : numpy.ndarray
            1D array with corresponding wavelengths
        """
        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens,
            slit_width=self.slit_width
        )
        # Getting Min/Max Wavelength for FTS comparison; padding by 30 pixels on either side
        # Same selection process as in flat fielding.
        apx_wavemin = self.central_wavelength - np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax = self.central_wavelength + np.nanmean(self.slit_edges) * grating_params['Spectral_Pixel'] / 1000
        apx_wavemin -= 30 * grating_params['Spectral_Pixel'] / 1000
        apx_wavemax += 30 * grating_params['Spectral_Pixel'] / 1000
        fts_wave, fts_spec = spex.fts_window(apx_wavemin, apx_wavemax)
        fts_core = sorted(np.array(self.fts_line_cores))

        firs_line_cores = sorted(np.array(self.firs_line_cores))

        fts_core_waves = [scinterp.CubicSpline(np.arange(len(fts_wave)), fts_wave)(lam) for lam in fts_core]
        # Update FIRS selected line cores by redoing core finding with wide, then narrow range
        firs_line_cores = np.array([
            spex.find_line_core(
                reference_profile[int(lam) - 10:int(lam) + 11]
            ) + int(lam) - 10 for lam in firs_line_cores
        ])
        firs_line_cores = np.array([
            spex.find_line_core(
                reference_profile[int(lam) - 5:int(lam) + 7]
            ) + int(lam) - 5
            for lam in firs_line_cores
        ])
        angstrom_per_pixel = np.abs(fts_core_waves[1] - fts_core_waves[0]) / np.abs(
            firs_line_cores[1] - firs_line_cores[0])
        zerowvl = fts_core_waves[0] - (angstrom_per_pixel * firs_line_cores[0])
        wavelength_array = (np.arange(0, len(reference_profile)) * angstrom_per_pixel) + zerowvl
        return wavelength_array

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
                    alignment_image[i, j, int(hairline_minimum):int(hairline_maximum), :],
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
                    np.nanmedian(deskewed_image[int(hairline_minimum):int(hairline_maximum), 50:-50], axis=1)
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
        cutout_beams : numpy.ndarray
            Array containing the hairline-aligned beams. Shape (2, 4, nslits, ny, nx, nlambda)
        hairline_centers : numpy.ndarray
            Center of hairlines. Beams should all be registered to this common hairline.

        Returns
        -------
        cutout_beams : numpy.ndarray
            Array containing the spectrally-aligned and deskewed beams. Shape (2, 4, nslits, ny, nx, nlambda)
        spectral_centers : numpy.ndarray
            Center of the per beam, per slit spectral line used for registration. Shape (2, nslits)
        """
        x1, x2 = 20, 21
        spectral_centers = np.zeros((2, self.nslits))
        for iternum in range(5):
            order = 1 if iternum < 2 else 2
            for beam in range(cutout_beams.shape[0]):
                for slit in range(cutout_beams.shape[2]):
                    upper_hairline_center = hairline_centers[beam, slit] + np.diff(self.full_hairlines[0, 0])
                    spectral_image = cutout_beams[
                        beam, 0, slit, :, int(self.firs_line_cores[0] - x1):int(self.firs_line_cores[0] + x2)
                    ].copy()
                    # Replace hairlines with NaNs to keep them from throwing off the skews
                    hair_min = int(hairline_centers[beam, slit] - 4)
                    hair_max = int(hairline_centers[beam, slit] + 5)
                    hair_min = 0 if hair_min < 0 else hair_min
                    spectral_image[hair_min:hair_max] = np.nan
                    hair_min = int(upper_hairline_center - 4)
                    hair_max = int(upper_hairline_center + 5)
                    hair_min = int(spectral_image.shape[0] - 1) if hair_min > spectral_image.shape[0] - 1 else hair_max
                    spectral_image[hair_min:hair_max] = np.nan
                    skews = spex.spectral_skew(
                        spectral_image, order=order, slit_reference=0.5
                    )
                    for profile in range(cutout_beams.shape[3]):
                        cutout_beams[beam, :, slit, profile, :] = scind.shift(
                            cutout_beams[beam, :, slit, profile, :], (0, skews[profile]),
                            mode='nearest', order=1
                        )
                    spectral_centers[beam, slit] = spex.find_line_core(
                        np.nanmedian(
                            cutout_beams[
                                beam, 0, slit, :, int(self.firs_line_cores[0] - 10):int(self.firs_line_cores[0] + 10)
                            ], axis=0
                        )
                    ) + int(self.firs_line_cores[0] - 10)
            x1 -= 3
            x2 -= 3

        return cutout_beams, spectral_centers

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
        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.indir = config["FIRS"]["rawFileDirectory"]
        self.final_dir = config["FIRS"]["reducedFileDirectory"]
        self.reduced_file_pattern = config["FIRS"]["reducedFilePattern"]
        self.parameter_map_pattern = config["FIRS"]["reducedParameterMapPattern"]
        self.t_matrix_file = config['FIRS']['tMatrixFile']

        self.nslits = int(config["FIRS"]["nSlits"]) if "nslits" in config["FIRS"].keys() else self.nslits
        self.slit_width = int(config["FIRS"]['slitWidth']) if "slitwidth" in config["FIRS"].keys() else self.slit_width
        self.slit_spacing = float(config['FIRS']['slitSpacing']) if 'slitspacing' in config['FIRS'].keys() \
            else self.slit_spacing
        self.grating_angle = float(config["FIRS"]["gratingAngle"]) if "gratingangle" in config["FIRS"].keys() \
            else self.grating_angle
        self.grating_rules = float(config["FIRS"]["gratingRules"]) if "gratingrules" in config["FIRS"].keys() \
            else self.grating_rules
        self.blaze_angle = float(config["FIRS"]["blazeAngle"]) if "blazeangle" in config["FIRS"].keys() \
            else self.blaze_angle
        self.central_wavelength = float(config["FIRS"]["centralWavelength"]) \
            if "centralwavelength" in config["FIRS"].keys() else self.central_wavelength
        self.internal_crosstalk_line = float(config["FIRS"]["crosstalkWavelength"]) \
            if "crosstalkwavelength" in config['FIRS'].keys() else self.internal_crosstalk_line
        self.crosstalk_range = float(config["FIRS"]["crosstalkRange"]) \
            if "crosstalkrange" in config["FIRS"].keys() else self.crosstalk_range
        self.spectral_order = int(config["FIRS"]["spectralOrder"]) if "spectralorder" in config["FIRS"].keys() \
            else self.spectral_order
        self.pixel_size = int(config["FIRS"]["pixelSize"]) if "pixelsize" in config["FIRS"].keys() \
            else self.pixel_size
        self.nhair = int(config['FIRS']['totalHairlines']) if 'totalhairlines' in config['FIRS'].keys() \
            else self.nhair
        self.beam_threshold = float(config['FIRS']['intensityThreshold']) \
            if "intensitythreshold" in config['FIRS'].keys() else self.beam_threshold
        self.hairline_width = float(config["FIRS"]["hairlineWidth"]) \
            if "hairlinewidth" in config["FIRS"].keys() else self.hairline_width
        self.n_subslits = int(config["FIRS"]['slitDivisions']) if "slitdivisions" in config["FIRS"].keys() \
            else self.n_subslits
        self.fringe_frequency = config['FIRS']['fringeFrequency'] if 'fringefrequency' in config['FIRS'].keys() \
            else self.fringe_frequency
        if type(self.fringe_frequency) is str:
            self.fringe_frequency = [float(i) for i in self.fringe_frequency.split(",")]
        self.verbose = config["FIRS"]['verbose'] if 'verbose' in config['FIRS'].keys() \
            else self.verbose
        if type(self.verbose) is str:
            self.verbose = True if self.verbose.lower() == "true" else False
        self.v2q = config["FIRS"]["v2q"] if "v2q" in config["FIRS"].keys() \
            else self.v2q
        self.v2q = False if self.v2q.lower() == "false" else self.v2q
        self.v2u = config["FIRS"]["v2u"] if "v2u" in config["FIRS"].keys() \
            else self.v2u
        self.v2u = False if self.v2u.lower() == "false" else self.v2u
        self.q2v = config["FIRS"]["q2v"] if "q2v" in config["FIRS"].keys() \
            else self.q2v
        self.q2v = False if self.q2v.lower() == "false" else self.q2v
        self.u2v = config["FIRS"]["u2v"] if "u2v" in config["FIRS"].keys() \
            else self.u2v
        self.u2v = False if self.u2v.lower() == "false" else self.u2v

        self.despike = config['FIRS']['despike'] if "despike" in config['FIRS'].keys() else self.despike
        if type(self.despike) is str:
            self.despike = True if self.despike.lower() == "true" else False
        self.plot = config["FIRS"]['plot'] if 'plot' in config['FIRS'].keys() \
            else self.plot
        if type(self.plot) is str:
            self.plot = True if self.plot.lower() == "true" else False
        self.save_figs = config['FIRS']['saveFigs'] if 'savefigs' in config['FIRS'].keys() else self.save_figs
        if type(self.save_figs) is str:
            self.save_figs = True if self.save_figs.lower() == "true" else False
        self.crosstalk_continuum = config['FIRS']['crosstalkContinuum'] \
            if 'crosstalkcontinuum' in config['FIRS'].keys() else self.crosstalk_continuum
        if self.crosstalk_continuum is not None:
            self.crosstalk_continuum = [int(i) for i in self.crosstalk_continuum.split(",")]
        self.analysis_ranges = config['FIRS']['analysisRanges'] if 'analysisranges' in config['FIRS'].keys() \
            else self.analysis_ranges
        self.defringe = config['FIRS']['defringeMethod'] if 'defringemethod' in config['FIRS'].keys() \
            else self.defringe
        self.spectral_transmission = config['FIRS']['spectralTransmission'] \
            if 'spectraltransmission' in config['FIRS'].keys() else self.spectral_transmission
        if type(self.spectral_transmission) is str:
            self.spectral_transmission = True if self.spectral_transmission.lower() == "true" else False
        self.slit_camera_lens = float(config["FIRS"]["slitCameraLens"]) if "slitcameralens" in config['FIRS'].keys() \
            else self.slit_camera_lens
        self.telescope_plate_scale = float(config["FIRS"]["telescopePlateScale"]) \
            if "telescopeplatescale" in config["FIRS"].keys() else self.telescope_plate_scale
        self.dst_collimator = float(config["FIRS"]["DSTCollimator"]) \
            if "dstcollimator" in config["FIRS"].keys() else self.dst_collimator
        self.spectrograph_collimator = float(config["FIRS"]["spectrographCollimator"]) \
            if "spectrographcollimator" in config["FIRS"].keys() else self.spectrograph_collimator
        self.camera_lens = float(config['FIRS']['cameraLens']) if "cameralens" in config['FIRS'].keys() \
            else self.camera_lens
        self.site_latitude = float(config['FIRS']['siteLatitude']) if "sitelatitude" in config['FIRS'].keys() \
            else self.site_latitude
        self.cal_retardance = float(config['FIRS']['calRetardance']) if "calretardance" in config['FIRS'].keys() \
            else self.cal_retardance
        self.ilimit = config['FIRS']['polcalClipThreshold'] if 'polcalclipthreshold' in config['FIRS'].keys() \
            else self.ilimit
        if type(self.ilimit) is str:
            self.ilimit = [float(i) for i in self.ilimit.split(",")]
        self.polcal_processing = config['FIRS']['polcalProcessing'] if "polcalprocessing" in config['FIRS'].keys() \
            else self.polcal_processing
        if type(self.polcal_processing) is str:
            self.polcal_processing = True if self.polcal_processing.lower() == "true" else False

        if (
            ("qmodulationpattern" in config['FIRS'].keys()) and
            ("umodulationpattern" in config['FIRS'].keys()) and
            ("vmodulationpattern" in config['FIRS'].keys()) and
            ("polnorm" in config['FIRS'].keys())
        ):
            qmod = [int(mod) for mod in config["FIRS"]["qModulationPattern"].split(",")]
            umod = [int(mod) for mod in config["FIRS"]["uModulationPattern"].split(",")]
            vmod = [int(mod) for mod in config["FIRS"]["vModulationPattern"].split(",")]
            self.pol_demod = np.array([
                [1, 1, 1, 1],
                qmod,
                umod,
                vmod
            ], dtype=int)
            self.pol_norm = [float(i) for i in config['FIRS']['polNorm'].split(",")]

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
        time_sort = np.argsort(np.array(final_starttime))
        self.obssum_info = np.rec.fromarrays(
            [
                np.array(final_obsseries)[time_sort], np.array(final_starttime)[time_sort],
                np.array(final_endtime)[time_sort], np.array(final_exptime)[time_sort],
                np.array(final_coadd)[time_sort], np.array(final_repeats)[time_sort],
                np.array(final_nfiles)[time_sort], np.array(final_obstype)[time_sort]
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

    def set_up_live_plot(
            self, field_images: np.ndarray, slit_images: np.ndarray,
            internal_crosstalks: np.ndarray, dy: float, dx: float
    ) -> tuple:
        """
        Initializes live plotting statements for monitoring reduction progress

        Parameters
        ----------
        field_images : numpy.ndarray
            Array of field images. Will be increasingly filled-in with each loop.
            Shape (nlines, 4, ny, nx)
        slit_images : numpy.ndarray
            Array of IQUV slit images. Shape (4, nslits, ny, nlambda).
            I'm still undecided whether I want to flatten along nslits, or create a new axes subplot for each slit
        internal_crosstalks : numpy.ndarray
            Internal crosstalk values for monitoring. Shape (4, nslits, ny), where the 4-axis corresponds to:
                1.) V->Q
                2.) V->U
                3.) Q->V
                4.) U->V
        dy : float
            Plate scale along the slit. Should be ~0.15" for Virgo array in default configuration
        dx : float
            Step scale along raster. Default is dense-sampling, but may be different.

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
        crosstalk_fig : matplotlib.pyplot.figure
            Figure containing crosstalk values
        v2q : list
            List containing per-slit crosstalk plot information
        v2u : list
            List containing per-slit crosstalk plot information
        q2v : list
            List containing per-slit crosstalk plot information
        u2v : list
            List containing per-slit crosstalk plot information
        """
        # Close all figures to reset the plotting landscape
        plt.close("all")
        plt.ion()
        plt.pause(0.005)

        if self.nslits > 1:
            # Combine multiple slits into single arrays for the purposes of plotting
            # Field images are already flattened
            flattened_slit_images = np.concatenate(
                [slit_images[:, i, :, :] for i in range(slit_images.shape[1])], axis=2
            )
        else:
            flattened_slit_images = slit_images[:, 0, :, :]
        slit_aspect_ratio = flattened_slit_images.shape[2] / flattened_slit_images.shape[1]
        slit_fig = plt.figure("Reduced Slit Images", figsize=(5, 5/slit_aspect_ratio))
        slit_gs = slit_fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
        slit_ax_i = slit_fig.add_subplot(slit_gs[0, 0])
        slit_i = slit_ax_i.imshow(flattened_slit_images[0], cmap='gray', origin='lower')
        slit_ax_i.text(10, 10, "I", color='C1')
        slit_ax_q = slit_fig.add_subplot(slit_gs[0, 1])
        slit_q = slit_ax_q.imshow(flattened_slit_images[1], cmap='gray', origin='lower')
        slit_ax_q.text(10, 10, "Q", color='C1')
        slit_ax_u = slit_fig.add_subplot(slit_gs[1, 0])
        slit_u = slit_ax_u.imshow(flattened_slit_images[2], cmap='gray', origin='lower')
        slit_ax_u.text(10, 10, "U", color='C1')
        slit_ax_v = slit_fig.add_subplot(slit_gs[1, 1])
        slit_v = slit_ax_v.imshow(flattened_slit_images[3], cmap='gray', origin='lower')
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
            if self.analysis_ranges == "default":
                field_fig_list.append(
                    plt.figure(self.default_analysis_names[j], figsize=(5, 5 / field_aspect_ratio + 1))
                )
                field_fig_list[-1].suptitle(self.default_analysis_names[j])
            else:
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
            
        if not any((self.v2q, self.v2u, self.q2v, self.u2v)):
            plt.show(block=False)
            plt.pause(0.05)
            return (
                field_fig_list,
                field_i, field_q, field_u, field_v,
                slit_fig,
                slit_i, slit_q, slit_u, slit_v,
                None, None, None, None, None
            )
        else:
            crosstalk_fig = plt.figure("Internal Crosstalks Along Slit", figsize=(4, 2.5))
            v2q_ax = crosstalk_fig.add_subplot(141)
            v2u_ax = crosstalk_fig.add_subplot(142)
            q2v_ax = crosstalk_fig.add_subplot(143)
            u2v_ax = crosstalk_fig.add_subplot(144)

            v2q = []
            v2u = []
            q2v = []
            u2v = []

            v2q_ax.set_xlim(-1.05, 1.05)
            v2q_ax.set_ylim(0, internal_crosstalks.shape[2])
            v2q_ax.set_title("V->Q Crosstalk")
            v2q_ax.set_ylabel("Position Along Slit")

            v2u_ax.set_xlim(-1.05, 1.05)
            v2u_ax.set_ylim(0, internal_crosstalks.shape[2])
            v2u_ax.set_title("V->U Crosstalk")
            v2u_ax.set_xlabel("Crosstalk Value")

            q2v_ax.set_xlim(-1.05, 1.05)
            q2v_ax.set_ylim(0, internal_crosstalks.shape[2])
            q2v_ax.set_title("Q->V Crosstalk [residual]")

            u2v_ax.set_xlim(-1.05, 1.05)
            u2v_ax.set_ylim(0, internal_crosstalks.shape[2])
            u2v_ax.set_title("U->V Crosstalk [residual]")

            for slit in range(self.nslits):
                v2q.append(v2q_ax.plot(
                    internal_crosstalks[0, slit, :], np.arange(internal_crosstalks.shape[2]),
                    color='C{0}'.format(slit), label="Crosstalk for slit {0} of {1}".format(slit+1, self.nslits)
                ))

                v2u.append(v2u_ax.plot(
                    internal_crosstalks[1, slit, :], np.arange(internal_crosstalks.shape[2]),
                    color='C{0}'.format(slit)
                ))

                q2v.append(q2v_ax.plot(
                    internal_crosstalks[2, slit, :], np.arange(internal_crosstalks.shape[2]),
                    color='C{0}'.format(slit)
                ))

                u2v.append(u2v_ax.plot(
                    internal_crosstalks[3, slit, :], np.arange(internal_crosstalks.shape[2]),
                    color="C{0}".format(slit)
                ))

            crosstalk_fig.legend(loc="lower center")

            plt.show(block=False)
            plt.pause(0.05)
            return (
                field_fig_list, field_i, field_q, field_u, field_v,
                slit_fig, slit_i, slit_q, slit_u, slit_v,
                crosstalk_fig, v2q, v2u, q2v, u2v
            )

    def update_live_plot(
            self,
            field_fig_list: list, field_i: list, field_q: list, field_u: list, field_v: list,
            slit_fig: matplotlib.pyplot.figure, slit_i: matplotlib.image.AxesImage, slit_q: matplotlib.image.AxesImage,
            slit_u: matplotlib.image.AxesImage, slit_v: matplotlib.image.AxesImage,
            crosstalk_fig: matplotlib.pyplot.figure, v2q: list, v2u: list, q2v: list, u2v: list,
            field_images: np.ndarray, slit_images: np.ndarray, internal_crosstalks: np.ndarray,
            step: int
    ) -> None:
        """
        Updates the plots created in firsCal.set_up_live_plot()

        Parameters
        ----------
        field_fig_list : list
        field_i : list
        field_q : list
        field_u : list
        field_v : list
        slit_fig : matplotlib.pyplot.Figure
        slit_i : matplotlib.image.AxesImage
        slit_q : matplotlib.image.AxesImage
        slit_u : matplotlib.image.AxesImage
        slit_v : matplotlib.image.AxesImage
        crosstalk_fig : matplotlib.pyplot.Figure
        v2q : list
        v2u : list
        q2v : list
        u2v : list
        field_images : numpy.ndarray
        slit_images : numpy.ndarray
        internal_crosstalks : numpy.ndarray
        step : int
            Step of reduction process. Necessary for normalization.

        Returns
        -------

        """

        if self.nslits > 1:
            # Combine multiple slits into single arrays for the purposes of plotting
            # Field images are already flattened
            flattened_slit_images = np.concatenate(
                [slit_images[:, i, :, :] for i in range(slit_images.shape[1])], axis=2
            )
        else:
            flattened_slit_images = slit_images[:, 0, :, :]

        slit_i.set_array(flattened_slit_images[0])
        slit_i.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[0]) - 3 * np.std(slit_images[0]),
                vmax=np.mean(slit_images[0]) + 3 * np.std(slit_images[0])
            )
        )
        slit_q.set_array(flattened_slit_images[1])
        slit_q.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[1]) - 3 * np.std(slit_images[1]),
                vmax=np.mean(slit_images[1]) + 3 * np.std(slit_images[1])
            )
        )
        slit_u.set_array(flattened_slit_images[2])
        slit_u.set_norm(
            matplotlib.colors.Normalize(
                vmin=np.mean(slit_images[2]) - 3 * np.std(slit_images[2]),
                vmax=np.mean(slit_images[2]) + 3 * np.std(slit_images[2])
            )
        )
        slit_v.set_array(flattened_slit_images[3])
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
            for slit in range(self.nslits):
                v2q[slit][0].set_data(internal_crosstalks[0, slit], np.arange(internal_crosstalks.shape[2]))
                v2u[slit][0].set_data(internal_crosstalks[1, slit], np.arange(internal_crosstalks.shape[2]))
                q2v[slit][0].set_data(internal_crosstalks[2, slit], np.arange(internal_crosstalks.shape[2]))
                u2v[slit][0].set_data(internal_crosstalks[3, slit], np.arange(internal_crosstalks.shape[2]))
            crosstalk_fig.canvas.draw()
            crosstalk_fig.canvas.flush_events()
        return

    def package_scan(
            self, datacube: np.ndarray, wavelength_array: np.ndarray, hairline_centers: tuple, filelist: list
    ) -> str:
        """
        Packages reduced scan into FITS HDUList. HDUList has 7 extensions:
            1.) Empty data attr with top-level header info
            2--5.) Stokes-I, Q, U, V
            6.) Wavelength Array
            7.) Metadata array, which contains:
                Pointing Lat/Lon, Timestamp, Scintillation, light level, slit position(s)

        Parameters
        ----------
        datacube : numpy.ndarray
            5D reduced Stokes data in shape (4, nslits, ny, nx, nlambda)
        wavelength_array : numpy.ndarray
            2D final wavelength grid in shape (nslits, nlambda)
        hairline_centers : tuple
            Tuple containing the hairline centers that the datacube was registered to.
        filelist : list
            List of Level-0 Science Files. Will need to parse headers for metadata information

        Returns
        -------
        filename : str
        """

        prsteps = [
            'DARK-SUBTRACTION',
            'FLATFIELDING',
            'WAVELENGTH-CALIBRATION',
            'TELESCOPE-MULLER',
            'I->QUV CROSSTALK',
            'FRINGE-CORRECTION'
        ]
        prstep_comments = [
            "firsCal/SSOSoft",
            "firsCal/SSOSoft",
            "1984 FTS Atlas",
            "2010 Measurements",
            "firsCal/SSOSoft",
            "firsCal/SSOSoft"
        ]
        if self.v2q:
            prsteps.append(
                'V->Q CROSSTALK'
            )
            prstep_comments.append(
                "firsCal/SSOSoft"
            )
        if self.v2u:
            prsteps.append(
                'V->U CROSSTALK'
            )
            prstep_comments.append(
                "firsCal/SSOSoft"
            )
        if self.q2v:
            prsteps.append(
                'Q->V CROSSTALK'
            )
            prstep_comments.append(
                "firsCal/SSOSoft"
            )
        if self.u2v:
            prsteps.append(
                'U->V CROSSTALK'
            )
            prstep_comments.append(
                "firsCal/SSOSoft"
            )
        # For the FIRS Virgo 1k, the optical path downstream of the slit is symmetric.
        # There's the 1524 mm collimator, the grating, the 1524 mm collimator again,
        # and two 400 mm lenses to extend the beam.
        # So the camera dy is the same as the slit plate scale.
        # I do not believe this is true for the old visible arm.
        # Currently not an issue, as there are no LCVRs for that channel,
        # But if we ever re-commission it, someone will need to have a switch in here for which arm.
        firs_plate_scale = self.telescope_plate_scale * self.dst_collimator / self.slit_camera_lens
        step_startobs = []
        solar_x = []
        solar_y = []
        rotan = []
        llvl = []
        scin = []
        slitpos = []
        for file in filelist:
            with fits.open(file) as hdul:
                # Values we'll only need once:
                if file == filelist[0]:
                    exptime = hdul[0].header['EXP_TIME']
                    xposure = hdul[0].header['SUMS'] * exptime
                    nsumexp = hdul[0].header['SUMS']
                    # Step size is set by the slit width in the raw file headers.
                    # Actual slit width is user-set
                    stepsize = hdul[0].header['SLITWDTH'] * firs_plate_scale
                    reqmapsize = hdul[0].header['NO. LOOP'] * stepsize * self.nslits
                    actmapsize = stepsize * len(filelist) * self.nslits
                    rsun = hdul[0].header['DST_SDIM'] / 2
                    camera_name = hdul[0].header['CAMERA']
                    # Getting the slit position is non-trivial, as the scanning mirror is before the slit camera lens.
                    # The offset of the FSM is not one-to-one with the displacement on the slit, thanks to the influence
                    # of the lens. Rather than try to estimate the displacement on the slit through the lens
                    # (which involves assumptions about the thickness and composition of the lens),
                    # we can instead back this assumption out of the header, assuming the lateral FSM dx has an eventual
                    # displacement equal to what the system thinks the slit width is, i.e., slw/dx ==> arcsec/mm of disp
                    fsm_dx = hdul[0].header['FSM DX']
                    asec_per_mmdisp = stepsize / fsm_dx
                step_startobs.append(hdul[0].header['OBS_STAR'])
                rotan.append(hdul[0].header['DST_GDRN'] - 13.3)
                llvl.append(hdul[0].header['DST_LLVL'])
                scin.append(hdul[0].header['DST_SEE'])
                center_coord = SkyCoord(
                    hdul[0].header['DST_SLNG'] * u.deg, hdul[0].header['DST_SLAT'] * u.deg,
                    obstime=hdul[0].header['OBS_STAR'], observer='earth', frame=frames.HeliographicStonyhurst
                ).transform_to(frames.Helioprojective)
                solar_x.append(center_coord.Tx.value)
                solar_y.append(center_coord.Ty.value)
                slitpos.append(hdul[0].header['FSM CP'] * asec_per_mmdisp)
        rotan = np.nanmean(rotan)
        date, time = step_startobs[0].split("T")
        date = date.replace("-", "")
        time = str(round(float(time.replace(":", "")), 0)).split(".")[0]
        outname = self.reduced_file_pattern.format(
            date, time, datacube.shape[1] * datacube.shape[3]
        )
        outfile = os.path.join(self.final_dir, outname)

        # Have to account for offset maps, maps that were aborted partway, etc.
        if len(slitpos) % 2 == 0:
            slit_pos_center = (slitpos[int(len(slitpos) / 2) - 1] + slitpos[int(len(slitpos) / 2)]) / 2
            center_x = (solar_x[int(len(solar_x) / 2) - 1] + solar_x[int(len(solar_x) / 2)]) / 2
            center_y = (solar_y[int(len(solar_y) / 2) - 1] + solar_y[int(len(solar_y) / 2)]) / 2
        else:
            slit_pos_center = slitpos[int(len(slitpos) / 2)]
            center_x = solar_x[int(len(solar_x) / 2)]
            center_y = solar_y[int(len(solar_y) / 2)]
        if round(reqmapsize, 1) != round(actmapsize, 1):
            dx = slit_pos_center * np.cos((90 - rotan) * np.pi / 180)
            center_x += dx
            dy = slit_pos_center * np.sin((90 - rotan) * np.pi / 180)
            center_y -= dy  # Note sign
        # Finally, we need to account for the number of slits.
        # Multi-slit units have slit positions that depend on the slit spacing
        if self.nslits != 1:
            res_slit_pos = np.zeros((self.nslits, len(slitpos)))
            if self.nslits % 2 == 1:
                # We don't have a tri-slit unit, but I want one, so I'm writing my code with that in mind.
                # It's aspirational. How shall I to a tri-slit unit aspire?
                for slit in range(-int(self.nslits / 2), int(self.nslits / 2) + 1):
                    res_slit_pos[slit + int(self.nslits / 2), :] = (np.array(slitpos) +
                                                                    slit * self.slit_spacing * firs_plate_scale)
            else:
                # Now the trouble -- with a quad or dual-slit unit, the spacings are off by 0.5
                offsets = np.arange(self.nslits) - 0.5 - (self.nslits / 2 - 1)
                for slit in range(self.nslits):
                    res_slit_pos[slit, :] = np.array(slitpos) + offsets[slit]  * self.slit_spacing * firs_plate_scale
            res_slit_pos = res_slit_pos.flatten()
        else:
            res_slit_pos = np.array(slitpos)
        """
        Decision time. Do we flatten the number of slits.
        Pros:
            - Easier to use a (nstokes, ny, nx, nlambda) data product than an (nstokes, nslits, ny, nx, nlambda) product
            - Single WCS
        Cons:
            - Slit-to-slit differences washed out
            - Time axis will jump as you proceed across the map 
        Given the audience for the final data product, I think probably the best thing to do is to flatten it.
        """

        # Start assembling HDUList
        ext0 = fits.PrimaryHDU()
        ext0.header['DATE'] = (np.datetime64('now').astype(str), "File Creation Date and Time (UTC)")
        ext0.header['ORIGIN'] = 'NMSU/SSOC'
        ext0.header['TELESCOP'] = ('DST', "Dunn Solar Telescope, Sacramento Peak NM")
        ext0.header['INSTRUME'] = ("FIRS", "Facility InfraRed Spectropolarimeter")
        ext0.header['AUTHOR'] = "sellers"
        ext0.header['CAMERA'] = camera_name
        ext0.header['DATA_LEV'] = 1.5

        if self.central_wavelength == 10830:
            ext0.header['WAVEBAND'] = 'Si I 10827, He I 10830'

        ext0.header['STARTOBS'] = step_startobs[0]
        ext0.header['ENDOBS'] = (np.datetime64(step_startobs[-1]) + np.timedelta64(xposure, 'ms')).astype(str)
        ext0.header['BTYPE'] = "Intensity"
        ext0.header['BUNIT'] = 'Corrected DN'
        ext0.header['EXPTIME'] = (exptime, 'ms for single exposure')
        ext0.header['XPOSUR'] = (xposure, 'ms for total coadded exposure')
        ext0.header['NSUMEXP'] = (nsumexp, "Summed images per modulation state")
        ext0.header['NSLITS'] = (self.nslits, "Number of slits in Field Mask")
        ext0.header['SLIT_WID'] = (self.slit_width, "[um] FIRS Slit Width")
        ext0.header['SLIT_ARC'] = (
            round(firs_plate_scale * self.slit_width / 1000, 2),
            "[arcsec, approx] FIRS Slit Width"
        )
        ext0.header['MAP_EXP'] = (round(reqmapsize, 3), "[arcsec] Requested Map Size")
        ext0.header['MAP_ACT'] = (round(actmapsize, 3), "[arcsec] Actual Map Size")

        ext0.header['WAVEUNIT'] = (-10, "10^(WAVEUNIT), Angstrom")
        ext0.header['WAVEREF'] = ("FTS", "Kurucz 1984 Atlas Used in Wavelength Determination")
        ext0.header['WAVEMIN'] = (round(wavelength_array[0, 0], 3), "[AA] Angstrom")
        ext0.header['WAVEMAX'] = (round(wavelength_array[0, -1], 3), "[AA], Angstrom")
        ext0.header['GRPERMM'] = (self.grating_rules, "[mm^-1] Lines per mm of Grating")
        ext0.header['GRBLAZE'] = (self.blaze_angle, "[degrees] Blaze Angle of Grating")
        ext0.header['GRANGLE'] = (self.grating_angle, "[degrees] Operating Angle of Grating")
        ext0.header['SPORDER'] = (self.spectral_order, "Spectral Order")
        grating_params = spex.grating_calculations(
            self.grating_rules, self.blaze_angle, self.grating_angle,
            self.pixel_size, self.central_wavelength, self.spectral_order,
            collimator=self.spectrograph_collimator, camera=self.camera_lens, slit_width=self.slit_width,
        )
        ext0.header['SPEFF'] = (round(float(grating_params['Total_Efficiency']), 3),
                                'Approx. Total Efficiency of Grating')
        ext0.header['LITTROW'] = (round(float(grating_params['Littrow_Angle']), 3), '[degrees] Littrow Angle')
        ext0.header['RESOLVPW'] = (
            round(self.central_wavelength / (0.001 * float(grating_params['Spectrograph_Resolution'])), 0),
            "Maximum Resolving Power of Spectrograph"
        )

        for h in range(len(hairline_centers)):
            ext0.header['HAIRLIN{0}'.format(h)] = (round(float(hairline_centers[h]), 3),
                                                   "Center of Registration Hairline")
        ext0.header['RSUN_ARC'] = rsun
        ext0.header['XCEN'] = (round(center_x, 2), "[arcsec], Solar-X of Map Center")
        ext0.header['YCEN'] = (round(center_y, 2), "[arcsec], Solar-Y of Map Center")
        ext0.header['FOVX'] = (round(actmapsize, 3), "[arcsec], Field-of-view of raster-x")
        ext0.header['FOVY'] = (
            round(datacube.shape[2] * firs_plate_scale * self.pixel_size / 1000 , 3),
            "[arcsec], Field-of-view of raster-y"
        )
        ext0.header['ROT'] = (round(rotan, 3), "[degrees] Rotation from Solar-North")

        for i in range(len(prsteps)):
            ext0.header['PRSTEP{0}'.format(i+1)] = (prsteps[i], prstep_comments[i])
        ext0.header['COMMENT'] = "Full WCS Information Contained in Individual Data HDUs"
        ext0.header['COMMENT'] = "{0} Slits have been flattened".format(self.nslits)

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
            if self.nslits == 1:
                flattened_datacube =  datacube[i, 0, :, :, :]
            else:
                flattened_datacube = np.concatenate(
                    [datacube[i, slit, :, :, :] for slit in range(self.nslits)],
                    axis=1
                )
            ext = fits.ImageHDU(flattened_datacube)
            ext.header['EXTNAME'] = 'STOKES-' + stokes[i]
            ext.header['RSUN_ARC'] = rsun
            ext.header['CDELT1'] = (stepsize, "arcsec")
            ext.header['CDELT2'] = (round(firs_plate_scale * self.pixel_size / 1000 , 3), "arcsec")
            ext.header['CDELT3'] = (wavelength_array[0, 1] - wavelength_array[0, 0], "Angstrom")
            ext.header['CTYPE1'] = 'HPLN-TAN'
            ext.header['CTYPE2'] = 'HPLT-TAN'
            ext.header['CTYPE3'] = 'WAVE'
            ext.header['CUNIT1'] = 'arcsec'
            ext.header['CUNIT2'] = 'arcsec'
            ext.header['CUNIT3'] = 'Angstrom'
            ext.header['CRVAL1'] = (center_x, "Solar-X, arcsec")
            ext.header['CRVAL2'] = (center_y, "Solar-Y, arcsec")
            ext.header['CRVAL3'] = (wavelength_array[0, 0], "Angstrom")
            ext.header['CRPIX1'] = np.mean(np.arange(flattened_datacube.shape[1])) + 1
            ext.header['CRPIX2'] = np.mean(np.arange(flattened_datacube.shape[0])) + 1
            ext.header['CRPIX3'] = 1
            ext.header['CROTA2'] = (rotan, "degrees")
            for h in range(len(hairline_centers)):
                ext.header['HAIRLIN{0}'.format(h)] = (round(hairline_centers[h], 3), "Center of registration hairline")
            fits_hdus.append(ext)

        ext_wvl = fits.ImageHDU(np.mean(wavelength_array, axis=0))

        ext_wvl.header['EXTNAME'] = 'lambda-coordinate'
        ext_wvl.header['BTYPE'] = 'lambda axis'
        ext_wvl.header['BUNIT'] = '[AA]'

        fits_hdus.append(ext_wvl)

        # Metadata extension.
        timestamps = [np.datetime64(t) for t in step_startobs]
        # Repeat nslits times, since we flattened the datacube
        timestamps = np.array(timestamps * self.nslits)
        timedeltas = timestamps - timestamps[0].astype("datetime64[D]")
        timedeltas = timedeltas.astype("timedelta64[ms]").astype(float) / 1000
        columns = [
            fits.Column(
                name='T_ELAPSED',
                format='D',
                unit='SECONDS',
                array=timedeltas,
                time_ref_pos=timestamps[0].astype('datetime64[D]').astype(str)
            ),
            fits.Column(
                name='SLIT_POS',
                format='D',
                unit='ARCSEC',
                array=res_slit_pos
            ),
            fits.Column(
                name='TEL_SOLX',
                format='D',
                unit='ARCSEC',
                array=np.array(solar_x * self.nslits)
            ),
            fits.Column(
                name='TEL_SOLY',
                format='D',
                unit='ARCSEC',
                array=np.array(solar_y * self.nslits)
            ),
            fits.Column(
                name='LIGHTLVL',
                format='D',
                unit='UNITLESS',
                array=np.array(llvl * self.nslits)
            ),
            fits.Column(
                name='TELESCIN',
                format='D',
                unit='ARCSEC',
                array=np.array(scin * self.nslits)
            )
        ]
        ext_met = fits.BinTableHDU.from_columns(columns)
        ext_met.header['EXTNAME'] = 'METADATA'
        ext_met.header['COMMENT'] = 'Columns correspond to x-axis in data extensions'
        ext_met.header['COMMENT'] = 'Columns are shape (ny, nx, nlambda)'

        fits_hdus.append(ext_met)

        fits_hdulist = fits.HDUList(fits_hdus)
        fits_hdulist.writeto(outfile, overwrite=True)

        return outfile

    def firs_analysis(
            self, datacube: np.ndarray, bound_indices: np.ndarray
    ) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
        """
        Performs moment analysis, determines mean circular/linear polarization, and net circular polarization maps
        for each of the given spectral windows. See Martinez Pillet et. al., 2011 for discussion of mean circular
        and linear polarization. See Solanki & Montavon, 1993 for net circular polarization.

        Parameters
        ----------
        datacube : numpy.ndarray
            5D datacube of shape (4, nslits, ny, nx, nlambda)
        bound_indices : numpy.ndarray
            Array of regions for analysis. Shape (2, nslits, numlines)

        Returns
        -------
        parameter_maps : numpy.ndarray
            Array of derived parameter maps. Has shape (nlines, 6, ny, nx*nslits).
            6 is from the number of derived parameters: Intensity, velocity, velocity width, net circular polarization,
            mean circular polarization, mean linear polarization.
        """
        parameter_maps = np.zeros((bound_indices.shape[2], 6, datacube.shape[2], datacube.shape[3] * self.nslits))
        mean_profile = np.nanmean(datacube[0, :, :, :, :], axis=(0, 1, 2))
        wavelength_array = self.tweak_wavelength_calibration(mean_profile)
        if self.analysis_ranges != "default":
            reference_wavelengths = []
            for i in range(bound_indices.shape[2]):
                line_core = spex.find_line_core(
                    mean_profile[int(bound_indices[0, :, i].mean()):int(bound_indices[1, :, i].mean())]
                ) + int(bound_indices[0, :, i].mean())
                reference_wavelengths.append(
                    np.interp(np.arange(len(wavelength_array)), wavelength_array, line_core)
                )
        else:
            reference_wavelengths = self.default_reference_wavelengths

        with tqdm.tqdm(
            total=parameter_maps.shape[0] * parameter_maps.shape[2] * parameter_maps.shape[3],
            desc="Constructing Derived Parameter Maps"
        ) as pbar:
            for i in range(parameter_maps.shape[0]):
                for y in range(parameter_maps.shape[2]):
                    for slit in range(self.nslits):
                        for x in range(datacube.shape[3]):
                            spectral_profile = datacube[
                                0, slit, y, x, int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])
                            ]
                            intens, vel, wid = spex.moment_analysis(
                                wavelength_array[int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                spectral_profile,
                                reference_wavelengths[i]
                            )
                            parameter_maps[i, 0:3, y, slit*datacube.shape[3] + x] = np.array([intens, vel, wid])
                            pseudo_continuum = np.nanmean(
                                spectral_profile.take([-2, -1, 0, 1])
                            )
                            # Net V
                            parameter_maps[i, 3, y, slit*datacube.shape[3] + x] = pol.net_circular_polarization(
                                datacube[3, slit, y, x, int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                wavelength_array[int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])]
                            )
                            # Mean V
                            parameter_maps[i, 4, y, slit*datacube.shape[3] + x] = pol.mean_circular_polarization(
                                datacube[3, slit, y, x, int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                wavelength_array[int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                reference_wavelengths[i],
                                pseudo_continuum
                            )
                            # Mean QU
                            parameter_maps[i, 5, y, slit*datacube.shape[3] + x] = pol.mean_linear_polarization(
                                datacube[1, slit, y, x, int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                datacube[2, slit, y, x, int(bound_indices[0, slit, i]):int(bound_indices[1, slit, i])],
                                pseudo_continuum
                            )
                            pbar.update(1)

        return parameter_maps, reference_wavelengths, mean_profile, wavelength_array

    def package_analysis(
            self, analysis_maps: np.ndarray, rwvls: list, indices: np.ndarray,
            mean_profile: np.ndarray, wavelength_array: np.ndarray, reference_file:str
    ) -> str:
        """
        Write FIRS first-order analysis maps to FITS file

        Parameters
        ----------
        analysis_maps : numpy.ndarray
        rwvls : list
        indices : numpy.ndarray
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
        outfile = ""
        for i in range(len(rwvls)):
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
            if self.analysis_ranges == "default":
                # If we're using default ranges, we must also be at 10830
                ext0.header['WAVEBAND'] = self.default_analysis_names[i]
            ext0.header["WAVEMIN"] = (round(wavelength_array[int(indices[0, 0, i])], 3), "Lower Bound for Analysis")
            ext0.header["WAVEMAX"] = (round(wavelength_array[int(indices[1, 0, i])], 3), "Upper Bound for Analysis")
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
            if self.analysis_ranges == "default":
                linename = self.default_analysis_names[i].replace(" ", "_")
            else:
                linename = "LINE{0}".format(i)
            outname = self.parameter_map_pattern.format(
                date,
                time,
                linename,
                round(ext0.header['WAVEMIN'], 2),
                round(ext0.header['WAVEMAX'], 2)
            )
            outfile = os.path.join(self.final_dir, outname)
            fits_hdu_list = fits.HDUList(fits_hdus)
            fits_hdu_list.writeto(outfile, overwrite=True)
        return

    def package_crosstalks(
            self,
            i2quv_crosstalks: np.ndarray, internal_crosstalks: np.ndarray,
            index: int, repeat: int
    ) -> str:
        """

        Parameters
        ----------
        i2quv_crosstalks : numpy.ndarray
            Array of shape (3, 2, nslits, ny, nx) of I->QUV parameters
            2-axis is because the crosstalks are a linear fit
        internal_crosstalks : numpy.ndarray
            Array of shape (4, nslits, ny, nx) of internal crosstalks.
            Order is V->Q, V->U, Q->V, U->V
        index : int
            Observation number
        repeat : int
            Index of map within an observation

        Returns
        -------
        crosstalk_file : str
            Name of file where crosstalk parameters are stored

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
        ext0.header['Q2V'] = (self.q2v, "True=by slit, Full=by slit and row")
        ext0.header['U2V'] = (self.u2v, "True=by slit, Full=by slit and row")
        ext0.header['COMMENT'] = "Crosstalks applied in order:"
        ext0.header['COMMENT'] = "I->QUV"
        ext0.header['COMMENT'] = "V->Q"
        ext0.header['COMMENT'] = "V->U"
        ext0.header['COMMENT'] = "Q->V"
        ext0.header['COMMENT'] = "U->V"

        i2quv_ext = fits.ImageHDU(i2quv_crosstalks)
        i2quv_ext.header['EXTNAME'] = "I2QUV"
        i2quv_ext.header[""] = "<QUV> = <QUV> - (coef[0]*[0, 1, ... nlambda] + coef[1]) * I"

        v2q_ext = fits.ImageHDU(internal_crosstalks[0])
        v2q_ext.header['EXTNAME'] = "V2Q"
        v2q_ext.header[""] = "Q = Q - coef*V"

        v2u_ext = fits.ImageHDU(internal_crosstalks[1])
        v2u_ext.header['EXTNAME'] = "V2U"
        v2u_ext.header[""] = "U = U - coef*V"

        q2v_ext = fits.ImageHDU(internal_crosstalks[2])
        q2v_ext.header['EXTNAME'] = "Q2V"
        q2v_ext.header[""] = "V = V - coef*Q"

        u2v_ext = fits.ImageHDU(internal_crosstalks[3])
        u2v_ext.header['EXTNAME'] = "U2V"
        u2v_ext.header[""] = "V = V - coef*U"

        hdul = fits.HDUList([ext0, i2quv_ext, v2q_ext, v2u_ext, q2v_ext, u2v_ext])
        filename = "FIRS_MAP_{0}_REPEAT_{1}_CROSSTALKS.fits".format(index, repeat)
        crosstalk_file = os.path.join(self.final_dir, filename)
        hdul.writeto(crosstalk_file, overwrite=True)

        return
