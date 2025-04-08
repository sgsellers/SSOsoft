import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as scinterp
import scipy.io as scio
import scipy.ndimage as scind
import scipy.optimize as scopt
from matplotlib.widgets import Slider

"""
This file contains generalized helper functions for Polarimeter reductions.
ssosoft.spectral.spectraTools contains spectrograph tools, while this module contains 
additional functions used in polarimetry. 
While actual reduction packages are instrument-specific, these functions represent
general-use bits of code that can be reused at a low level.
See docstrings on each function for a comprehensive explanation of the use of each function.

At present moment, there is significant duplication between items in spectraTools, items
in spinorCal, and this file. A future refactor will redefine the boundaries between these
various modules. That refactor is slated for after the firsCal pipeline is stable and live.
"""

def linear_retarder(axis_angle: float, retardance: float) -> np.ndarray:
    """Returns Mueller matrix for a linear retarder.
    At some point, I'll come back to this and add in kwargs for dichroism

    Parameters
    ----------
    axis_angle : float
        Angle of fast axis in radians
    retardance : float
        Degree of retardance in radians

    Returns
    -------
    ret_mueller : numpy.ndarray
        Mueller matrix of retarder
    """

    c2 = np.cos(2*axis_angle)
    s2 = np.sin(2*axis_angle)

    ret_mueller = np.array(
        [
            [1, 0, 0, 0],
            [0, c2**2 + s2**2 * np.cos(retardance), c2 * s2 * (1 - np.cos(retardance)), -s2 * np.sin(retardance)],
            [0, c2 * s2 * (1 - np.cos(retardance)), s2**2 + c2**2 * np.cos(retardance), c2 * np.sin(retardance)],
            [0, s2 * np.sin(retardance), -c2 * np.sin(retardance), np.cos(retardance)]
        ]
    )

    return ret_mueller


def rotation_mueller(phi: float, reverse: bool=False) -> np.ndarray:
    """Sets up a Mueller matrix for rotation about the optical axis.
    Used for transforming between reference frames

    Parameters
    ----------
    phi : float
        Rotation angle in radians
    reverse: float
        Flips the sign on the sins. The product of the reversed and natural
        rotation matrix is the identity matrix

    Returns
    -------
    rotationMueller
    """

    neg = -1 if reverse else 1

    rotation_mueller_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(2*phi), neg * np.sin(2*phi), 0],
        [0, neg * -np.sin(2*phi), np.cos(2*phi), 0],
        [0, 0, 0, 1]
    ])

    return rotation_mueller_matrix


def linear_analyzer_polarizer(axis_angle: float, px: float=1, py: float=0) -> np.ndarray:
    """Returns the Mueller matrix for a non-ideal linear polarizer

    Parameters
    ----------
    axis_angle : float
        Angle of the polarization axis in radians
    px : float
        Efficiency in X
    py : float
        Efficiency in Y

    Returns
    -------
    pol_mueller : numpy.ndarray
        Mueller matrix of linear polarizer analyzer
    """

    c2 = np.cos(2*axis_angle)
    s2 = np.sin(2*axis_angle)

    alpha = px**2 + py**2
    beta = (px**2 - py**2)/alpha
    gamma = 2*px*py/alpha

    pol_mueller = (alpha/2) * np.array(
        [
            [1, beta * c2, beta * s2, 0],
            [beta * c2, c2**2 + gamma*s2**2, (1-gamma) * c2*s2, 0],
            [beta * s2, (1 - gamma) * c2*s2, s2**2 + gamma*c2**2, 0],
            [0, 0, 0, gamma]
        ]
    )
    return pol_mueller


def mirror(rs_over_rp: float, retardance: float) -> np.ndarray:
    """ Returns the Mueller matrix of a mirror

    Parameters
    ----------
    rs_over_rp : float
        Ratio of s- and p-polarization state reflectance
    retardance : float
        Degree of retardance in radians

    Returns
    -------
    pol_mirror : numpy.ndarray
        Mueller matrix of a mirror
    """

    pol_mirror = np.array([
        [(1 + rs_over_rp)/2., (1 - rs_over_rp)/2., 0, 0],
        [(1 - rs_over_rp)/2., (1 + rs_over_rp)/2., 0, 0],
        [0, 0, np.sqrt(rs_over_rp)*np.cos(retardance), np.sqrt(rs_over_rp)*np.sin(retardance)],
        [0, 0, -np.sqrt(rs_over_rp)*np.sin(retardance), np.sqrt(rs_over_rp)*np.cos(retardance)]
    ])

    return pol_mirror


def matrix_inversion(input_array: np.ndarray, output_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple matrix inversion and solve. Used in constructing Mueller matrix for optical train

    Parameters
    ----------
    input_array : numpy.ndarray
        Input Stokes Vector
    output_array : numpy.ndarray
        Output Stokes vector

    Returns
    -------
    error : numpy.ndarray
        Chi squared array
    matrix : numpy.ndarray
        Solution matrix
    """

    d = input_array.T @ input_array
    d_inv = np.linalg.inv(d)
    a = d_inv @ input_array.T
    matrix = a @ output_array
    chi = np.nansum((output_array-input_array@matrix)**2)/input_array.size

    error = np.array([a[:, i]**2 for i in range(a.shape[1])])
    error = np.sqrt(error * chi)
    matrix = matrix.T

    return error, matrix


def check_mueller_physicality(mueller_mtx: np.ndarray) -> tuple[bool, float, float]:
    """Checks if an input matrix is a "reasonable" Mueller matrix. From J.C. del Toro Iniesta:
        - First Element must be non-negative
        - First row must be a physically meaningful Stokes vector, i.e.,
            M00 >= sqrt(M01**2 + M02**2 + M03**2)
        - First Column is a physically meaningful Stokes vector
        C. Beck's algorithm does this by applying a Stokes vector defined between 0--2pi on
        a grid of theta and phi where:
        - I = 1
        - Q = cos(phi) * sin(theta)
        - U = sin(phi) * sin(theta)
        - V = cos(theta)

    Parameters
    ----------
    mueller_mtx : numpy.ndarray
        4x4 Mueller matrix to check for physicality

    Returns
    -------
    bool
        False if matrix is found to unphysical, True otherwise
    master_imin : float
        Minimum value for the Stokes-I value of the output vector. Unphysical if < 0
    master_pmin : float
        Minimum value for I**2 - (Q**2 + U**2 + V**2). Unphysical if < 0

    """

    if mueller_mtx[0, 0] < 0:
        return False

    theta = np.arange(0, 1, 1/400) * 2 * np. pi
    phi = np.arange(0, 1, 1/1000) * 2 * np.pi

    master_pmin = 1
    master_imin = 10

    for i in range(len(theta)):
        stokes_vec = np.array([
            np.ones(len(phi)), # I
            np.cos(phi) * np.sin(theta[i]), # Q
            np.sin(phi) * np.sin(theta[i]), # U
            np.cos(theta[i]) * np.ones(len(phi)) # V
        ])

        stokes_prime = np.array([
            mueller_mtx @ stokes_vec[:, i] for i in range(len(phi))
        ]).T

        # Cast to 16-bit to avoid floating point errors in ideal Mueller matrices
        pvec = (stokes_prime[0, :]**2 - np.nansum(stokes_prime[1:, :]**2, axis=0)).astype(np.float16)

        master_pmin = np.nanmin(pvec) if np.nanmin(pvec) < master_pmin else master_pmin
        master_imin = np.nanmin(stokes_prime[0, :]) if np.nanmin(stokes_prime[0, :]) < master_imin else master_imin

        if (master_pmin < 0) or (master_imin < 0):
            return False, master_imin, master_pmin

    return True, master_imin, master_pmin


def get_dst_matrix(
        telescope_geometry: list, central_wavelength: float, reference_frame: float, matrix_file: str
) -> np.ndarray:
    """
    Gets DST telescope matrix from IDL save (2010 matrix) or numpy save (TBD, hopefully we measure it in the future)
    file. Returns the Mueller matrix of the telescope from these measurements.

    Parameters
    ----------
    telescope_geometry : numpy.ndarray
        3-element vector containing the coelostat azimuth, coelostat elevation, and Coude table angle
    central_wavelength : float
        In angstrom, wavelength to interpolate measured values to
    reference_frame : float
        In degrees, the orientation of the reference frame relative to the telescope matrix.
    matrix_file : str
        Path to telescope matrix file

    Returns
    -------
    tmatrix : numpy.ndarray
        4x4 Mueller matrix of telescope parameters
    """

    filename, filetype = os.path.splitext(matrix_file)
    if "idl" in filetype:
        txparams = scio.readsav(matrix_file)
    else:
        txparams = scio.readsav(matrix_file)

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
    ref_frame_orientation = reference_frame * np.pi / 180

    wvls = txparams['tt'][5::7]
    entrance_window_retardance = scinterp.interp1d(
        wvls, txparams['tt'][6::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength) * np.pi / 180
    exit_window_retardance = scinterp.interp1d(
        wvls, txparams['tt'][7::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength) * np.pi / 180
    coelostat_reflectance = scinterp.interp1d(
        wvls, txparams['tt'][8::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength)
    coelostat_retardance = scinterp.interp1d(
        wvls, txparams['tt'][9::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength) * np.pi / 180
    primary_reflectance = scinterp.interp1d(
        wvls, txparams['tt'][10::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength)
    primary_retardance = scinterp.interp1d(
        wvls, txparams['tt'][11::7], kind='linear', fill_value='extrapolate'
    )(central_wavelength) * np.pi / 180

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

    entrance_window_mueller = linear_retarder(
        entrance_window_orientation, entrance_window_retardance
    )
    elevation_mueller = mirror(
        coelostat_reflectance, coelostat_retardance
    )
    azel_rotation_mueller = rotation_mueller(phi_elevation)
    azimuth_mueller = mirror(
        coelostat_reflectance, coelostat_retardance
    )
    azvert_rotation_mueller = rotation_mueller(phi_azimuth)
    primary_mueller = mirror(
        primary_reflectance, primary_retardance
    )
    exit_window_mueller = linear_retarder(
        exit_window_orientation, exit_window_retardance
    )
    refframe_rotation_mueller = rotation_mueller(
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


def spherical_coordinate_transform(telescope_angles: list, site_latitude: float) -> list:
    """
    Transforms from telescope pointing to parallatic angle using the site latitude

    Parameters
    ----------
    telescope_angles : list
        List of telescope angles. In order, these should be (telescope_azimuth, telescope_elevation)
    site_latitude : float
        In degrees, the latitude of the telescope site.

    Returns
    -------
    coordinate_angles : list
        List of telescope angles. In order, these will be (hour_angle, declination, parallatic angle)

    """

    sin_lat = np.sin(site_latitude * np.pi / 180.)
    cos_lat = np.cos(site_latitude * np.pi / 180.)

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


def internal_crosstalk_2d(base_image: np.ndarray, contamination_image: np.ndarray) -> float:
    """
    Determines a single crosstalk value for a pair of 2D images.
    Minimizes the linear correlation between:
        baseImage - crosstalk_value * contaminationImage
            and
        contaminationImage
    Essentially, finds the crosstalk_value that makes baseImage and contaminationImage least similar
    The v2qu_crosstalk function should be used for individual vectors (uses cosine similarity,
    which scales to 2D poorly). This should be used as an initial guess, with v2qu_crosstalk providing
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


def i2quv_crosstalk(stokes_i: np.ndarray, stokes_quv: np.ndarray) -> np.ndarray:
    """
    Corrects for Stokes-I => QUV crosstalk. In older DST pipelines, this was done by
    taking the ratio of a continuum section in I, and in QUV, then subtracting
    QUV_nu = QUV_old - ratio * I.

    We're going to take a slightly different approach. Instead of a single ratio value,
    we'll use a line, mx+b, such that QUV_nu = QUV_old - (mx+b)*I.
    We'll solve for m, b such that a second line m'x+b' fit to QUV_nu has m'=b'=0

    This should solve issues that we saw in old pipelines where there was a significant
    slope to reduced stokes vectors, and a wavelength-dependant degree of contamination in
    line cores.

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


def v2qu_crosstalk(stokes_v: np.ndarray, stokes_qu: np.ndarray) -> np.ndarray:
    """
    Contrary to I->QUV crosstalk, we want the Q/U profiles to be dissimilar to V.
    Q in particular is HEAVILY affected by crosstalk from V.
    Take the assumption that QU = QU - aV, where 'a' is chosen such that the error between
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

    fit_result = scopt.least_squares(
        error_function,
        x0=0,
        args=[stokes_v[50:-50], stokes_qu[50:-50]],
        bounds=[-0.5, 0.5]
    )

    v2qu_crosstalk_value = fit_result.x

    corrected_qu = stokes_qu - v2qu_crosstalk_value * stokes_v

    return corrected_qu, v2qu_crosstalk_value


def fourier_fringe_correction(
        fringe_cube: np.ndarray, freqency_cutoff: float,
        smoothing: int or tuple[int, int], dlambda: float
) -> np.ndarray:
    """
    Performs simple masking in Fourier space for fringe correction.
    Returns a normalized fringe template for division

    Parameters
    ----------
    fringe_cube: numpy.ndarray
        Datacube with fringes
    freqency_cutoff:
        Cutoff frequency. Everything outside this is considered fringe
    smoothing: tuple of int or int
        Pass-through to scipy.ndimage.median_filter. Smooths datacube in spatial/spectral dimension
    dlambda: float
        Wavelength resolution for use in determining cutoff freqs.

    Returns
    -------
    fringe_template: numpy.ndarray
        Normalized fringes from fringe_cube. Dividing fringe_cube by this should yield a non-fringed image.
    """
    smooth_cube = scind.median_filter(fringe_cube, smoothing)
    fringe_template = np.zeros(smooth_cube.shape)
    fftfreqs = np.fft.fftfreq(smooth_cube.shape[1], dlambda)
    lowcut = fftfreqs <= -freqency_cutoff
    highcut = fftfreqs >= freqency_cutoff
    for i in range(fringe_template.shape[0]):
        prof = np.fft.fft(smooth_cube[i, :])
        prof[lowcut] = 0
        prof[highcut] = 0
        fringe_template[i, :] = np.real(np.fft.ifft(prof))
    return fringe_template


def select_fringe_freq(wvl: np.ndarray, profile: np.ndarray, init_period: float) -> float:
    """
    Allows user to adjust fringe frequencies and select best cutoff for template.

    Parameters
    ----------
    wvl: numpy.ndarray
        Wavelength array
    profile: numpy.ndarray
        Spectral profile
    init_period: float
        Initial periodicities

    Returns
    -------
    period_slider.value: float
        Value for cut
    """

    def fourier_cutter(wave, prof, freq):
        ft = np.fft.fft(prof)
        fq = np.fft.fftfreq(len(prof), wave[1] - wave[0])
        ft[fq >= 1 / freq] = 0
        ft[fq <= -1 / freq] = 0
        return np.real(np.fft.ifft(ft))

    fig, ax = plt.subplots()
    static, = ax.plot(wvl, profile, lw=2, label='Original')
    fourier, = ax.plot(wvl, fourier_cutter(wvl, profile, init_period), lw=2, label='Fringe Template')
    corr, = ax.plot(
        wvl,
        profile / (fourier_cutter(
            wvl, profile, init_period
        ) / np.nanmedian(fourier_cutter(wvl, profile, init_period))) + np.nanmedian(profile) / 4,
        lw=2, label='Corrected')
    fig.subplots_adjust(bottom=0.25)
    ax.set_xlabel("Wavelength")
    ax.set_title("Set desired period, then close window")
    ax.legend(loc='lower right')
    axpd = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    period_slider = Slider(ax=axpd, label='Period', valmin=1e-5, valmax=6, valinit=init_period)

    def update(val):
        fourier.set_ydata(fourier_cutter(wvl, profile, period_slider.val))
        corr.set_ydata(profile / (
                fourier_cutter(
                    wvl, profile, period_slider.val
                ) / np.nanmedian(fourier_cutter(wvl, profile, period_slider.val))
        ) + np.nanmedian(profile) / 4)
        fig.canvas.draw_idle()

    period_slider.on_changed(update)
    plt.show()
    return 1 / period_slider.val