import warnings
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as nrec
import numpy.polynomial.polynomial as npoly
import numpy.typing as npt
import scipy.integrate as scint
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import scipy.optimize as scopt
import scipy.signal as scig
import tqdm
from astropy.constants import c
from matplotlib.widgets import Slider

c_kms = c.value / 1e3

"""
This file contains generalized helper functions for Spectrograph/Spectropolarimeter
reductions. While the actual reduction packages are specific to the instrument, these
functions represent bits of code that can be reused at a low level. See the individual
docstrings for a comprehensive explanation of each function given below.
"""

def count_recursive_calls(func):
    """Fancy wrapper for counting number of recursions.
    Used in spectraTools.detect_beams_hairlines
    """

    def wrapper(*args, **kwargs):
        wrapper.num_calls += 1
        return func(*args, **kwargs)

    wrapper.num_calls = 0
    return wrapper


def find_nearest(array, value):
    """
    Determines the index of the closest value in the array to the provided value
    :param array: array-like
        array
    :param value: float
        value
    :return idx: int
        index
    """
    array = np.nan_to_num(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_line_core(profile, wvl=None):
    """
    Uses the Fourier Phase method to determine the position of a spectral line core.
    See Schlichenmaier & Schmidt, 2000 for details on this application.
    It's fast, insensitive to noise, but does require a very narrow range.

    :param profile: array-like
        The line profile to determine the centroid of. Assumed to be 1-D
    :param wvl: array-like
        Optional, the wavelengths corresponding to the line profile.
        If wvl is given, the returned value will be a wavelength
    :return center: float
        The position of the line core
        If wvl is not given, in pixel number
        Otherwise, in wavelength space.
    """

    profile_fft = np.fft.fft(profile)
    center = -np.arctan(
        np.imag(profile_fft[1]) / np.real(profile_fft[1])
    ) / (2 * np.pi) * len(profile) + (len(profile) / 2.)
    if wvl is not None:
        center = scinterp.interp1d(np.arange(len(wvl)), wvl, kind='linear')(center)
    return center


def linear_retarder(axis_angle, retardance):
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


def rotation_mueller(phi, reverse=False):
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


def linear_analyzer_polarizer(axis_angle, px=1, py=0):
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


def mirror(rs_over_rp, retardance):
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


def matrix_inversion(input_array, output_array):
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


def check_mueller_physicality(mueller_mtx):
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


def fts_window(wavemin, wavemax, atlas='FTS', norm=True, lines=False):
    """
    For a given wavelength range, return the solar reference spectrum within that range.

    :param wavemin: float
        Blue end of the wavelength range
    :param wavemax: float
        Red end of the wavelength range
    :param atlas: str
        Which atlas to use. Currently accepts "Wallace" and "FTS"
        Wallace uses the 2011 Wallace updated atlas
        FTS uses the 1984 FTS atlas
    :param norm: bool
        If False, and the atlas is set to "FTS", will return the solar irradiance.
        This includes the blackbody curve, etc.
    :param lines: bool
        If True, returns additional arrays denoting line centers and names
        within the wavelength range.
    :return wave: array-like
        Array of wavelengths
    :return spec: array-like
        Array of spectral values
    :return line_centers: array-like, optional
        Array of line center positions
    :return line_names: array-like, optional
        Array of line names
    """

    def read_data(path, fname) -> np.array:
        with resources.path(path, fname) as df:
            return np.load(df)

    if atlas.lower() == 'wallace':
        if (wavemax <= 5000.) or (wavemin <= 5000.):
            atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_290-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_290-1000nm_Observed.npy')
        else:
            atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_500-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_500-1000nm_Corrected.npy')
    else:
        atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Wavelengths.npy')
        if norm:
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Atlas.npy')
        else:
            warnings.warn("Using solar irradiance (i.e., not normalized)")
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Irradiance.npy')
            atlas_spectrum *= 462020 # Conversion to erg/cm2/s/nm
            atlas_spectrum /= 10 # Conversion to erg/cm2/s/Angstrom

    idx_lo = find_nearest(atlas_angstroms, wavemin) - 5
    idx_hi = find_nearest(atlas_angstroms, wavemax) + 5

    wave = atlas_angstroms[idx_lo:idx_hi]
    spec = atlas_spectrum[idx_lo:idx_hi]

    if lines:
        line_centers_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_CentralWavelengths.npy'
        )
        line_names_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_IonNames.npy'
        )
        line_selection = (line_centers_full < wavemax) & (line_centers_full > wavemin)
        line_centers = line_centers_full[line_selection]
        line_names = line_names_full[line_selection]
        return wave, spec, line_centers, line_names
    else:
        return wave, spec


def rolling_median(data, window):
    """
    Simple rolling median function, rolling by the central value.
    Preserves the edges to provide an output array of the same shape as the input.
    I wasn't a fan of any of the prewritten rolling median function edge behaviours.
    Hence this. Kind of a kludge, tbh.

    :param data: array-like
        Array of data to smooth
    :param window: int
        Size of median window.
    :return rolled: array-like
        Rolling median of input array
    """

    rolled = np.zeros(len(data))
    half_window = int(window / 2)
    if half_window >= 4:
        for i in range(half_window):
            rolled[i] = np.nanmedian(data[i:i + 1])
            rolled[-(i + 1)] = np.nanmedian(data[(-(i + 4)):(-(i + 1))])
    else:
        rolled[:half_window] = data[:half_window]
        rolled[-(half_window + 1):] = data[-(half_window + 1):]
    for i in range(len(data) - window):
        rolled[half_window + i] = np.nanmedian(data[i:half_window + i])
    return rolled


def select_lines_singlepanel(array, nselections, fig_name="Popup Figure"):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From the plot of a 1D array.

    :param array: array-like
        Array to plot and select from
    :param nselections: int
        Number of expected selections
    :param fig_name: str
        Name of the produced figure
    :return xvals: array-like
        Array of selected x-values with length nselections
    """

    fig = plt.figure(fig_name)
    ax = fig.add_subplot(111)
    ax.set_title("Select " + str(nselections) + " Positions, then Click Again to Exit")
    spectrum, = ax.plot(array)

    xvals = []

    def onselect(event):
        if len(xvals) < nselections:
            xcd = event.xdata
            ax.axvline(xcd, c='C1', linestyle=':')
            fig.canvas.draw()
            xvals.append(xcd)
            print("Selected: " + str(xcd))
        else:
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect)
    plt.show(block=True)
    return np.array(xvals)


def select_lines_singlepanel_unbound_xarr(array, xarr=None,
                                          fig_name="Popup Figure!", n_selections=None, vlines=None):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From the plot of a 1D array.

    :param array: array-like
        Array to plot and select from
    :param xarr: array-like, optional
        Optional x array to plot against.
    :param fig_name: str
        Name of the figure window
    :param n_selections: None or int, optional
        If set, closes figure after nSelections are made
    :param vlines: None or array-like, optional
        If set, draws axvlines at all specified positions
    :return xvals: array-like
        Array of selected x-values with length nselections
    """

    fig = plt.figure(fig_name)
    ax = fig.add_subplot(111)
    ax.set_title("Select Positions, then close window")
    if xarr is None:
        xarr = np.arange(len(array))
    spectrum, = ax.plot(xarr, array)
    if vlines is not None:
        for line in vlines:
            ax.axvline(line, linestyle=":", c='C0')

    xvals = []

    def onselect(event):
        xcd = event.xdata
        ax.axvline(xcd, c='C1', linestyle=':')
        fig.canvas.draw()
        xvals.append(find_nearest(xarr, xcd))
        print("Selected: " + str(xcd))
        if (n_selections is not None) and (len(xvals) >= n_selections):
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect)
    plt.show()
    return np.array(xvals)


def select_spans_singlepanel(array, xarr=None, fig_name="Popup Figure!", n_selections=None):
    """
    Matplotlib-based function to select x range from the plot of a 1D array.

    :param array: array-like
        Array to plot and select from
    :param xarr: array-like
        Optional x array to plot against.
    :param fig_name:
    :param n_selections: None, int, optional
        If provided, closes figure after nSelections are made
    :return xvals: numpy.ndarray
        Array of selected x-spans with shape (2, nselections)
    """
    fig = plt.figure(fig_name)
    ax = fig.add_subplot(111)
    ax.set_title("Click to select min and max of spectral regions. Close window when done.")
    if xarr is None:
        xarr = np.arange(len(array))
    spectrum, = ax.plot(xarr, array)

    xvals = []
    n = 1

    def onselect(event):
        nonlocal n
        xcd = event.xdata
        xvals.append(find_nearest(xarr, xcd))
        ax.axvline(xcd, c='C' + str(n), linestyle=':')

        if (len(xvals) % 2 == 0) & (len(xvals) != 0):
            ax.axvspan(xarr[xvals[-2]], xarr[xvals[-1]], fc='C' + str(n), alpha=0.3)
            n += 1
        fig.canvas.draw()
        print("Selected: " + str(xcd))

        if (n_selections is not None) and (n >= int(n_selections)):
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect)
    plt.show(block=True)
    return np.sort(np.array(xvals).reshape(int(len(xvals) / 2), 2))


def select_spans_doublepanel(array1: np.ndarray, array2: np.ndarray, nselections: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Matplotlib-based function to select ranges from two arrays

    Parameters
    ----------
    array1 : np.ndarray
        Top panel array
    array2 : np.ndarray
        Bottom panel array
    nselections : int
        Number of spans to select on each plot

    Returns
    -------
    xvals_top : np.ndarray
        Span ranges for top panel
    xvals_bottom : np.ndarray
        Span ranges for bottom panel
    """
    fig = plt.figure("Select {0} min/max ranges on each panel".format(nselections))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.suptitle("Select {0} min/max ranges on each panel".format(nselections))
    spectrum1, = ax1.plot(array1)
    spectrum2, = ax2.plot(array2)

    xarr_top = np.arange(len(array1))
    xarr_bottom = np.arange(len(array2))

    xvals_top = []
    xvals_bottom = []

    n_top = 1
    n_bottom = 1

    def onselect_2panel(event):
        nonlocal n_top, n_bottom
        if event.inaxes == ax1:
            if len(xvals_top) < int(nselections * 2):
                xcd = event.xdata
                ax1.axvline(xcd, c='C{0}'.format(n_top), linestyle=':')
                if (len(xvals_top) % 2 == 0) and (len(xvals_top) != 0):
                    ax1.axvspan(
                        xarr_top[xvals_top[-2]], xarr_top[xvals_top[-1]],
                        fc='C{0}'.format(n_top), alpha=0.3
                    )
                    n_top += 1
                fig.canvas.draw()
                xvals_top.append(int(xcd))
                print("Selected: " + str(xcd))
        elif event.inaxes == ax2:
            if len(xvals_bottom) < int(nselections * 2):
                xcd = event.xdata
                ax2.axvline(xcd, c='C{0}'.format(n_bottom), linestyle=':')
                if (len(xvals_bottom) % 2 == 0) and (len(xvals_bottom) != 0):
                    ax1.axvspan(
                        xarr_bottom[xvals_bottom[-2]], xarr_bottom[xvals_bottom[-1]],
                        fc='C{0}'.format(n_bottom), alpha=0.3
                    )
                    n_bottom += 1
                fig.canvas.draw()
                xvals_bottom.append(int(xcd))
                print("Selected: " + str(xcd))

        if (len(xvals_top) >= int(nselections * 2)) & (len(xvals_bottom) >= int(nselections * 2)):
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect_2panel)
    plt.show(block=True)
    return (
        np.sort(np.array(xvals_top).reshape(nselections, 2)).astype(int),
        np.sort(np.array(xvals_bottom).reshape(nselections, 2)).astype(int)
    )


def select_lines_doublepanel(array1, array2, nselections):
    """
    Matplotlib-based function to select an x-value, or series of x-values
    From two plots of 1D arrays.

    :param array1: array-like
        First array to plot and select from
    :param array2: array-like
        Second array to plot and select from.
    :param nselections: int
        Number of expected selections
        NOTE: It is assumed that nselections are split evenly between
        array1 and array2. Please don't try to break this.
    :return xvals1: array-like
        Array of selected x-values from array1 with length nselections/2
    :return xvals2: array-like
        Array of selected x-values from array2 with length nselections/2
    """

    fig = plt.figure("Select " + str(int(nselections / 2)) + " Positions on each plot")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.suptitle("Select " + str(int(nselections / 2)) + " Positions on each plot")
    spectrum1, = ax1.plot(array1)
    spectrum2, = ax2.plot(array2)

    xvals1 = []
    xvals2 = []

    def onselect_2panel(event):
        if event.inaxes == ax1:
            if len(xvals1) < int(nselections / 2):
                xcd = event.xdata
                ax1.axvline(xcd, c='C1', linestyle=':')
                fig.canvas.draw()
                xvals1.append(int(xcd))
                print("Selected: " + str(xcd))
        elif event.inaxes == ax2:
            if len(xvals2) < int(nselections / 2):
                xcd = event.xdata
                ax2.axvline(xcd, c='C1', linestyle=':')
                fig.canvas.draw()
                xvals2.append(int(xcd))
                print("Selected: " + str(xcd))

        if (len(xvals1) >= int(nselections / 2)) & (len(xvals2) >= int(nselections / 2)):
            fig.canvas.mpl_disconnect(conn)
            plt.close(fig)

    conn = fig.canvas.mpl_connect('button_press_event', onselect_2panel)
    plt.show(block=True)
    return np.array(xvals1, dtype=np.int_), np.array(xvals2, dtype=np.int_)


def spectral_skew(image, order=2, slit_reference=0.25):
    """
    Adaptation of the deskew1.pro function included in firs-soft, spinor-soft.
    In the y-direction of the input array, it determines the position of the line
    core, then fits a polynomial of "order" (default 2) to these shifts, as well as
    a line. It then normalizes the shifts relative to the position of the linear fit
    at "slit_reference" along the slit. It then returns these shifts for use with
    scipy.ndimage.shift.

    Ideally, you'd give this function an image that's narrow in wavelength space.
    Possibly several times, then average the relative shifts, or create a profile
    of shifts in the wavelength direction of the wider image.
    :param image: array-like
        2D array of data to find offsets to. Assumes the slit is in the y-direction,
        with wavelength in the x-direction. Also assumes that the hairlines are masked
        out as NaN.
    :param order: int
        Order of polynomial to determine the relative shift profile along the slit
    :param slit_reference: float
        Fractional height of the image to determine shifts relative to. Default in
        deskew1.pro was 0.25. My quick tests showed 0.5 working better. It's a keyword arg.
    :return shifts: array-like
        1D array of shifts along the slit for the provided line core.
    """

    core_positions = np.zeros(image.shape[0])
    for i in range(len(core_positions)):
        core_positions[i] = find_line_core(image[i, :])

    # We assume that the hairlines are NaN slices.
    # These return NaN from find_line_core
    # We need to cut the NaN values, while preserving the spacing along y.
    yrange = np.arange(image.shape[0])
    nancut = np.nan_to_num(core_positions) != 0
    core_positions_tofit = core_positions[nancut]
    yrange_tofit = yrange[nancut]

    # I very earnestly miss when it was just numpy.polyfit
    # And not numpy.polynomial.polynomial.Polynomial.fit().convert().coef
    # It feels like they're playing a joke on me.
    polycoeff = npoly.Polynomial.fit(yrange_tofit, core_positions_tofit, order).convert().coef
    lincoeff = npoly.Polynomial.fit(yrange_tofit, core_positions_tofit, 1).convert().coef

    core_polynomial = np.zeros(image.shape[0])
    for i in range(len(polycoeff)):
        core_polynomial += polycoeff[i] * yrange ** i
    core_linear = lincoeff[0] + yrange * lincoeff[1]
    shifts = (core_polynomial - core_linear[int(slit_reference * image.shape[0])])

    if np.abs(np.nanmean(shifts)) >= 7.5:
        warnings.warn(
            "Large average shift (" + str(np.nanmean(shifts)) + ") measured along the slit. Check your inputs.")

    return -shifts


@count_recursive_calls
def detect_beams_hairlines(
        image, threshold=0.5, hairline_width=5, line_width=15,
        expected_hairlines=2, expected_slits=1, expected_beams=1,
        fallback=False
):
    """
    Detects beam edges and intensity thresholds from an image (typically a flat).
    This function makes no assumptions as to the number of beams/slits used.
    Theoretically, it should work on a quad slit dual beam configuration from FIRS,
    which to my knowledge has not been used in some time.
    
    It works on derivative intensity thresholds. First across dimension 1 (left/right), detecting the number of slits
    from the averaged profile. Then across dimansion 2 (up/down) for each slit. This detects the top/bottom of the beam
    (important for FIRS and other polarizing beam split instruments), as well as smaller jumps from the harlines.
    
    :param image: numpy.ndarray
        2D image (typically an averaged flat field) for beam detection.
    :param threshold: float
        Threshold for derivative profile separation.
    :param hairline_width: int
        Maximum width of hairlines in pixels
    :param line_width: int
        Maximum width of spectal line in pixels
    :param expected_hairlines: int
        Expected number of hairlines. Used to recurse the function to fine-tune threshold
    :param expected_beams: int
        Expected number of beams in y. Used to recurse the function to fine-tune threshold
    :param expected_slits: int
        Expected number of slits in x. Used to recurse the function to fine-tune threshold
    :param fallback: bool
        If True, when the function fails, fall back to user selection of hairlines via interactive widget
    :return beam_edges: numpy.ndarray
        Numpy.ndarray of shape (2, nBeams), where each pair is (y0, y1) for each beam
    :return slit_edges: numpy.ndarray
        Numpy.ndarray of shape (2, nSlits), where each pair is (x0, x1) for each slit
    :return hairline_centers: numpy.ndarray
        Locations of hairlines. Shape is 1D, with length nhairlines, and the entries are the center of the hairline
    """
    mask = image.copy()
    thresh_val = threshold * np.nanmedian(mask)
    mask[mask <= thresh_val] = 1
    mask[mask >= thresh_val] = 0

    # Hairlines and y-limits first
    yprofile = np.nanmedian(mask, axis=1)
    yprofile_grad = np.gradient(yprofile)
    pos_peaks, _ = scig.find_peaks(yprofile_grad)
    neg_peaks, _ = scig.find_peaks(-yprofile_grad)
    # Every peak that has no corresponding opposite sign peak within hairline width
    # is a beam edge. Otherwise, it's a hairline.
    hairline_starts = []
    edges = []
    for peak in pos_peaks:
        if len(neg_peaks[(neg_peaks >= peak - hairline_width) & (neg_peaks <= peak + hairline_width)]) > 0:
            hairline_starts.append(peak)
        else:
            edges.append(peak)
    hairline_ends = []
    for peak in neg_peaks:
        if len(pos_peaks[(pos_peaks >= peak - hairline_width) & (pos_peaks <= peak + hairline_width)]) > 0:
            hairline_ends.append(peak)
        else:
            edges.append(peak)
    # This should return same sized lists for hairline_starts and hairline_ends. In case it isn't:
    hairline_starts = hairline_starts[:len(hairline_ends)]
    # Sort the beam edges
    # Should also pad with 0 and -1 if there's no gap between the beam and the detector edge
    # We'll do this by checking if the first edge is < 100, and adding 0 as the first edge if it isn't
    # Then, if the length of edges is even, we're good. If it's odd, we add the last index as well.
    edges = sorted(edges)
    # Fudge: No edges detected (beam fills)
    if len(edges) == 0:
        edges = [0, len(yprofile) - 1]
    if edges[0] > 100:
        edges = [0] + edges
    if len(edges) % 2 == 1:
        edges.append(len(yprofile) - 1)
    # Now we'll check the hairlines, and return the mean value of the start/end pair
    hairline_centers = []
    for i in range(len(hairline_starts)):
        hairline_centers.append((hairline_starts[i] + hairline_ends[i]) / 2)

    # We can use similar logic to find the beam edges in x.
    # Flatten in the other direction, and avoid spectral line residuals in the same way we picked out hairlines
    # We'll use a different, wider window for the spectral line avoidance, specifically for Ca II 8542 and H alpha
    # 10 should be okay.
    xprofile = np.nanmedian(mask, axis=0)
    xprofile_grad = np.gradient(xprofile)
    pos_peaks, _ = scig.find_peaks(xprofile_grad)
    neg_peaks, _ = scig.find_peaks(-xprofile_grad)
    xedges = []
    for peak in pos_peaks:
        if len(neg_peaks[(neg_peaks >= peak - line_width) & (neg_peaks <= peak + line_width)]) == 0:
            xedges.append(peak)
    for peak in neg_peaks:
        if len(pos_peaks[(pos_peaks >= peak - line_width) & (pos_peaks <= peak + line_width)]) == 0:
            xedges.append(peak)
    # Clean it up. If there's no edges found, the beam fills the chip, index to 0, -1
    if len(xedges) == 0:
        xedges = [0, len(xprofile) - 1]
    # If there's no edges found in the first 50, add a zero to the front
    xedges = sorted(xedges)
    if xedges[0] > 50:
        xedges = [0] + xedges
    # If there are now an even number of edges, most likely situation is that we missed the end of the last slit.
    if len(xedges) % 2 == 1:
        xedges.append(len(xprofile) - 1)

    beam_edges = np.array(edges).reshape(int(len(edges) / 2), 2)
    slit_edges = np.array(xedges).reshape(int(len(xedges) / 2), 2)
    hairline_centers = np.array(hairline_centers)
    # Recursion in event of improperly detected hairlines
    # If there's more than 50 recursions happening, that's enough to have
    # looped from threshold 0.85 to 5e-2 twice without finding a correct
    # solution. Raise an error.
    if detect_beams_hairlines.num_calls > 50:
        if fallback:
            print(
                "expected_hairlines={0}, expected_beams={1}, expected_slits={2}, ".format(
                    expected_hairlines, expected_beams, expected_slits
                )
            )
            print(
                "Expected values not found within 50 iterations. Falling back to user selection."
            )
            beam_edges, slit_edges, hairline_centers = select_beam_edges_hairlines(
                np.nanmedian(image, axis=1), np.nanmedian(image, axis=0),
                expected_hairlines, expected_beams, expected_slits
            )
            return beam_edges, slit_edges, hairline_centers
        raise Exception(
            "expected_hairlines={0}, expected_beams={1}, expected_slits={2}, ".format(
                expected_hairlines, expected_beams, expected_slits
            ) + "expected number of hairlines, beams, or slits not found within 50 iterations"
        )

    if (
            (
                (len(hairline_centers) != expected_hairlines) or
                (beam_edges.shape[0] != expected_beams) or
                (slit_edges.shape[0] != expected_slits)
            ) and
            (threshold > 5e-2)
    ):
        # Attempt to decrement the threshold until the correct number
        # of hairlines is detected.
        beam_edges, slit_edges, hairline_centers = detect_beams_hairlines(
            image, threshold=0.9*threshold,
            hairline_width=hairline_width,
            line_width=line_width,
            expected_hairlines=expected_hairlines,
            expected_beams=expected_beams,
            expected_slits=expected_slits,
            fallback=fallback
        )
    elif threshold <= 5e-2:
        # If the threshold falls without the correct number of hairlines
        # being detected, reset the threshold and continue
        beam_edges, slit_edges, hairline_centers = detect_beams_hairlines(
            image, threshold=0.85,
            hairline_width=hairline_width,
            line_width=line_width,
            expected_hairlines=expected_hairlines,
            expected_beams=expected_beams,
            expected_slits=expected_slits,
            fallback=fallback
        )


    return beam_edges, slit_edges, hairline_centers


def select_beam_edges_hairlines(
        average_y_profile: npt.NDArray, average_x_profile: npt.NDArray,
        expected_hairlines: int, expected_beams: int, expected_slits: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Fallback function for user selection of hairlines, beam edges, and slit edges
    via widgets found elsewhere in this file. Called by detect_beams_hairlines with
    the fallback=True keyword argument.

    Typically encountered in situations where a hairline is close to an image edge,
    such as with the FLIR cameras, which have a smaller chip size than the Sarnoff
    cameras

    Parameters
    ----------
    average_y_profile : numpy.ndarray
        Average profile along the y-axis of the original image
    average_x_profile : numpy.ndarray
        Average profile along the x-axis of the original image
    expected_hairlines : int
        Number of hairlines to select
    expected_beams : int
        Twice this number of edges to select in y
    expected_slits : int
        Twice this number of edges to select in x

    Returns
    -------
    beam_edges: numpy.ndarray
        Edges of the beam reshaped to (expectedBeams, 2)
    slit_edges: numpy.ndarray
        Edges of the slit image reshaped to (expectedSlits, 2)
    hairlines: numpy.ndarray
        Hairline centers in a 1D array
    """

    # Select Beam Edges
    print(
        "\nSelect {0} Intensity Jumps. These correspond to the edges of the spatial beam(s)\n".format(
            int(2 * expected_beams)
        )
    )
    print("Beam Edges May Extend across the range. If the beam fills the range, select the edges of the range")
    approx_beam_edges = select_lines_singlepanel(
        average_y_profile, int(2 * expected_beams),
        fig_name="Select {0} Positions corresponding to edges of the spatial beam".format(
            int(2 * expected_beams)
        )
    )
    # Clean the beam edges to remove out-of-bounds values
    approx_beam_edges[approx_beam_edges < 0] = 0
    approx_beam_edges[approx_beam_edges >= len(average_y_profile)] = len(average_y_profile) - 1
    beam_edges = np.sort(np.round(approx_beam_edges)).reshape(expected_beams, 2).astype(int)
    # Select Slit Edges
    print(
        "\nSelect {0} Intensity Jumps. These correspond to the edges of the spectral beam(s)\n".format(
            int(2 * expected_slits)
        )
    )
    print("Take care not to select spectral lines")
    approx_slit_edges = select_lines_singlepanel(
        average_x_profile, int(2 * expected_slits),
        fig_name="Select {0} Positions corresponding to the edges of the spectral beam".format(
            int(2 * expected_slits)
        )
    )
    # Clean slit edges to remove out-of-bounds values
    approx_slit_edges[approx_slit_edges < 0] = 0
    approx_slit_edges[approx_slit_edges >= len(average_x_profile)] = len(average_x_profile) - 1
    slit_edges = np.sort(np.round(approx_slit_edges)).reshape(expected_slits, 2).astype(int)
    # Select Hairlines
    print(
        "\nSelect {0} Intensity dips. These should correspond to the hairlines\n".format(
            int(expected_hairlines)
        )
    )
    approx_hairlines = select_lines_singlepanel(
        average_y_profile, int(expected_hairlines),
        fig_name="Select {0} Positions corresponding to hairline positions".format(
            int(expected_hairlines)
        )
    )
    approx_hairlines = np.sort(approx_hairlines)
    # Clean up hairlines by finding subpixel line center
    hairlines = np.array([find_line_core(average_y_profile[int(i - 3):int(i + 4)]) + i - 3 for i in approx_hairlines])
    return beam_edges, slit_edges, hairlines


def create_gaintables(flat, lines_indices,
                      hairline_positions=None, neighborhood=6,
                      hairline_width=3, edge_padding=10):
    """
    Creates the gain of a given input flat field beam.
    This assumes a deskewed field, and will determine the shift of the template mean profile, then detrend the spectral
    profile, leaving only the background detector flat field.
    :param flat: numpy.ndarray
        Dark-corrected flat field image to determine the gain for
    :param lines_indices: list
        Indices of the spectral line to use for deskew. Form [idx_low, idx_hi]
    :param hairline_positions: list
        List of hairline y-centers. If None, hairlines are not masked. May cause issues in line position finding.
    :param neighborhood: int
        Size of region to use in median filtering for comparison profile calculation.
    :param hairline_width: int
        Width of hairlines for masking. Default is 3
    :param edge_padding: int
        Cuts the profile arrays by this amount on each end to avoid edge effects.
    :return gaintable: numpy.ndarray
        Gain table from iterating along overlapping subdivisions
    :return coarse_gaintable: numpy.ndarray
        Gain table from using full slit-averaged profile
    :return init_skew_shifts: numpy.ndarray
        Shifts used in initial flat field deskew. Can be applied to final gain-corrected science maps.
    """
    masked_flat = flat.copy()
    if hairline_positions is not None:
        for line in hairline_positions:
            masked_flat[int(line - hairline_width - 1):int(line + hairline_width), :] = np.nan
    init_skew_shifts = spectral_skew(masked_flat[:, lines_indices[0]:lines_indices[1]])
    init_deskew = np.zeros(masked_flat.shape)
    for i in range(masked_flat.shape[0]):
        init_deskew[i, :] = scind.shift(masked_flat[i, :], init_skew_shifts[i], mode='nearest')
    mean_profile = np.nanmean(
        init_deskew[
            int(init_deskew.shape[0] / 2 - 30):int(init_deskew.shape[0] / 2 + 30), :
        ],
        axis=0
    )
    mean_profile_center = find_line_core(mean_profile[lines_indices[0] - 3:lines_indices[1] + 3]) + lines_indices[0] - 3
    shifted_lines = np.zeros(masked_flat.shape)
    sh = []
    for i in range(masked_flat.shape[0]):
        if i == 0:
            last_nonnan = np.nan
        line_position = find_line_core(
            masked_flat[i, lines_indices[0] - 3:lines_indices[1] + 3]
        ) + lines_indices[0] - 3
        if np.isnan(line_position):
            if np.isnan(last_nonnan):
                shift = 0
            else:
                shift = last_nonnan - mean_profile_center
        else:
            shift = line_position - mean_profile_center
            last_nonnan = line_position
        sh.append(shift)
        shifted_lines[i, :] = scind.shift(mean_profile, shift, mode='nearest')
    coarse_gaintable = flat / shifted_lines
    if hairline_positions is not None:
        for line in hairline_positions:
            coarse_gaintable[int(line - hairline_width - 1):int(line + hairline_width), :] = 1

    # Smooth rough gaintable in the chosen line
    if lines_indices[0] < 20:
        lowidx = 0
    else:
        lowidx = lines_indices[0] - 20
    if lines_indices[1] > flat.shape[0] - 20:
        highidx = flat.shape[0] - 1
    else:
        highidx = lines_indices[1] + 20
    for i in range(coarse_gaintable.shape[1]):
        coarse_gaintable[lines_indices[0] - 7:lines_indices[1] + 7, i] = np.nanmean(coarse_gaintable[lowidx:highidx, i])

    corrected_flat = masked_flat / coarse_gaintable

    skew_shifts = spectral_skew(corrected_flat[:, lines_indices[0]:lines_indices[1]])
    deskew_corrected_flat = np.zeros(corrected_flat.shape)
    for j in range(corrected_flat.shape[0]):
        deskew_corrected_flat[j, :] = scind.shift(corrected_flat[j, :], skew_shifts[j], mode='nearest')
    shifted_lines = np.zeros(corrected_flat.shape)
    if hairline_positions is not None:
        for line in hairline_positions:
            # Need a contingency for if a hairline is at the edge of the flat
            deskew_corrected_flat[
                int(line - hairline_width - 1):int(line + hairline_width), :
            ] = deskew_corrected_flat[(int(line + hairline_width + 2))]
            corrected_flat[
                int(line - hairline_width - 1):int(line + hairline_width), :
            ] = corrected_flat[(int(line + hairline_width + 2))]
    mean_profiles = scind.median_filter(deskew_corrected_flat, size=(neighborhood, 1))
    for j in tqdm.tqdm(range(corrected_flat.shape[0]), desc="Constructing Gain Tables"):
        ref_profile = corrected_flat[j, :] / np.nanmedian(corrected_flat[j, :])
        mean_profile = mean_profiles[j, :] / np.nanmedian(mean_profiles[j, :])
        mean_profile = scind.shift(mean_profile, -skew_shifts[j], mode='nearest')
        line_shift = iterate_shifts(
            ref_profile[edge_padding:-edge_padding],
            mean_profile[edge_padding:-edge_padding]
        )
        sh.append(line_shift)
        shifted_lines[j, :] = scind.shift(mean_profile, line_shift, mode='nearest')
    gaintable = flat / shifted_lines
    gaintable /= np.nanmedian(gaintable)
    if hairline_positions is not None:
        for line in hairline_positions:
            gaintable[int(line - hairline_width - 1):int(line + hairline_width), :] = 1

    return gaintable, coarse_gaintable, init_skew_shifts


def iterate_shifts(reference_profile, mean_profile, nzones=5):
    """
    Determines best shift for the mean profile to the reference profile from the median shift in each of N zones
    :param reference_profile: numpy.ndarray
        Profile to determine shifts to
    :param mean_profile: numpy.ndarray
        Profile to shift
    :param nzones: int
        Number of subfields to consider shifts for
    :return: float
        Median of each subfield shift
    """
    reference_slices = np.array_split(reference_profile, nzones)
    mean_slices = np.array_split(mean_profile, nzones)
    shifts = np.zeros(len(mean_slices))
    for i in range(len(reference_slices)):
        shifts[i] = scopt.minimize_scalar(
            fit_profile,
            bounds=(-5, 5),
            args=(reference_slices[i], mean_slices[i])
        ).x
    return np.nanmedian(shifts)


def fit_profile(shift, reference_profile, mean_profile, landing_width=5):
    """
    Alternate minimization of shift residuals using the final "gain" image
    :param shift: float
        Value for shift
    :param reference_profile: numpy.ndarray
        Reference profile to divide against
    :param mean_profile: numpy.ndarray
        Mean Profile for division
    :param landing_width: int
        determines slope/bg of residuals. Higher to negate edge effects
    :return fit_metric: float
        Sum of "gain" profile
    """
    shifted_mean = scind.shift(mean_profile, shift, mode='nearest')
    divided = reference_profile / shifted_mean
    slope = (np.nanmean(divided[-landing_width:]) - np.nanmean(divided[:landing_width])) / divided.size
    bg = slope * np.arange(divided.size) + np.nanmean(divided[:landing_width])
    gainsub = np.abs(divided - bg)
    fit_metric = np.nansum(gainsub[np.isfinite(gainsub)])
    return fit_metric


def prefilter_correction(spectral_image, wavelength_array,
                         reference_profile, reference_wavelength,
                         polynomial_order=2, edge_padding=10, smoothing=(20, 4)):
    """
    Performs prefilter/grating efficiency correction for the spectral image.
    The algorithm is similar to the gain table creation, but the reference profile here is a fiducial, such as the FTS
    atlas. The spectral image is median smoothed in both axes, the reference profile is matched and divided out,
    Then a polynomial fit to the residuals functions as a prefilter correction
    :param spectral_image: numpy.ndarray
        Single slit position of the shape (ny, nlambda)
    :param wavelength_array: numpy.ndarray
        Array of reference wavelengths, of the shape (nlambda)
    :param reference_profile: numpy.ndarray
        From the FTS atlas or other, of the shape (nlambda(ref)). Will be interpolated to wavelength_array grid
    :param reference_wavelength: numpy.ndarray
        From the FTS atlas or other, of the shape (nlamda(ref)). Used in interpolation
    :param polynomial_order: int
        Order of polynomial for residual fit. Usually 2 is sufficient. The major wavelength-direction variation is
        due to a combination of grating efficiency drop-off and prefilter efficiency drop-off, both of which are
        vaguely quadratic.
    :param edge_padding: int
        Amount to cut from the wavelength edges before fit. Final result is applied over the full wavelength range
    :param smoothing: tuple or int
        Passed through to scipy.ndimage.median_filter as the filter footprint
    :return prefilter_profiles: numpy.ndarray
        Shape (ny, nlambda) of normalized polynomials along the slit. Dividing by this should detrend prefilter curve.
    """

    ref_prof = scinterp.CubicSpline(
        reference_wavelength,
        reference_profile
    )(wavelength_array)
    if type(smoothing) is tuple:
        smoothed_ref = scind.median_filter(ref_prof, size=smoothing[1])
    else:
        smoothed_ref = scind.median_filter(ref_prof, size=smoothing)
    smoothed_profiles = scind.median_filter(spectral_image, size=smoothing)
    prefilter_profiles = np.zeros(spectral_image.shape)
    for j in tqdm.tqdm(range(smoothed_profiles.shape[0]), desc="Determining prefilter curves"):
        data_profile = smoothed_profiles[j, :] / np.nanmedian(smoothed_profiles[j, :])
        line_shift = iterate_shifts(
            data_profile[edge_padding:-edge_padding],
            smoothed_ref[edge_padding:-edge_padding]
        )
        shifted_ref = scind.shift(smoothed_ref, line_shift, mode='nearest')
        profile_to_fit = data_profile[edge_padding:-edge_padding] / shifted_ref[edge_padding:-edge_padding]
        coef = np.polynomial.polynomial.Polynomial.fit(
            np.arange(len(profile_to_fit)), profile_to_fit, polynomial_order
        ).convert().coef
        poly_prof = np.zeros(smoothed_profiles.shape[1])
        for i in range(len(coef)):
            poly_prof += coef[i] * np.arange(len(poly_prof)) ** i
        prefilter_profiles[j, :] = poly_prof / np.nanmedian(poly_prof)
    return prefilter_profiles


def fourier_fringe_correction(fringe_cube, freqency_cutoff, smoothing, dlambda):
    """
    Performs simple masking in Fourier space for fringe correction.
    Returns a normalized fringe template for division
    :param fringe_cube: numpy.ndarray
        Datacube with fringes
    :param freqency_cutoff:
        Cutoff frequency. Everything outside this is considered fringe
    :param smoothing: tuple of int or int
        Pass-through to scipy.ndimage.median_filter. Smooths datacube in spatial/spectral dimension
    :param dlambda: float
        Wavelength resolution for use in determining cutoff freqs.
    :return fringe_template: numpy.ndarray
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


def select_fringe_freq(wvl, profile, init_period):
    """
    Allows user to adjust fringe frequencies and select best cutoff for template.
    :param wvl: numpy.ndarray
        Wavelength array
    :param profile: numpy.ndarray
        Spectral profile
    :param init_period: float
        Initial periodicities
    :return period_slider.value: float
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


def moment_analysis(wave, intens, refwvl, continuum_correct=True):
    """
    Performs simple moment analysis of an input spectral profile.
    :param wave: numpy.ndarray
        Wavelength grid
    :param intens: numpy.ndarray
        Intensity values
    :param refwvl: float
        Reference wavelength value
    :param continuum_correct: bool
        If True, does a quick correction for pseudo continuum via start/end values
    :return i: float
        Intensity value
    :return v: float
        Doppler velocity (km/s)
    :return w: float
        Doppler width (km/s)
    """
    if continuum_correct:
        slope = (np.nanmean(intens[-3:]) - np.nanmean(intens[:3])) / (wave[-1] - wave[0])
        line = wave * slope
        line /= np.nanmedian(line)
        intens /= line
    i = scint.simpson(intens, x=wave)
    m1 = scint.simpson(intens * (wave - refwvl), x=wave)
    m2 = scint.simpson(intens * (wave - refwvl) ** 2, x=wave)
    v = (c_kms / refwvl) * (m1 / i)
    w = np.sqrt((c_kms / refwvl) * (m2 / i))
    return i, v, w


def mean_circular_polarization(stokes_v, wavelength, reference_wavelength, continuum_i):
    """
    Computes mean circular polarization as described by Martinez Pillet (2011).
    Adapted for many spectral positions.
    Parameters
    ----------
    stokes_v : numpy.ndarray
    wavelength : numpy.ndarray
    reference_wavelength : float
    continuum_i : float

    Returns
    -------
    mcp : float
    """
    sign_vector = np.array([1 if x > reference_wavelength else -1 for x in wavelength])

    mcp = (1 / (len(wavelength) * continuum_i)) * np.nansum(
        sign_vector * np.abs(stokes_v)
    )
    return mcp


def mean_linear_polarization(stokes_q, stokes_u, continuum_i):
    """
    Computes mean linear polarization as described by Martinez Pillet (2011).
    Parameters
    ----------
    stokes_q : numpy.ndarray
        Stokes-Q profile
    stokes_u : numpy.ndarray
        Stokes-U profile
    continuum_i : float
        Mean Stokes-I continuum intensity

    Returns
    -------
    float
        Mean linear polarization
    """
    return (1 / (len(stokes_q) * continuum_i)) * np.nansum(np.sqrt(stokes_q ** 2 + stokes_u ** 2))


def net_circular_polarization(stokes_v, wavelength, abs_value=False):
    """
    Just integrates the V-profile. Solanki & Montavon (1993)
    Parameters
    ----------
    stokes_v: numpy.ndarray
        Stokes-V
    wavelength: numpy.ndarray
        Wavelengths
    abs_value: bool
        True to integrate the absolute value of V
    Returns
    -------
    float
        Net circular polarizaed light.
    """
    if abs_value:
        return scint.simpson(np.abs(stokes_v), x=wavelength)
    else:
        return scint.simpson(stokes_v, x=wavelength)


def chi_square(fit, prior):
    return np.nansum((fit - prior) ** 2) / len(fit)


def image_align(image, reference):
    """
    Wraps scipy.signal.fftconvolve to align two images.

    Parameters
    ----------
    image : numpy.ndarray
        2D image to align
    reference : numpy.ndarray
        2d reference image for alignment

    Returns
    -------
    aligned_image : numpy.ndarray
        Aligned image
    shifts : numpy.ndarray
        Shifts to align image
    """

    y0 = image.shape[0] / 2.
    x0 = image.shape[1] / 2.
    y0_ref = reference.shape[0] / 2.
    x0_ref = reference.shape[1] / 2.

    shifts = np.zeros(2)
    aligned_image = np.zeros(image.shape)

    img = image - np.nanmean(image)
    ref = reference - np.nanmean(reference)

    corrmap = scig.fftconvolve(img, ref[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(corrmap), corrmap.shape)
    shifts[0] = (y0 - y - (y0 - y0_ref))
    shifts[1] = (x0 - x - (x0 - x0_ref))

    scind.shift(image, shifts, output=aligned_image, mode='constant', cval=np.nanmean(image))

    return aligned_image, shifts


def image_align_apod(
        image: np.ndarray, reference: np.ndarray,
        subtile: list or None=None
) -> tuple[np.ndarray, list]:
    """
    Align image to reference using a subtile for alignment.
    By default chooses central 256 pixels.

    Parameters
    ----------
    image : numpy.ndarray
    reference : numpy.ndarray
    subtile : list

    Returns
    -------
    aligned : numpy.ndarray
    shifts : list

    """
    
    def window_apod(tile_size: int, fraction: float) -> np.ndarray:
        """
        Creates an apodization window for normalizing prior to cross-correlation

        Parameters
        ----------
        tile_size : int
            Size of the subfield tile
        fraction : float
            Fraction of tile the function should be wide

        Returns
        -------
        window : numpy.ndarray
            2D window function
        """
        apodization_size = int(tile_size * fraction + 0.5)
        x = np.arange(apodization_size) / apodization_size * np.pi / 2.
        y = np.sin(x) ** 2
        z = np.ones(tile_size)
        z[:apodization_size] = y
        z[-apodization_size:] = np.flip(y)

        window = np.outer(z, z)
        return window

    if subtile is None:
        subtile = [image.shape[0]//2 - 128, image.shape[1]//2-128, 256]
    window = window_apod(subtile[2], 0.4375)
    window /= np.mean(window)

    # Clip to subtile:
    ref = reference[subtile[0]:subtile[0] + subtile[2], subtile[1]:subtile[1] + subtile[2]]
    img = image[subtile[0]:subtile[0] + subtile[2], subtile[1]:subtile[1] + subtile[2]]
    # Normalization
    mean_ref = np.mean(ref)
    std_ref = np.std(ref)
    mean_img = np.mean(img)
    std_img = np.std(img)

    # Correlation
    ref_ft = np.fft.rfft2((ref - mean_ref)/std_ref * window)
    img_ft = np.fft.rfft2((img - mean_img)/std_img * window)
    # Shift zero-frequencies to center of spectrum
    xcorr = np.fft.fftshift(
        np.fft.irfft2(
            np.conj(img_ft) * ref_ft
        )
    ) / (ref.shape[0] * ref.shape[1])
    # Integer shift
    max_idx = np.argmax(xcorr)
    yshift = max_idx // xcorr.shape[0] - xcorr.shape[0] // 2
    xshift = max_idx % xcorr.shape[0] - xcorr.shape[1] // 2
    tolerance = subtile[2]/2
    if np.abs(yshift) > tolerance:
        yshift = 0
    if np.abs(xshift) > tolerance:
        xshift = 0
    aligned = np.roll(image, (yshift, xshift))
    shifts = [yshift, xshift]
    return aligned, shifts


def grating_calculations(
        gpmm, blaze, alpha, pix_size, wavelength, order,
        collimator=3040, camera=1700, slit_width=40, slit_scale=3.76*1559/780,
        grating_length=206, pupil_diameter=762*1559/54864, slit_camera=780
):
    """
    Basic grating calculation set. Calculates dispersion,
    Parameters
    ----------
    gpmm : float
        lines per mm of grating
    blaze : float
        blaze angle of grating [degrees]
    alpha : float
        incident angle of light/tilt angle of grating [degrees]
    pix_size : float
        size of detector pixels [um]
    wavelength : float
        Wavelength of interest in Angstrom
    order : int
        Spectral order of interest
    collimator : float
        focal length of spectrograph collimator [mm]. For SPINOR/HSG, this is usually 3040 mm
    camera : float
        focal length of spectrograph camera lens [mm]. For SPINOR/HSG, this is usually 1700 mm
    slit_width : float
        Width of spectrograph slit [um].
    slit_scale : float
        Plate scale of image on slit unit.
        Typically ~7.52 asec/mm for the 780 mm lens that we usually use to feed HSG/SPINOR/FIRS
    grating_length : float
        Length in mm of grating. Typically 206 for SPINOR/HSG/FIRS gratings.
    pupil_diameter : float
        Diameter of pupil [mm] before the lens that images the field onto the slit unit.
        Default value is the DST Port 4 pupil, evaluates to ~21.65 mm
    slit_camera : float
        Focal length of camera lens on the slit unit. Usually 780 mm

    Returns
    -------
    grating_params : numpy.rec.recarray
        Numpy recarray containing the following attributes:
        eff : float
            Grating efficiency
        shade : float
            Grating shading percent
        net_eff : float
            Combined efficiency from eff and shade
        littrow : float
            Littrow angle of the reflected beam
        spectral_scale : float
            Spectral pixel width
        spectral_resolution : float
            From slit width, diffraction resolution, etc.
    """
    # Conversion factors
    angstrom_per_m = 1e10
    um_per_m = 1e6
    mm_per_m = 1e3

    # f-number = focal length / diameter
    fnum = slit_camera / pupil_diameter

    # theta = arcsin(m * lambda * d - sin(alpha)), d = grating scale, m = order, alpha = incident angle
    refl_angle = np.arcsin(
        order * (wavelength/angstrom_per_m)  * (gpmm * mm_per_m ) - np.sin(alpha * np.pi/180)
    ) * 180/np.pi

    # magnification, M = cos(alpha)/cos(theta)
    slit_mag = np.cos(alpha * np.pi/180.) / np.cos(refl_angle * np.pi/180.)

    # Resolving power, R = W*m
    # Here, W is really a stand-in for the number of illuminated rulings on the grating.
    # m is, as always, the order
    # We compare the projected width of the grating at the angle alpha
    # size of the illuminated patch
    n_lines = grating_length * gpmm
    projected_width = (grating_length/mm_per_m) * np.cos(alpha * np.pi/180.)
    geometric_width = (collimator/mm_per_m) / fnum
    slit_diffraction_width = 2 * (collimator/mm_per_m) * (wavelength/angstrom_per_m) / (slit_width/um_per_m)
    illumination_width = np.sqrt(geometric_width**2 + slit_diffraction_width**2)
    # i.e., if the illumination width is smaller than projected width, some percent of the total
    # number of rulings will be illuminated, expressed as a percent of the projection width.
    # If the projected width is the limiting factor, 100% of the rules will be illuminated
    effective_grating_width = np.nanmin([projected_width, illumination_width])/projected_width
    grating_resolution = order * effective_grating_width * n_lines
    diffraction_resolution_ma = 1000 * wavelength / grating_resolution

    # Dispersion, D = camera * m * d / cos(theta)
    dispersion = (camera/mm_per_m) * order * (gpmm*mm_per_m) / np.cos(refl_angle * np.pi/180)
    # unitless quantity; m/m. Convert to um/mA
    dispersion_um_ma = dispersion * um_per_m / angstrom_per_m / 1000
    spectral_scale = pix_size / dispersion_um_ma
    # Spectral slit width = camera * slit_width * slit_magnification / (dispersion * collimator)
    spectral_slit_width = camera * slit_width * slit_mag / (dispersion_um_ma * collimator)

    # Add diffraction resln., slit width, and pixel size in quadrature
    spectral_resolution = np.sqrt(diffraction_resolution_ma**2 + spectral_scale**2 + spectral_slit_width**2)

    # grating efficiency calculation:
    # gamma = pi * 1/d * cos(theta) * [sin(alpha - blaze) + sin(theta - blaze)]/(lambda*cos(alpha - blaze))
    # eff. (sin(gamma)/gamma)^2
    gamma = np.pi * np.cos(
        refl_angle * np.pi/180
    ) * (np.sin((alpha - blaze)*np.pi/180) + np.sin((refl_angle - blaze)*np.pi/180))/(
        gpmm*mm_per_m*(wavelength/angstrom_per_m)*np.cos((alpha - blaze)*np.pi/180)
    )
    grating_efficiency = (np.sin(gamma)/gamma)**2
    incident_shade = 1 - np.sin(np.abs((alpha-blaze)*np.pi/180))/np.sin((90-blaze)*np.pi/180)
    reflect_shade = 1 - np.sin(np.abs((refl_angle-blaze)*np.pi/180))/np.sin((90-blaze)*np.pi/180)
    shade = np.nanmin([incident_shade, reflect_shade])

    eff = grating_efficiency * shade

    littrow = alpha - refl_angle

    grating_params = np.rec.fromarrays(
        [
            np.array(grating_efficiency),
            np.array(shade),
            np.array(eff),
            np.array(littrow),
            np.array(spectral_scale),
            np.array(spectral_resolution)
        ],
        names=[
            "Grating_Efficiency",
            "Shading",
            "Total_Efficiency",
            "Littrow_Angle",
            "Spectral_Pixel",
            "Spectrograph_Resolution"
        ]
    )

    return grating_params


def solve_grating(
        gpmm, blaze, pix_size, wavelength,
        alpha=None,
        collimator=3040, camera=1700, slit_width=40, slit_scale=3.76*1559/780,
        grating_length=206, pupil_diameter=762*1559/54864, slit_camera=780,
        min_laser_angle=1.7
):
    """
    Solve for best grating angle and order for a given wavelength.
    Wraps grating_calculations function to solve first for most efficient order
    at alpha=blaze, then solves for alpha that maximizes efficiency in that order.
    If alpha is given, rather than being None, just solves for order, assuming that
    someone with a goal has decided on the best grating angle.

    Most params are passthroughts to grating_calculations
    Parameters
    ----------
    gpmm : float
        Lines per mm of used grating
    blaze : float
        Blaze angle (in degrees)
    pix_size : float
        Size of camera pixels in um
    wavelength : float
        Wavelength of interest (in Angstrom)
    alpha : None-type or float
        If None, solves for the most efficient grating angle.
        If grating angle is given (in degrees), solves for best order.
    collimator : float
        Focal length of collimator upstream of the grating.
        For HSG/SPINOR, usually 3040 mm
    camera : float
        Focal length of camera lens downstream of grating
        For HSG/SPINOR, ususally 1700 mm.
    slit_width : float
        Width of slit in um. Contributes to spectral resolution
    slit_scale : float
        Plate scale at the slit unit in arcsec/mm
    grating_length : float
        Length of ruled area in mm. Sets the number of illuminated lines
    pupil_diameter : float
        Diameter of pupil image upstream of slit unit. For DST, typically 21.65 mm
    slit_camera : float
        Focal length of camera lens that places the field on the slit unit.
        For HSG/SPINOR, usually 780 mm
    min_laser_angle : float or None
        Minimum allowable reflection angle, referenced to incoming beam.
        HSG/SPINOR in particular cannot accomodate angles less than 1.7 degrees,
        as these angles clip the rastering box structure. Can be None if this is
        of no concern

    Returns
    -------
    grating_solution : numpy.recarray
        Numpy structured array contating grating solution, to include:
            Order : int
            Alpha : float, degrees
            Grating_Efficiency : float, fractional
            Shading : float, fractional
            Total_Efficiency : float, fractional
            Littrow_Angle : float, degrees
            Spectral_Pixel : float, milliangstroms
            Spectrograph_Resolution : float, milliangstroms
    """
    if alpha is not None:
        init_alpha = alpha
    else:
        init_alpha = blaze
    order_arr = np.arange(1, 150, 1).astype(int)
    effs = np.array([
        grating_calculations(
            gpmm, blaze, init_alpha, pix_size, wavelength, x,
            collimator=collimator, camera=camera, slit_width=slit_width,
            slit_scale=slit_scale, grating_length=grating_length,
            pupil_diameter=pupil_diameter, slit_camera=slit_camera
        )['Total_Efficiency'] for x in order_arr
    ])

    best_order = order_arr[list(effs).index(np.nanmax(effs))]

    if alpha is None:
        max_ang = blaze + 20
        if max_ang >= 90:
            max_ang = 89
        alpha_arr = np.arange(blaze - 20, max_ang, 0.05)
        effs = np.zeros(alpha_arr.shape)
        littrow = np.zeros(alpha_arr.shape)
        for a in range(len(alpha_arr)):
            params = grating_calculations(
                gpmm, blaze, alpha_arr[a], pix_size, wavelength, best_order,
                collimator=collimator, camera=camera, slit_width=slit_width,
                slit_scale=slit_scale, grating_length=grating_length,
                pupil_diameter=pupil_diameter, slit_camera=slit_camera
            )
            effs[a] = params["Total_Efficiency"]
            littrow[a] = params["Littrow_Angle"]

        best_alpha = alpha_arr[list(effs).index(np.nanmax(effs))]
    else:
        best_alpha = alpha

    grating_params = grating_calculations(
        gpmm, blaze, best_alpha, pix_size, wavelength, best_order,
        collimator=collimator, camera=camera, slit_width=slit_width,
        slit_scale=slit_scale, grating_length=grating_length,
        pupil_diameter=pupil_diameter, slit_camera=slit_camera
    )
    # Incorrect. Returns an iterable. Figure out how to return modded recarray.
    grating_params = nrec.append_fields(
        [grating_params],
        ['Order', 'Alpha', 'Laser_Angle'],
        [
            np.array([best_order]),
            np.array([round(best_alpha, 2)]),
            np.array([round(grating_params['Littrow_Angle']/2, 3)])
        ],
        asrecarray=True
    )
    return grating_params

