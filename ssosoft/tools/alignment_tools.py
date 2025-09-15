import warnings

import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scinterp
import scipy.ndimage as scind
import sunpy.coordinates.frames as frames
import sunpy.map as smap
import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import RotatedSunFrame
from sunpy.net import Fido
from sunpy.net import attrs as a

"""
This is a toolbox for image alignment. This is not intended to be a complete
drop-in replacement for an iterative subfield-based approach, but rather to
correct misalignmnets between multiple channels of DST cameras and, more
importantly, the provide a decent first-order correction for the variable
offsets between the DST's solar coordinates and the "true" solar coordinates
of a given feature.

In the case of quiet-Sun observations, a more exhaustive strategy MUST be
pursued on a case-by-case basis.
"""


class PointingError(Exception):
    """
    Exception raised when pointing information is not minimally complete
    """
    def __init__(self, message="Pointing information is incomplete"):
        self.message = message
        super().__init__(self.message)

class ToleranceError(Exception):
    """Exception raised for solutions outside the allowed tolerances.

    Attributes:
    -----------
    tolerance -- tolerance value exceeded
    message -- explanation of error
    """

    def __init__(self, tolerance, message="Solution lies outside tolerance range: "):
        self.tolerance = tolerance
        self.message = message
        super().__init__(self.message, self.tolerance)

def _find_nearest(array: np.ndarray, value: float) -> int:
    """Determines index of closest array value to specified value
    """
    return int(np.abs(array-value).argmin())

def _window_apod(tile_size: int, fraction: float) -> np.ndarray:
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
    apodization_size = int(tile_size*fraction + 0.5)
    x = np.arange(apodization_size) / apodization_size * np.pi/2.
    y = np.sin(x)**2
    z = np.ones(tile_size)
    z[:apodization_size] = y
    z[-apodization_size:] = np.flip(y)

    window = np.outer(z, z)
    return window

def _image_align(
        image: np.ndarray, reference: np.ndarray,
        tolerance: float | None=None, subtile: list | None=None
) -> tuple[np.ndarray, list]:
    """
    Align image to reference using a subtile for alignment.
    By default, chooses the central 256 pixels, with a tolerance of 128 pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Image to align
    reference : numpy.ndarray
        Reference image for alignment
    tolerance : float
        Max number of acceptable pixels for shift
    subtile : list
        If given, should be of format [y0, x0, size], defines the subfield to use for alignment

    Raises
    ------
    ToleranceError
        If xshift or yshift falls outside the acceptable range

    Returns
    -------
    aligned : numpy.ndarray
        Pixel-aligned image
    shifts : list
        Of format [yshift, xshift]
    """
    if subtile is None and tolerance is None:
        tolerance = 128
        subtile = [image.shape[0]//2-128, image.shape[1]//2-128, 256]
    elif subtile is not None and tolerance is None:
        tolerance = subtile[2]
    elif tolerance is not None and subtile is None:
        subtile = [image.shape[0]//2-128, image.shape[1]//2-128, 256]
        tolerance = subtile[2]//2 if tolerance > subtile[2] - 1 else tolerance
    window = _window_apod(subtile[2], 0.4375)
    window /= np.mean(window)

    # Clip to the subtile to be used for alignment
    ref = reference[subtile[0]:subtile[0]+subtile[2], subtile[1]:subtile[1]+subtile[2]]
    img = image[subtile[0]:subtile[0] + subtile[2], subtile[1]:subtile[1] + subtile[2]]
    # Normalization statistics
    mean_ref = np.mean(ref)
    std_ref = np.std(ref)
    mean_img = np.mean(img)
    std_img = np.std(img)

    # Correlation
    ref_ft = np.fft.rfft2((ref-mean_ref)/std_ref * window)
    img_ft = np.fft.rfft2((img - mean_img)/std_img * window)
    # Shift zero-frequencies to center of spectrum
    xcorr = np.fft.fftshift(
        np.fft.irfft2(
            np.conj(img_ft) * ref_ft
        )
    ) / (ref.shape[0] * ref.shape[1])
    # Integer shift
    max_idx = np.argmax(xcorr)
    yshift = max_idx // xcorr.shape[0] - xcorr.shape[0]//2
    xshift = max_idx % xcorr.shape[0] - xcorr.shape[1]//2

    if (np.abs(yshift) > tolerance) or (np.abs(xshift) > tolerance):
        raise ToleranceError(tolerance)
    aligned = scind.shift(image, (yshift, xshift), mode="grid-wrap")
    shifts = [yshift, xshift]
    return aligned, shifts

def read_rosa_zyla_image(
        filename: str,
        slat: float=None, slon: float=None, dx: float=None, dy: float=None,
        rotation: float=None, obstime: str=None,
        translation: list=[]
    ) -> tuple[np.ndarray, dict]:
    """
    Reads a ROSA or HARDCam Level-1 or 1.5 FITS file and preps it for alignment. If the FITS file is from an
    older version of SSOSoft without useful metadata, these must be provided.

    Likewise, these cameras can be afflicted by an unknown series of flips and rotations induced by the optical path.
    If these are not corrected for (as in older versions of SSOSoft), these translations must be provided.
    See the SSOSoft github repository (https://github.com/sgsellers/SSOsoft) for an example of the correct target
    image orientation.

    :param filename: str
        Path to file containing image to be used for alignment.
    :param slat: float, or None-type
        Stonyhurst Latitude. Required if not in the FITS header
    :param slon: float, or None-type
        Stonyhurst Longitude. Required if not in the FITS header
    :param dx: float
        Pixel Scale in X. Required if not in the FITS header
    :param dy: float
        Pixel Scale in Y. Required if not in the FITS header
    :param rotation: float
        Rotation relative to solar-north. Note that this is NOT the same as the obs series GDRAN for the DST.
        The DST guider is rotated such that 13.3 degrees is solar north, so subtract 13.3 from the GDRAN.
        Note also that the DST GDRAN is set up such that positive is clockwise, so for scipy and other functions,
        the sign of the GDRAN must be reversed. This is done in the code. You just subtract off that 13.3, and we'll
        call it even, okay?
    :param obstime: str
        String containing obstime in format YYYY-MM-DDTHH:MM:SS.mmm. Required if not in FITS header.
    :param translation: list
        List of bulk translations. Currently accepts rot90, fliplr, flipud, flip, corresponding to numpy
        array manipulation functions of the same name. If you are unsure whether the data have already undergone
        bulk translation to match the telescope's orientation, check the PRSTEP keywords in the FITS header.
        If there are no PRSTEP keywords, or if the PRSTEP keywords do not indicate alignment to Solar-North, this
        must be provided.
    :return image: numpy.ndarray
        Image array
    :return header_dict: dictionary
        Python Dictionary containing minimal header keywords.
    """
    with fits.open(filename) as f:
        required_keywords = [
            "CDELT1", "CDELT2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CROTA2", "STARTOBS"
        ]
        # Check to see if all required keywords are in FITS header. If they aren't all there, check if
        # kwargs are provided
        if not all([i in list(f[0].header.keys()) for i in required_keywords]):
            if not all([slat, slon, dx, dy, rotation]):
                raise PointingError()
            # If they are, populate the metadata dictionary from kwargs
            else:
                coord = SkyCoord(
                    slon*u.deg, slat*u.deg,
                    obstime=obstime, observer="earth",
                    frame=frames.HeliographicStonyhurst
                ).transform_to(frames.Helioprojective)
                header_dict = {
                    "CDELT1": dx,
                    "CDELT2": dy,
                    "CTYPE1": "HPLN-TAN",
                    "CTYPE2": "HPLT-TAN",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CRVAL1": coord.Tx.value,
                    "CRVAL2": coord.Ty.value,
                    "CRPIX1": f[0].data.shape[1],
                    "CRPIX2": f[0].data.shape[0],
                    "CROTA2": rotation,
                    "DATE-AVG": obstime
                }
        else:
            header_dict = {
                "CDELT1": f[0].header["CDELT1"],
                "CDELT2": f[0].header["CDELT2"],
                "CTYPE1": f[0].header["CTYPE1"],
                "CTYPE2": f[0].header["CTYPE2"],
                "CUNIT1": f[0].header["CUNIT1"],
                "CUNIT2": f[0].header["CUNIT2"],
                "CRVAL1": f[0].header["CRVAL1"],
                "CRVAL2": f[0].header["CRVAL2"],
                "CRPIX1": f[0].header["CRPIX1"],
                "CRPIX2": f[0].header["CRPIX2"],
                "CROTA2": f[0].header["CROTA2"],
                "DATE-AVG": f[0].header["STARTOBS"]
            }
        image = f[0].data
        for i in translation:
            if "rot90" in i.lower():
                image = np.rot90(image)
            if "fliplr" in i.lower():
                image = np.fliplr(image)
            if "flipud" in i.lower():
                image = np.flipud(image)
    return image, header_dict

def read_spinor_image(
        filename: str,
        slat: float=None, slon: float=None, dx: float=None, dy: float=None,
        rotation: float=None, obstime: str=None,
        translation: list=[]
    ) -> tuple[np.ndarray, dict]:
    """Reads SPINOR image from Level-1 FITS file

    Parameters
    ----------
    filename : str
        Path to file
    slat : float, optional
        In not in header, Stonyhurst Lat, by default None
    slon : float, optional
        If not in header, Stonyhurst Lon, by default None
    dx : float, optional
        If not in header, x-scale in arcsec, by default None
    dy : float, optional
        If not in header, y-scale in arcsec, by default None
    rotation : float, optional
        If not in header, rotation in degrees, by default None
    obstime : str, optional
        If not in header, observation time string, by default None
    translation : list, optional
        If using an older product, additional translations required, by default []

    Returns
    -------
    tuple[np.ndarray, dict]
        Image and ponting info dictionary
    """
    with fits.open(filename) as f:
        required_keywords = [
            "CDELT1", "CDELT2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CROTA2"
        ]
        # Check to see if all required keywords are in FITS header. If they aren't all there, check if
        # kwargs are provided
        if not all([i in list(f[1].header.keys()) for i in required_keywords]):
            if not all([slat, slon, dx, dy, rotation]):
                raise PointingError()
            # If they are, populate the metadata dictionary from kwargs
            else:
                coord = SkyCoord(
                    slon*u.deg, slat*u.deg,
                    obstime=obstime, observer="earth",
                    frame=frames.HeliographicStonyhurst
                ).transform_to(frames.Helioprojective)
                header_dict = {
                    "CDELT1": dx,
                    "CDELT2": dy,
                    "CTYPE1": "HPLN-TAN",
                    "CTYPE2": "HPLT-TAN",
                    "CUNIT1": "arcsec",
                    "CUNIT2": "arcsec",
                    "CRVAL1": coord.Tx.value,
                    "CRVAL2": coord.Ty.value,
                    "CRPIX1": f[1].data.shape[1],
                    "CRPIX2": f[1].data.shape[0],
                    "CROTA2": rotation,
                    "DATE-AVG": obstime
                }
        else:
            dt = (np.datetime64(f[0].header["ENDOBS"]) - np.datetime64(f[0].header["STARTOBS"])) / 2
            header_dict = {
                "CDELT1": f[1].header["CDELT1"],
                "CDELT2": f[1].header["CDELT2"],
                "CTYPE1": f[1].header["CTYPE1"],
                "CTYPE2": f[1].header["CTYPE2"],
                "CUNIT1": f[1].header["CUNIT1"],
                "CUNIT2": f[1].header["CUNIT2"],
                "CRVAL1": f[1].header["CRVAL1"],
                "CRVAL2": f[1].header["CRVAL2"],
                "CRPIX1": f[1].header["CRPIX1"],
                "CRPIX2": f[1].header["CRPIX2"],
                "CROTA2": f[1].header["CROTA2"],
                "DATE-AVG": (np.datetime64(f[0].header["STARTOBS"]) + dt).astype(str)
            }
        image = np.mean(f[1].data, axis=-1)
        for i in translation:
            if "rot90" in i.lower():
                image = np.rot90(image)
            if "fliplr" in i.lower():
                image = np.fliplr(image)
            if "flipud" in i.lower():
                image = np.flipud(image)

    return image, header_dict

def align_images(
    data_image: np.ndarray, data_dict: dict,
    reference_smap: smap.GenericMap,
    niter: int=3, rotation_correct: bool=False,
    subtile: None | list=None, tolerance: None | float=None
) -> dict:
    """Performs iterative bulk and fine alignment of reference and data images.
    General flow is:
        1.) Interpolate reference image to data image scale
        2.) Align reference image to data image
        3.) Update centering of reference image
        4.) Repeat niter times

    Parameters
    ----------
    data_image : np.ndarray
        2D array containing image to align. Expected to be rotated to solar north and cropped
    data_dict : dict
        Python dictionary of WCS coordinates. Should contain:
            a.) DATE-AVG
            b.) CDELT1/2
            c.) CRPIX1/2 NOTE: MUST Be altered for new yrange if hairlines are clipped
            d.) CTYPE1/2
            e.) CUNIT1/2
            f.) CRVAL1/2
            g.) CROTA2 Rotation from Solar-north
    reference_smap : smap.GenericMap
        Full-disk Sunpy map containing reference image. Generally HMI
    niter : int, optional
        Number of sequential alignments to perform, by default 3
    rotation_correct : bool, optional
        If True, after alignment, attempts to determine relative rotation via linear correlation, by default False
    subtile : None or list, optional
        If given, passes through to _image_align with format [y0, x0, size] defining the alignment tile, by default None
    tolerance : None or float, optional
        If given, passes through to _image_align defining the pixel tolerance of shifts, by default None

    Returns
    -------
    data_dict : dict
        Python dictionary of correct alignment keywords, copied and modified from data_dict
    """
    # Work with a copy of the original
    data_dict = data_dict.copy()
    # The image alignment routine default values for the subtile and tolerance are very strict.
    # The original intention was for fine-tuning alignment, but the DST pointing values are frequently very far off
    # We'll override these values with a much wider range
    if subtile is None:
        subtile = [50, 50, min(data_image.shape) - 100] # All but the outer 50 pixels
        if tolerance is None:
            tolerance = 0.75 * (min(data_image.shape) - 100) # shift by up to 3/4 of the frame
    elif tolerance is None:
        tolerance = subtile[2] * 0.75

    # Sequential Alignment. Sometimes DST coordinates are far off, so repeat alignment with increasingly-accurate
    # Reference image submaps.

    for i in range(niter):
        # Near or beyond the limb, HMI and AIA pixel coordinates break down.
        # To be safe, we're going to do some fuckery with coordinates instead of a simple submap.
        # In the future, we'll probably want to pad the reference FOV, just in case the coordinates are very far off.
        # To do that, we'll need a dummy map of a larger size
        # Pad by FOV/2 (i.e., and extra FOV/4 on each side)
        dummy_array = np.ones(
            (data_image.shape[0] + int(data_image.shape[0]/2),
             data_image.shape[1] + int(data_image.shape[1]/2))
        )
        dummy_header = data_dict.copy()
        dummy_header["CRPIX1"] = dummy_array.shape[1]/2
        dummy_header["CRPIX2"] = dummy_array.shape[0]/2
        dummy_map = smap.Map(data_image, data_dict)
        with frames.Helioprojective.assume_spherical_screen(dummy_map.observer_coordinate):
            reference_submap = reference_smap.reproject_to(dummy_map.wcs)
        # On successive iterations, we're probably close enough to use the central default subtile:
        if i == 0:
            tol = tolerance
            st = subtile
        else:
            tol = None
            st = subtile
        _, shifts = _image_align(
            data_image, reference_submap.data,
            tolerance=tol, subtile=st
        )
        data_dict["CRVAL1"] = dummy_map.center.Tx.value + (shifts[1] * dummy_map.scale[0].value)
        data_dict["CRVAL2"] = dummy_map.center.Ty.value + (shifts[0] * dummy_map.scale[1].value)
    if rotation_correct:
        data_map = smap.Map(data_image, data_dict)
        with frames.Helioprojective.assume_spherical_screen(data_map.observer_coordinate):
            reference_submap = reference_smap.reproject_to(data_map.wcs)
        rotangle = determine_relative_rotation(data_image, reference_submap.data)
        data_dict["CROTA2"] = rotangle

    return data_dict

def align_rotate_map(
        data_image: np.ndarray, data_dict: dict, reference_smap: smap.GenericMap,
        niter: int=3, rotation_correct: bool=True,
        subtile: list | None=None, tolerance: list | None=None
    ) -> dict:
    """
    Prepares a rotated data image for alignment using the align_images function.
    This function wraps align_images after derotating the image to approximately Solar-North.
    This should be used for alignment in any case where the rotation angle is greater than about 5 degrees.
    :param data_image: numpy.ndarray
        2d numpy array containing original reduced image
    :param data_dict: dictionary
        Contains all neccessary FITS keywords
    :param reference_smap: sunpy.map.Map
        Reference image sunpy map. Typically assumed to be an HMI or similar full-disk map
    :param niter: int
        Passed to align_images. Performs sequential alignment, which may be useful in the case where the DST
        coordinates are extremely far off the correct values (more than ~1/2 the FOV)
    :param rotation_correct: bool
        If True, attempts a rotation correction with aligned image. Used to fine-tune CROTA2.
        Usually, the DST rotation angles are pretty good, but if you're not sure, this will fine-tune from
        +/- 5 degrees on an 0.2 degree grid.
    :return corrected_rotated_dictionary: dictionary
        Python dictionary containing updated FITS keywords.
    """
    data_map = smap.Map(data_image, data_dict)
    rotated_map = data_map.rotate(order=3)
    # Doing the rotation pads with nans.
    # We have to remove these, and the safest way is to replace them with the mean
    rotated_image = np.nan_to_num(rotated_map.data)
    rotated_image[rotated_image == 0] = np.nanmean(data_image)
    # Creating a derotated header.
    rotated_header = data_dict.copy()
    rotated_header["CRPIX1"] = rotated_image.shape[1]/2
    rotated_header["CRPIX2"] = rotated_image.shape[0]/2
    rotated_header["CROTA2"] = 0
    corrected_rotated_dictionary = align_images(
        rotated_image, rotated_header, reference_smap,
        niter=niter, rotation_correct=rotation_correct,
        subtile=subtile, tolerance=tolerance
    )
    # Updating the corrected dictionary to have the correct rotation angle and image sizes
    corrected_rotated_dictionary["CROTA2"] = data_dict["CROTA2"] + corrected_rotated_dictionary["CROTA2"]
    corrected_rotated_dictionary["CRPIX1"] = data_dict["CRPIX1"]
    corrected_rotated_dictionary["CRPIX2"] = data_dict["CRPIX2"]
    return corrected_rotated_dictionary

def determine_relative_rotation(data_image: np.ndarray, reference_image: np.ndarray) -> float:
    """
    Iteratively determines relative rotation of two images.
    Assumes both images are scaled the same, and are co-aligned within tolerances.
    Will crop larger array to smaller to account for single-pixel discrepancies

    :param data_image: numpy.ndarray
        2D numpy array containing image data
    :param reference_image: numpy.ndarray
        2D numpy array containing reference image data
    :return rotation: float
        Relative angle between the two images
    """
    if data_image.shape != reference_image.shape:
        if len(data_image.flatten()) > len(reference_image.flatten()):
            data_image = data_image[:reference_image.shape[0], :reference_image.shape[1]]
        else:
            reference_image = reference_image[:data_image.shape[0], :data_image.shape[1]]

    # Rotate between +/- 5 degrees
    rotation_angles = np.linspace(-2.5, 2.5, num=50)
    correlation_vals = np.zeros(len(rotation_angles))
    for i in range(len(rotation_angles)):
        rotated_image = scind.rotate(data_image, rotation_angles[i], reshape=False)
        correlation_vals[i] = np.nansum(
            rotated_image * reference_image
        ) / np.sqrt(
            np.nansum(rotated_image**2) * np.nansum(reference_image**2)
        )
    interp_range = np.linspace(rotation_angles[0], rotation_angles[-1], 1000)
    corr_interp = scinterp.interp1d(
        rotation_angles,
        correlation_vals,
        kind="quadratic"
    )(interp_range)
    rotation = interp_range[list(corr_interp).index(np.nanmax(corr_interp))]
    if (rotation == interp_range[0]) or (rotation == interp_range[-1]):
        warnings.warn("Rotation offset could not be determined by linear correlation. Defaulting to 0 degrees.")
        rotation = 0
    return rotation


def scale_image(arr: np.ndarray, vmin: float=0, vmax: float=1) -> np.ndarray:
    """Rescales image to be between vmin and vmax.
    Mostly for converting float images to int images where the original vmin/vmax are close"""
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def fetch_reference_image(metadata: dict, savepath: str=".", channel: str="HMI"):
    """
    Fetches temporally-similar full-disk reference image for alignment.
    :param metadata: dict
        Python dictionary of metadata. For our purposes, we really only need a time.
    :param savepath: str
        Path to save reference image. Defaults to working directory
    :param channel: str
        Reference image channel. Default is HMI for whitelight.
        Otherwise, assumes that the string is an AIA filter wavelength, e.g., "171"
    :return refmap: sunpy.map.Map
        Sunpy map of reference image.
    """
    timerange = (
        metadata["DATE-AVG"],
        (np.datetime64(metadata["DATE-AVG"]) + np.timedelta64(5, "m")).astype(str)
    )
    if channel.lower() == "hmi":
        fido_search = Fido.search(
            a.Time(timerange[0], timerange[1]),
            a.Instrument(channel),
            a.Physobs.intensity
        )
    else:
        fido_search = Fido.search(
            a.Time(timerange[0], timerange[1]),
            a.Instrument("AIA"),
            a.Wavelength(int(channel) * u.Angstrom)
        )
    dl_file = Fido.fetch(fido_search[0, 0], path=savepath)
    if len(dl_file.errors) > 0:
        warnings.warn(
            "Context image retrieval encountered a download error. "
            "This is known to happen on occasion. "
            "We will continue attempting to fetch the reference image."
            "This may take some time, depending on the host server."
        )
    while len(dl_file.errors) > 0:
        dl_file = Fido.fetch(dl_file)
    refmap = smap.Map(dl_file).rotate(order=3)
    return refmap


def find_best_image_speckle_alpha(flist: list) -> str:
    """
    From a list of ROSA/Zyla Level-1 or 1.5 files, finds the best reconstruction, assuming the
    "SPKLALPH" FITS header keyword is present. If it is not, you can usually find the best
    :param flist: list
        List of filepaths for iteration
    :return best_file: str
        File in list with best alpha
    """
    spkl_alphs = np.array([fits.open(i)[0].header["SPKLALPH"] for i in flist])
    best_file = flist[spkl_alphs.argmax()]
    return best_file


def update_imager_pointing_values(
        flist: list, pointing_info: dict,
        additional_rotation: float=0., progress=True
    ) -> None:
    """Updates a list of files with new pointing information,
    by differentially rotating the central point to the starttime
    of each file in the series.

    Parameters
    ----------
    flist : list
        List of FITS files to update
    pointing_info : dict
        Dictionary containing reference point
    additional_rotation : float, optional
        If given, adds the value to CROTA2 during update.
        Used in RosaZylaDestretch for applying rotation of a sub-channel relative to the one used for alignment
    progress : bool, optional
        If False, disables tqdm bar
    """
    ref_point = SkyCoord(
        pointing_info["CRVAL1"] * u.arcsec, pointing_info["CRVAL2"] * u.arcsec,
        obstime=pointing_info["DATE-AVG"], observer="earth", frame=frames.Helioprojective
    )
    base_time = np.datetime64(pointing_info["DATE-AVG"])
    for file in tqdm.tqdm(flist):
        with fits.open(file, mode="update") as hdul:
            img_time = np.datetime64(hdul[0].header["STARTOBS"])
            dt = (img_time - base_time).astype("timedelta64[ms]").astype(int) * u.ms
            rotated_point = SkyCoord(
                RotatedSunFrame(
                    base=ref_point, duration=dt
                ), observer="earth", obstime=img_time.astype(str)
            ).transform_to(frames.Helioprojective)
            hdul[0].header["CRVAL1"] = round(rotated_point.Tx.value, 3)
            hdul[0].header["CRVAL2"] = round(rotated_point.Ty.value, 3)
            hdul[0].header["CROTA2"] = round(pointing_info["CROTA2"], 3)
            hdul[0].header["CROTA2"] += additional_rotation
            prsteps = len([i for i in hdul[0].header.keys() if "PRSTEP" in i])
            hdul[0].header[f"PRSTEP{prsteps+1}"] = ("SOLAR-ALIGN", "Correct solar coords.")
            hdul[0].header["COMMENT"] = "POINTING UPDATED ON {0}".format(
                np.datetime64("now")
            )
            hdul.flush()
    return


def align_derotate_channel_images(channel_target: str, reference_target: str) -> tuple[np.ndarray, float]:
    """Intended for performing fine-alignment between ROSA/HARDcam channels using the AF resolution target
    images. A few key assumptions are made:
        1.) The input files are FITS with a complete set of WCS keywords
        2.) The input files are correctly oriented, both relative to the telescope
            optical axis, and with each other.

    Parameters
    ----------
    channel_target : str
        Path to target image from the channel we're concerned with aligning
    reference_target : str
        Path to target image for the reference channel

    Returns
    -------
    tuple[np.ndarray, float]
        Pixel shifts to align the channel with the reference (in channel pixel numbers)
        and rotation angle in degrees
    """

    channel_map = smap.Map(channel_target)
    reference_map = smap.Map(reference_target)

    channel_map_projected = channel_map.reproject_to(reference_map.wcs)

    chan_data = np.nan_to_num(channel_map_projected.data)
    targ_data = np.nan_to_num(reference_map.data)

    aligned_channel, shifts = _image_align(chan_data, targ_data)
    rotation_angle = determine_relative_rotation(aligned_channel, targ_data)

    relative_xscale = reference_map.scale[0].value / channel_map.scale[0].value
    relative_yscale = reference_map.scale[1].value / channel_map.scale[1].value

    final_pixel_shifts = np.array([shifts[0] * relative_yscale, shifts[1] * relative_xscale])
    return final_pixel_shifts, rotation_angle


def verify_alignment_accuracy(
    data_image: np.ndarray, data_dict: dict, reference_smap: smap.GenericMap
):
    """Verify alignment of data and raise a Pointing Error if the user is dissatisfied

    Parameters
    ----------
    data_image : np.ndarray
        2D numpy array with image data
    data_dict : dict
        Pointing information
    reference_smap : smap.GenericMap
        HMI map for pointing verification
    """
    data_map = smap.Map(data_image, data_dict)
    reference_submap = reference_smap.reproject_to(data_map.wcs)
    composite_map = smap.Map(data_map, reference_submap, composite=True)
    composite_map.set_levels(1, [25, 50, 75]*u.percent)
    fig = plt.figure(num="Pointing alignment check", figsize=(15, 5))
    ax_dat = fig.add_subplot(131, projection=data_map)
    data_map.plot(axes=ax_dat)
    ax_dat.set_xlabel(" ")
    ax_dat.set_title("Pointing-Corrected Data")
    ax_ref = fig.add_subplot(132, projection=reference_submap)
    reference_submap.plot(axes=ax_ref)
    ax_ref.set_ylabel(" ")
    ax_ref.set_title("Reference Map")
    ax_comp = fig.add_subplot(133, projection=data_map)
    composite_map.plot(axes=ax_comp)
    ax_comp.set_xlabel(" ")
    ax_comp.set_ylabel(" ")
    ax_comp.set_title("Data with Reference Contours")
    fig.tight_layout()
    plt.show(block=False)
    # I know this isn't elegant. It's a fallback. Sue me.
    answer = input("Are you satisfied with this level of alignment [y/n]? ")
    if "n" in answer.lower():
        raise PointingError
    return
