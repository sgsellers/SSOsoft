import numpy as np
from scipy import ndimage


class ToleranceExceptionError(Exception):
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
        raise ToleranceExceptionError(tolerance)
    aligned = np.roll(image, (yshift, xshift))
    shifts = [yshift, xshift]
    return aligned, shifts

class Destretch:
    def __init__(
            self,
            destretch_target: np.ndarray, reference_image: np.ndarray,
            kernel_size: list, warp_vectors: None or np.ndarray=None,
            return_vectors: bool=False, repair_tolerance: float=0.5,
            overlap: bool=True, shift_smooth: int=3
    ) -> None:
        """
        Initializes destretch class.

        Parameters
        ----------
        destretch_target : numpy.ndarray
            2D numpy array containing image to destretch
        reference_image : numpy.ndarray
            2D numpy array containing reference image for destretch
        kernel_size : list
            List of kernel/subfield sizes. Will iterate over multiple kernel sizes.
            If a leading 0 is provided, performed global image alignment
        warp_vectors : None or numpy.ndarray
            If provided with the coordinate warp grid, will warp destretch_target by grid instead
            of determining the grid from reference_image, which is instead used for determining global shift
        return_vectors : bool
            If True, returns the (non-interpolated) warp grid for saving.
        repair_tolerance : float
            Fraction of a given window that sets the maximum shift of a subfield
        overlap : bool
            If true, subfields overlap
        shift_smooth : int
            Width of median filter on tile shift array
        """
        self.reference_image = reference_image
        self.destretch_target = destretch_target
        self.kernel_sizes = kernel_size
        self.target_size = self.destretch_target.shape
        self.warp_vectors = warp_vectors
        self.return_vectors = return_vectors
        self.repair_tolerance = repair_tolerance
        self.overlap = overlap
        self.shift_smooth = shift_smooth
        self.shifts = None
        self.kernel = None

        return

    def perform_destretch(self):
        if self.warp_vectors is None:
            warp_vectors = []
            for kernel in self.kernel_sizes:
                self.kernel = kernel
                if self.kernel == 0:
                    self.destretch_target, self.shifts = _image_align(
                        self.destretch_target,
                        self.reference_image,
                    )
                    warp_vectors.append(self.shifts)
                else:
                    ntiles, tile_corners = self.tile_coords()
                    ref_cube = self.segment_image(self.reference_image, ntiles, tile_corners)
                    img_cube = self.segment_image(self.destretch_target, ntiles, tile_corners)
                    correlation = self.compute_per_tile_correlation(ref_cube, img_cube)
                    tile_shifts = self.interpolate_peak(correlation)
                    registration = self.compute_registration_grid(tile_shifts, ntiles)
                    distortion_grid = self.compute_distortion_grid(registration, ntiles, tile_corners)
                    self.warp_image(distortion_grid)
                    warp_vectors.append(registration)
            if self.return_vectors:
                return self.destretch_target, warp_vectors
            else:
                return self.destretch_target
        else:
            # Since we need to reconstruct tile sizes, etc., must have kernels corresponding to warp vectors
            if len(self.kernel_sizes) != len(self.warp_vectors):
                raise ValueError("Kernel Sizes must correspond to warp vectors!")
            for i in range(len(self.kernel_sizes)):
                if self.kernel_sizes[i] == 0:
                    self.destretch_target, self.shifts = _image_align(
                        self.destretch_target,
                        self.reference_image
                    )
                else:
                    self.kernel = self.kernel_sizes[i]
                    ntiles, tile_corners = self.tile_coords()
                    distortion_grid = self.compute_distortion_grid(self.warp_vectors[i], ntiles, tile_corners)
                    self.warp_image(distortion_grid)
            if self.return_vectors:
                return self.destretch_target, self.warp_vectors
            else:
                return self.destretch_target

    def tile_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines number of tiles in y/x, and corner coords of each tile

        Returns
        -------
        ntiles : numpy.ndarray
            2-array of number of tiles in Y/X, determined from image shape,
            kernel size, and whether the tiles are requested to be overlapping.
        tile_corners : numpy.ndarray
            2D array of tile corner coordinates. Has shape (ntiles[0]*ntiles[1], 2)
            Tiles will not always be contiguous with the edges of the image.
            For ROSA cameras in particular, with their 1002x1004 chip size, power-of-2
            tiles will not be contiguous with the edges
        """
        if self.kernel % 2 != 0:
            # Clean the kernel sizes to be even
            self.kernel -= 1
        # Number of tiles without overlap
        ntiles = np.array(self.destretch_target.shape) // self.kernel
        # Total padding on both sides of image, i.e., offset from (0, 0) is half this
        edge_padding = np.array(self.destretch_target.shape) - ntiles*self.kernel
        if self.overlap:
            ntiles += ntiles - 1
            tiles_offset = self.kernel // 2
        else:
            tiles_offset = self.kernel
        tile_corners = np.empty((ntiles[0] * ntiles[1], 2), dtype=np.int16)
        ctr = 0
        for i in range(ntiles[0]):
            for j in range(ntiles[1]):
                tile_corners[ctr, :] = (edge_padding[0]//2 + i * tiles_offset,
                                       edge_padding[1]//2 + j * tiles_offset)
                ctr += 1
        return ntiles, tile_corners

    def segment_image(
            self, image: np.ndarray, ntiles: np.ndarray, tile_corners: np.ndarray
    ) -> np.ndarray:
        """
        Segments an image into an array of tiles with size self.kernel.

        Parameters
        ----------
        image : numpy.ndarray
            Image to tile
        ntiles : numpy.ndarray
            2-array of number of tiles in y/x
        tile_corners : np.ndarray
            Y/X coordinates of tile corners of shape (ntiles[0]*ntiles[1], 2)

        Returns
        -------
        tiled_image : np.ndarray
            Of shape (ntiles[0]* ntiles[1], self.kernel, self.kernel) containing tiled image.
        """
        if self.kernel % 2 != 0:
            # No idea how you get here with an incorrect kernel size
            # But just in case, kernels have to be even.
            self.kernel -= 1
        tiled_image = np.empty((ntiles[0] * ntiles[1], self.kernel, self.kernel), dtype=np.float32)
        for i in range(tiled_image.shape[0]):
            tiled_image[i, :, :] = image[
                                    tile_corners[i, 0]:tile_corners[i, 0] + self.kernel,
                                    tile_corners[i, 1]:tile_corners[i, 1] + self.kernel
                                   ]
        return tiled_image

    def compute_per_tile_correlation(
            self, reference_tiles: np.ndarray, target_tiles: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlation between each tile of a reference and target image

        Parameters
        ----------
        reference_tiles : np.ndarray
            Cube of tiles in a reference image, shape (ntilesX * ntilesY, self.kernel, self.kernel)
        target_tiles : np.ndarray
            Cube of tiles in a target image, shape same as reference_tiles

        Returns
        -------
        correlations : np.ndarray
            Correlation between target and reference tiles
        """
        # Step 1, apodization
        ref_apod = _window_apod(self.kernel, 0.125) # Narrow apodization for reference
        ref_apod /= np.mean(ref_apod)
        targ_apod = _window_apod(self.kernel, 0.4375)
        targ_apod /= np.mean(targ_apod)

        # Normalization stats
        ref_mean = np.mean(reference_tiles, axis=(1, 2), keepdims=True)
        ref_std = np.std(reference_tiles, axis=(1, 2), keepdims=True)
        targ_mean = np.mean(target_tiles, axis=(1, 2), keepdims=True)
        targ_std = np.std(target_tiles, axis=(1, 2), keepdims=True)

        # Correlation
        ref_fft = np.fft.rfftn(
            (reference_tiles - ref_mean)/ref_std * ref_apod[np.newaxis, :, :], axes=(1, 2)
        )
        targ_fft = np.fft.rfftn(
            (target_tiles - targ_mean)/targ_std * targ_apod[np.newaxis, :, :], axes=(1, 2)
        )
        correlations = np.fft.fftshift( # centering 0-freq
            np.fft.irfftn(
                np.conj(targ_fft) * ref_fft, axes=(1, 2) # correlation
            ), axes=(1, 2)
        ) / (self.kernel**2) # Scales as number of pixels in tile, need to norm

        return correlations

    def interpolate_peak(self, correlations: np.ndarray) -> np.ndarray:
        """
        Interpolate the peaks of the cross-correlation array.
        Also cleans up spurious cross-correlation results, i.e., if the solution
        lies outside self.repair_tolerance * self.kernel pixels away, defaults to no shifts.

        Parameters
        ----------
        correlations : numpy.ndarray
            3D cross-correlation maps. Shape (ntilesY * ntilesX, kernel, kernel)

        Returns
        -------
        tile_shifts : numpy.ndarray
            2D array of per-tile shifts. Shape (ntilesY * ntilesX, 2)
        """
        tile_shifts = np.zeros((correlations.shape[0], 2))
        for i in range(correlations.shape[0]):
            idx = np.argmax(correlations[i])
            yidx = idx // correlations.shape[1]
            xidx = idx % correlations.shape[1]
            # Check if shifts are outside of repair tolerance, and continue loop if they are
            if np.any(np.abs(np.array([yidx, xidx]) - self.kernel //2) >= self.repair_tolerance * self.kernel):
                continue
            # Interpolate peak
            yoffset = 0
            xoffset = 0
            if 0 < yidx < self.kernel - 1:
                denom = correlations[i, yidx - 1, xidx] - 2. * correlations[i, yidx, xidx] + correlations[
                    i, yidx + 1, xidx]
                yoffset = 0.5 * (correlations[i, yidx - 1, xidx] - correlations[i, yidx + 1, xidx]) / denom
                if (not np.isfinite(yoffset)) or yoffset > 1.:
                    yoffset = 0
            if 0 < xidx < self.kernel - 1:
                denom = correlations[i, yidx, xidx - 1] - 2. * correlations[i, yidx, xidx] + correlations[
                    i, yidx, xidx + 1]
                xoffset = 0.5 * (correlations[i, yidx, xidx - 1] - correlations[i, yidx, xidx + 1]) / denom
                if (not np.isfinite(xoffset)) or xoffset > 1.:
                    xoffset = 0
            tile_shifts[i, :] = (
                yidx - self.kernel // 2 + yoffset,
                xidx - self.kernel // 2 + xoffset
            )
        return tile_shifts

    def compute_registration_grid(self, shifts: np.ndarray, ntiles: np.ndarray) -> np.ndarray:
        """
        Remap shift array to an yshift/xshift grid, then median filter to smooth over the shifts.

        Parameters
        ----------
        shifts : numpy.ndarray
            2D array of shifts, shape (ntilesY * ntilesY, 2)
        ntiles : numpy.ndarray
            2-array of number of tiles in Y/X directions on image

        Returns
        -------
        registration_grid : numpy.ndarray
            Shift array remapped to correct dimensions (2, ntilesY, ntilesX)
        """
        ygrid = np.reshape(shifts[:, 0], ntiles)
        xgrid = np.reshape(shifts[:, 1], ntiles)
        ygrid = ndimage.median_filter(ygrid, size=self.shift_smooth)
        xgrid = ndimage.median_filter(xgrid, size=self.shift_smooth)
        registration_grid = np.array([ygrid, xgrid])
        return registration_grid

    def compute_distortion_grid(
            self, registration_grid: np.ndarray, ntiles: np.ndarray, tile_coords: np.ndarray
    ) -> np.ndarray:
        """
        Interpolates the coarse y/x shift grid to the original image shape.

        Parameters
        ----------
        registration_grid : numpy.ndarray
            Coarse shift grids, of shape (2, ntilesY, ntilesX)
        ntiles : numpy.ndarray
            2-array containing number of tiles in Y and X
        tile_coords : numpy.ndarray
            Y/X coordinates of tile corners, of shape (ntilesY * ntilesX, 2)

        Returns
        -------
        warp_grid : numpy.ndarray
            Warp coordinates for full image, shape (2, ny, nx)
        """
        # Sizes to interpolate to:
        y0 = np.min(tile_coords[:, 0])
        y1 = np.max(tile_coords[:, 0])
        dy = 2./self.kernel if self.overlap else 1./self.kernel
        x0 = np.min(tile_coords[:, 1])
        x1 = np.max(tile_coords[:, 1])
        dx = 2./self.kernel if self.overlap else 1./self.kernel
        # Interpolate grid
        # Mgrid returns more values than the original grid, but less than the
        # full image shape (because we had to crop for even kernels).
        meshgrid = np.mgrid[0:ntiles[0]:dy, 0:ntiles[1]:dx]
        ygrid = ndimage.map_coordinates(registration_grid[0], meshgrid, order=2, mode="nearest")
        xgrid = ndimage.map_coordinates(registration_grid[1], meshgrid, order=2, mode="nearest")
        # Center X/Y grids in arrays of the shape of the original image
        step = self.kernel // 2 if self.overlap else self.kernel
        warp_grid = np.zeros((2, *self.destretch_target.shape))
        warp_grid[
            0,
            int(y0 + step):int(y1 + step),
            int(x0 + step):int(x1 + step)
        ] = ygrid[:-step, :-step]
        warp_grid[
            1,
            int(y0 + step):int(y1 + step),
            int(x0 + step):int(x1 + step)
        ] = xgrid[:-step, :-step]

        return warp_grid

    def warp_image(self, warp_grid: np.ndarray) -> None:
        """
        Applies warp grid to self.destretch_target

        Parameters
        ----------
        warp_grid : numpy.ndarray
            Interpolated warp grid, shape (2, ny, nx)
        """
        image_grid = np.mgrid[:self.destretch_target.shape[0], :self.destretch_target.shape[1]]
        distortion_grid = image_grid - warp_grid
        self.destretch_target = ndimage.map_coordinates(
            self.destretch_target, distortion_grid,
            order=3, mode="nearest"
        )
        return
