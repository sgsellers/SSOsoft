import numpy as np
import scipy.ndimage as scindi
import scipy.signal as scisig

class ToleranceException(Exception):
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


def _image_align(image, reference, tolerance=None):
    """
    Wraps Scipy's fftconvolve function to return an aligned image to a given reference.
    If the tolerance values are exceeded, attempt instead to align on the inverse image and reference.
    Currently, no derotation is implemented or planned. Get a better telescope if that's an issue for you.

    Parameters:
    -----------
    image : array-like
        2D image to align
    reference : array-like
        2D reference image to align to
    tolerance : None or 2-tuple with (ytolerance,xtolerance).

    Returns:
    --------
    aligned_image : array-like
        Aligned image
    """

    y0 = image.shape[0]/2.
    x0 = image.shape[0]/2.

    y0_ref = reference.shape[0]/2.
    x0_ref = reference.shape[1]/2.

    shifts = np.empty(2)
    aligned_image = np.zeros(image.shape)

    img = image - np.nanmean(image)
    ref = reference - np.nanmean(reference)

    correlation_map = scisig.fftconvolve(img, ref[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    shifts[0] = (y0 - y - (y0 - y0_ref))
    shifts[1] = (x0 - x - (x0 - x0_ref))

    if tolerance is not None:
        if (shifts[0] > tolerance[0]) or (shifts[1] > tolerance[1]):
            im_inv = 1./img
            ref_inv = 1./ref

            corr_map_inv = scisig.fftconvolve(im_inv, ref_inv[::-1, ::-1], mode='same')

            y, x = np.unravel_index(np.argmax(corr_map_inv), corr_map_inv.shape)
            shifts[0] = (y0 - y - (y0 - y0_ref))
            shifts[1] = (x0 - x - (x0 - x0_ref))

            if (shifts[0] > tolerance[0]) or (shifts[1] > tolerance[1]):
                raise ToleranceException(tolerance)

    scindi.shift(image, shifts, output=aligned_image, mode='constant', cval=image[0, 0])

    return aligned_image, shifts


class Destretch:
    def __init__(self, destretch_target, reference_image, kernel_sizes,
                 warp_vectors=None, return_vectors=False):
        """Initializing the destretch class WITHOUT a reference image.
        The reference image can be provided later, but does not have to be, as the class expects warp_params.

        Parameters:
        -----------
        destretch_target : array-like
            The 2d image to be destretched. Any looping must be done outside the class. I'll provide an example script.
        kernel_sizes : list
            The series of kernel sizes that the destretch is run over.
            A leading zeros indicates that fine aligment should be performed.
        reference_image : array-like
            If warp_vectors is provided, reference_image is used for alignment purposes.
            Else, reference_image is used to calculate warp_vectors.
        warp_vectors : None or array-like
            If an array, assumed to be an arroy of valid warp vectors.
            If none, then reference_image should be provided.
            If both are none, then an error is raised
        return_vectors : bool
            If true, return a list of vector arrays used in the destretch algorithm.
        """
        self.reference_image = reference_image
        self.destretch_target = destretch_target
        self.kernel = kernel_sizes
        self.target_size = self.destretch_target.shape
        if (type(self.kernel) == int) or (type(self.kernel) == float):
            self.kernel_size = self.kernel
        else:
            self.kernel_size = None
        self.warp_vectors = warp_vectors
        self.return_vectors = return_vectors
        self.wxy = None
        self.bound_size = None
        self.control_size = None
        self.destretch_image = None

    """Omnibus class for image destretching. Copied or adapted in large part from Sunspot's image destretch IDL tools
    from the 90's. See original reg.pro for details of individual functions.
    Can be parallelized when using affine transformation.

    Attributes:
    -----------
        reference_image (array-like) -- image used as a reference for calculating destretch vectors
        destretch_target (array-like) -- image to warp
        kernel_sizes (list of int or int) -- kernel size(s) to use in destretch. List is done sequentially.
            A leading 0 indicates that the image should be fine-aligned.
        warp_vectors (list of ndarray) -- list of destretch coordinates, for destretching one image relative to another
        ncores (int) -- Used in affine transform destretch for parallelization
        return_vectors (bool) -- If True, returns the list of destretch coordinates
    """

    def perform_destretch(self):
        """Perform image b-spline driven destretch. There are several cases to account for here:
        1.  Reference image and either single kernel size or list thereof are provided.
            In this case, perform iterative destretch as normal.
        2.  Warp vectors are provided.
            In this case, perform just the coordinate remapping using scipy
        """
        if self.warp_vectors is None:
            if self.kernel_size is not None:
                if self.kernel_size == 0:
                    self.destretch_target, shifts = _image_align(
                        self.destretch_target,
                        self.reference_image,
                        tolerance=(self.target_size[0]/4., self.target_size[1]/4.))
                    # To save the vectors, the shifts need to have the same size as everything else.
                    # So we create a dummy array, and fill it with the shifts
                    shifts_arr = np.zeros((2, self.target_size[0], self.target_size[1]))
                    shifts_arr[0, :, :] += shifts[0]
                    shifts_arr[1, :, :] += shifts[1]
                    self.warp_vectors = [shifts_arr]
                else:
                    rcps = self.mkcps()
                    tcps = self.cps(rcps)
                    self.warp_vectors = [self.contruct_bspline(rcps, tcps)]
                    self.destretch_image = scindi.map_coordinates(
                        self.destretch_target,
                        self.warp_vectors[0],
                        order=1,
                        mode='constant',
                        prefilter=False,
                        cval=self.destretch_target[0, 0]
                    )
            else:
                self.warp_vectors = []
                for i in range(len(self.kernel)):
                    if self.kernel[i] == 0:
                        self.destretch_target, shifts = _image_align(
                            self.destretch_target,
                            self.reference_image,
                            tolerance=(self.target_size[0]/4., self.target_size[1]/4.)
                        )
                        shifts_arr = np.zeros((2, self.target_size[0], self.target_size[1]))
                        shifts_arr[0, :, :] += shifts[0]
                        shifts_arr[1, :, :] += shifts[1]
                        self.warp_vectors.append(shifts_arr)
                    else:
                        self.kernel_size = self.kernel[i]
                        rcps = self.mkcps()
                        tcps = self.cps(rcps)
                        wv = self.contruct_bspline(rcps, tcps)
                        self.warp_vectors.append(wv)
                        self.destretch_image = scindi.map_coordinates(
                            self.destretch_target,
                            wv,
                            order=1,
                            mode='constant',
                            prefilter=False,
                            cval=self.destretch_target[0, 0]
                        )
        else:
            if len(self.warp_vectors.shape) > 3:
                for i in range(self.warp_vectors.shape[0]):
                    if self.kernel[i] == 0:
                        self.destretch_target = scindi.shift(
                            self.destretch_target,
                            self.warp_vectors[i][:, 0, 0],
                            mode='constant',
                            cval=self.destretch_target[0, 0]
                        )
                    else:
                        self.destretch_image = scindi.map_coordinates(
                            self.destretch_target,
                            self.warp_vectors[i],
                            order=1,
                            mode='constant',
                            prefilter=False,
                            cval=self.destretch_target[0, 0]
                        )
            else:
                # Edge case where there's a destretch array that corresponds only to a bulk shift
                if self.kernel == 0:
                    self.destretch_target = scindi.shift(
                        self.destretch_target,
                        self.warp_vectors[:, 0, 0],
                        mode='constant',
                        cval=self.destretch_target[0, 0]
                    )
                else:
                    self.destretch_image = scindi.map_coordinates(
                        self.destretch_target,
                        self.warp_vectors,
                        order=1,
                        mode='constant',
                        prefilter=False,
                        cval=self.destretch_target[0, 0]
                    )

        mask = self.destretch_image == 0.
        self.destretch_image[mask] = self.destretch_target[mask]
        if self.return_vectors:
            return self.destretch_image, self.warp_vectors
        else:
            return self.destretch_image

    def mkcps(self):
        """Compute reference image control points given the image size and the size of the desired kernel.

        Parameters:
        -----------
        None additional

        Returns:
        --------
        rcps : array-like
            Array of control points for the reference image. Has the shape (2,npointsx,npointsy).
            This defines a grid in xy corresponding to kernel centers.

        Defines:
        --------
        self.wxy : int
            Wander limits (should be larger than the kernel size)
        self.bound_size : tuple
            A 2-tuple containing the boundary x/y size
        self.control_size : tuple
            A 2-tuple containing control point size
        """

        a = 40./np.sqrt(2.)
        b = 20./np.log(2.)

        self.wxy = int(a*np.exp(-self.kernel_size/b) + self.kernel_size)
        if self.wxy % 2 == 1:
            self.wxy += 1

        cpx = (self.target_size[0] - self.wxy + self.kernel_size)/self.kernel_size
        cpy = (self.target_size[1] - self.wxy + self.kernel_size)/self.kernel_size

        bx = ((self.target_size[0] - self.wxy + self.kernel_size) % self.kernel_size) / 2
        by = ((self.target_size[1] - self.wxy + self.kernel_size) % self.kernel_size) / 2

        rcps = np.zeros((2, int(cpx), int(cpy)))
        
        ly = by
        hy = ly + self.wxy

        for i in range(0, int(cpy)):
            lx = bx
            hx = lx + self.wxy
            for j in range(0, int(cpx)):
                rcps[0, j, i] = (lx + hx) / 2
                rcps[1, j, i] = (ly + hy) / 2
                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        self.bound_size = (bx, by)
        self.control_size = (cpx, cpy)

        return rcps

    def cps(self, reference_control_points):
        """Compute the offsets for the current image from reference.

        Parameters:
        -----------
        reference_control_points : array-like
            Calculated using self.mkcps(), the control points of the reference image.

        Returns:
        --------
        target_control_points : array-like
            Repaired control points for the target image, corresponding to the same points on the reference grid
        """
        wander_mask = np.ones((self.wxy, self.wxy))
        win = self.doref(wander_mask)
        ans = self.cploc(win, wander_mask)
        target_control_points = self.repair(reference_control_points, ans)
        return target_control_points

    def doref(self, wander_mask):
        """Within the reference window around a control points, normalize the reference window, calculate the complex
        conjugate of the 2d FFT of the window for matching against the same value in the target image.

        Parameters:
        -----------
        wander_mask: array-like
            array of ones of the size of the control point plus border
        Returns:
        --------
        reference_window : array-like
            Complex conjugate of the 2d FFT of the reference image control point window
        """
        reference_window = np.zeros(
            (int(self.wxy),
             int(self.wxy),
             int(self.control_size[0]),
             int(self.control_size[1])),
            dtype=np.csingle)
        ly = int(self.bound_size[1])
        hy = ly + self.wxy
        for j in range(0, int(self.control_size[1])):
            lx = int(self.bound_size[0])
            hx = lx + self.wxy
            for i in range(0, int(self.control_size[0])):
                z = self.reference_image[lx:hx, ly:hy]
                z = z - np.sum(z)/(self.wxy**2)
                reference_window[:, :, i, j] = np.conjugate(np.fft.fft(z * wander_mask))
                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        return reference_window

    def cploc(self, reference_window, wander_mask):
        """Locate control points on the target image from the reference window

        Parameters:
        -----------
        reference_window : array-like
            Complex array containing the reference image windows from self.doref.
            These are the complex conjugi of the FFT of each window.
        mask : array-like
            Array of ones determining the size of the wander window
        Returns:
        --------
        estimated_target_control_points : array-like
            Estimated locations of control points for the target image. Will require repair for fallacious solutions
        """

        estimated_target_control_points = np.zeros((2, int(self.control_size[0]), int(self.control_size[1])))

        # Setup gradient correction
        # Friedrich Woeger's code

        t1d = (np.arange(0, self.wxy) / (self.wxy - 1) - 0.5) * 2.0
        tx = np.zeros((self.wxy, self.wxy))
        for i in range(tx.shape[0]):
            tx[i, :] = t1d
        ty = np.zeros((self.wxy, self.wxy))
        for i in range(ty.shape[1]):
            ty[:, i] = t1d

        nnx = np.sum(tx * tx)
        nny = np.sum(ty * ty)
        ly = int(self.bound_size[1])
        hy = ly + self.wxy
        for j in range(0, int(self.control_size[1])):
            lx = int(self.bound_size[0])
            hx = lx + self.wxy
            for i in range(0, int(self.control_size[0])):
                ss = self.destretch_target[lx:hx, ly:hy]
                k_tx = np.sum(ss*tx)/nnx
                k_ty = np.sum(ss*ty)/nny
                ss = (ss - (k_tx*tx + k_ty*ty))*wander_mask
                cc = scindi.shift(
                    np.abs(
                        np.fft.fft(
                            np.fft.fft(ss)*reference_window[:, :, i, j]
                        )
                    ),
                    [self.wxy/2, self.wxy/2],
                    mode='grid-wrap'
                )
                mx = np.amax(cc)
                mx_loc = np.array([np.where(cc == mx)[0][0], np.where(cc == mx)[1][0]])
                # Following accounts for fractions of a pixel of shift:
                if (mx_loc[0] * mx_loc[1] > 0) and (mx_loc[0] < (cc.shape[0] - 1)) and (mx_loc[1] < (cc.shape[1] - 1)):
                    denom = mx*2 - cc[mx_loc[0] - 1, mx_loc[1]] - cc[mx_loc[0] + 1, mx_loc[1]]
                    xfra = (mx_loc[0] - 0.5) + (mx - cc[mx_loc[0]-1, mx_loc[1]])/denom

                    denom = mx * 2 - cc[mx_loc[0], mx_loc[1] - 1] - cc[mx_loc[0], mx_loc[1] + 1]
                    yfra = (mx_loc[1] - 0.5) + (mx - cc[mx_loc[0], mx_loc[1] - 1]) / denom

                    mx_loc = np.array([xfra, yfra])

                estimated_target_control_points[0, i, j] = lx + mx_loc[0]
                estimated_target_control_points[1, i, j] = ly + mx_loc[1]

                lx += self.kernel_size
                hx += self.kernel_size
            ly += self.kernel_size
            hy += self.kernel_size
        return estimated_target_control_points

    def repair(self, rcps, tcps, tolerance=0.3):
        """Finds and fixes control points in the target image that drifted too far from the reference grid.
        By "fix" I mean, revert to the reference grid for that point.

        Parameters:
        -----------
        rcps : array-like
            Grid defining the control points on reference image, of size (2,cpx,cpy)
        tcps : array-like
            Should be the same shape as rcps, containing corresponding points in the target image
        tolerance : float, kwarg
            Allowable offset of reference and target control points. In the IDL version, this was hardcoded.
            Here, I'm passing it as a keyword argument. May need to be altered in severe flow environments.

        Returns:
        --------
        target_control_points : array-like
            Edited grid of target control point coordinates.
        """
        kkx = rcps[0, 1, 0] - rcps[0, 0, 0]
        kky = rcps[1, 0, 1] - rcps[1, 0, 0]
        limit = (np.amax([kkx, kky]) * tolerance) ** 2
        difference = tcps - rcps

        bad_points = np.where((difference[0, :, :] ** 2 + difference[1, :, :] ** 2) > limit)

        target_control_points = tcps
        for i in range(len(bad_points[0])):
            target_control_points[:, bad_points[0][i], bad_points[1][i]] = rcps[:, bad_points[0][i], bad_points[1][i]]
        return target_control_points

    def contruct_bspline(self, reference_control_points, target_control_points):
        """From the IDL reg.pro, construct a b-spline for scene destretch.
        This was originally adapted from Foley & Van Dam by Phil Wiborg and Thomas Rimmele.

        Args:
            reference_control_points (array-like): Numpy array of shape (2, x, y) containing control points
                of the reference image.
            target_control_points (array-like): Numpy array of shape (2, x, y) containing the
                corresponding control points of the target image

        Returns:
            destretch_coordinates (array-like): Numpy array with the same shape as the target image containing
                the destretch coordinates
        """
        def extend(rcps, tcps, pad=12):
            """Extend reference and target displacements to cover entire scene

            Args:
                rcps (array-like): Reference control points
                tcps (array-like): Reference control points
                pad (int): Value to pad the array by. Likely, this should be related to the kernel size

            Returns:
                ercps (array-like): Expanded reference control points
                etcps (array-like): Expanded target control points
            """
            ns = tcps.shape[0] + pad
            nt = tcps.shape[1] + pad

            ercps = np.zeros((ns, nt))
            rdiffs = rcps[1, 0] - rcps[0, 0]
            rzeroes = rcps[0, 0] - 3*rdiffs

            zx = np.arange(ns)*rdiffs + rzeroes

            for i in range(nt):
                ercps[:, i] = zx

            etcps = np.zeros((ns, nt))
            etcps[int(pad/2):-int(pad/2), int(pad/2):-int(pad/2)] = tcps - rcps

            x = etcps[:, int(pad/2)]
            for i in range(int(pad/2)):
                etcps[:, i] = x
            x = etcps[:, -int((pad / 2) + 1)]
            for i in range(int(pad / 2)):
                etcps[:, -(i+1)] = x
            etcps = etcps.T
            x = etcps[:, int(pad / 2)]
            for i in range(int(pad / 2)):
                etcps[:, i] = x
            x = etcps[:, -int((pad / 2) + 1)]
            for i in range(int(pad / 2)):
                etcps[:, -(i + 1)] = x
            etcps = etcps.T

            etcps += ercps

            return ercps, etcps

        def patch(compx, compy, s, t):
            """Fills in the section of B-spline

            Args:
                compx: x-coord patch after matrix ops
                compy: y-coord patch after matric ops
                s: sequence in y
                t: sequence in x

            Returns:
                dstr_patch: minute patch of destretch coordinates

            """
            dstr_patch = np.zeros((2, len(s), len(t)))
            ss = np.array(
                [s**3, s**2, s, np.ones(len(s))]
            ).T
            tt = np.array(
                [t**3, t**2, t, np.ones(len(t))]
            )
            dstr_patch[0, :, :] = np.matmul(ss, np.matmul(compx, tt))
            dstr_patch[1, :, :] = np.matmul(ss, np.matmul(compy, tt))
            return dstr_patch

        # First step, exaggerate the drift of the target control points.
        # This is apparently a kludge to increase the magnitude of error
        # since the curve generally doesn't pass through the tie points.
        # Essentially increases the displacement of every target control point by a factor of 1.1
        target_control_points = 1.1*(target_control_points - reference_control_points) + reference_control_points

        # ds and dt are distances in x/y between reference control points, as they're an evenly spaced grid
        ds = reference_control_points[0, 1, 0] - reference_control_points[0, 0, 0]
        dt = reference_control_points[1, 0, 1] - reference_control_points[1, 0, 0]

        # Expand reference and target control points to cover whole image
        # The IDL script used a pad of 6. I thought initially this was easily alterable
        # I'm no longer so certain. It seems to be baked into the way we calculate our splines
        # Regardless, I've got too many other things to do without dealing with this.
        ercps_x, etcps_x = extend(reference_control_points[0, :, :],
                                  target_control_points[0, :, :],
                                  pad=6)
        ercps_y, etcps_y = extend(reference_control_points[1, :, :].T,
                                  target_control_points[1, :, :].T,
                                  pad=6)
        ercps = np.zeros((2, ercps_x.shape[0], ercps_x.shape[1]))
        ercps[0, :, :] = ercps_x
        ercps[1, :, :] = ercps_y.T

        etcps = np.zeros((2, etcps_x.shape[0], etcps_x.shape[1]))
        etcps[0, :, :] = etcps_x
        etcps[1, :, :] = etcps_y.T
        # Basis matrices for uniform B-splines
        Ms = np.array(
            [
                [-1, 3, -3, 1],
                [3, -6, 0, 4],
                [-3, 3, 3, 1],
                [1, 0, 0, 0]
            ]
        ) / 6.
        MsTrans = Ms.T

        destretch_coords = np.zeros(
            (
                2,
                self.destretch_target.shape[0],
                self.destretch_target.shape[1]
            )
        )
        for v in range(target_control_points.shape[2]+int(6/2)):
            t0 = ercps[1, 1, v+1]
            tn = ercps[1, 1, v+2]
            # print(t0, tn)
            if (tn > 0) and (t0 < destretch_coords.shape[2]-1):
                t0 = np.nanmax(np.array([t0, 0]))
                tn = np.nanmin(np.array([tn, destretch_coords.shape[2]-1]))
                t = np.arange((tn-t0))/dt + (t0-ercps[1, 1, v+1])/dt
                for u in range(target_control_points.shape[1]+int(6/2)):
                    s0 = ercps[0, u+1, v+1]
                    sn = ercps[0, u+2, v+1]
                    if (sn > 0) and (s0 < destretch_coords.shape[1]-1):
                        s0 = np.nanmax([s0, 0])
                        sn = np.nanmin([sn, destretch_coords.shape[1]-1])
                        s = np.arange((sn-s0))/ds + (s0-ercps[0, u+1, v+1])/ds
                        compx = np.matmul(MsTrans, np.matmul(etcps[0, u:u+4, v:v+4], Ms)).reshape((4, 4))
                        compy = np.matmul(MsTrans,
                                          np.matmul(
                                              etcps[1, u:u+4, v:v+4],
                                              Ms
                                          )
                                          ).reshape((4, 4))
                        destretch_coords[:, int(s0):int(sn), int(t0):int(tn)] = patch(compx, compy, s, t)
        return destretch_coords
