import pyroomacoustics as pra
from scipy.special import sph_harm
from spaudiopy.sph import sh_matrix
import numpy as np
import matplotlib.pyplot as plt
from pyroomacoustics import Directivity, all_combinations
import matplotlib.pyplot as plt


def spherical_harmonics(order, theta, phi):
    """Compute spherical harmonics up to a certain order."""
    harmonics = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            Y_nm = sph_harm(m, n, phi, theta)
            harmonics.append(Y_nm)
    return np.array(harmonics)

def HOA_array(samples=25, radius=1.0, n_order=4, azimuth_rotation=0.0):
    """
    Generate microphone positions and orientations very close to each other
    for an n-order Ambisonics array. Each microphone position is also assigned
    a spherical harmonic degree (n, m).

    Parameters
    ----------
    samples : int
        The number of samples (microphone positions).
    radius : float
        The radius of the sphere.
    n_order : int
        The order of Ambisonics (determines the degree (n, m) assignments).
    Returns
    -------
    positions : np.ndarray
        Array of 3D positions for the microphones.
    orientations : list of DirectionVector
        List of DirectionVector objects corresponding to each microphone position.
    degrees : list of tuples
        List of (n, m) degree pairs for each microphone.
    """
    positions = np.zeros((samples, 3))
    azimuth_rotation = float(azimuth_rotation)
    orientations = []
    degrees = []

    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    # Generate Fibonacci positions on the sphere
    for i in range(samples):
        y = np.array(((i * offset) - 1) + (offset / 2))
        r = np.sqrt(1 - y ** 2)

        phi = i * increment + azimuth_rotation

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        # 3D position on the sphere
        positions[i] = np.array([x, y, z]) * radius

        # Convert (x, y, z) to azimuth and colatitude
        azimuth = np.arctan2(y, x)  # Azimuth (angle in the XY plane)
        colatitude = np.arccos(z / radius)  # Colatitude (angle from zenith)

        # Store as DirectionVector
        orientation = pra.directivities.DirectionVector(azimuth, colatitude, degrees=False)
        orientations.append(orientation)

    # Assign spherical harmonic degrees (n, m) to each microphone
    degree_index = 0
    for n in range(n_order + 1):  # n goes from 0 to n_order
        for m in range(-n, n + 1):  # m goes from -n to +n
            if degree_index < samples:
                degrees.append((n, m))
                degree_index += 1
            else:
                break

    return positions, orientations, degrees


class SphericalHarmonicDirectivity(Directivity):
    """
    Object for directivities based on spherical harmonics.

    Parameters
    ----------
    orientation : DirectionVector
        Indicates the direction of the pattern.
    n : int
        The order of the spherical harmonic.
    m : int
        The degree of the spherical harmonic.
    """

    def __init__(self, orientation, n, m, azimuth_rotation=0.0):
        super().__init__()
        self._orientation = orientation

        self._n = n
        self._m = m
        self._acn = n * (n + 1) + m
        self._azimuth_rotation = float(azimuth_rotation)

    # -----------------------------------------------------
    # REQUIRED by new pyroomacoustics
    # -----------------------------------------------------
    @property
    def is_impulse_response(self):
        return False

    @property
    def filter_len_ir(self):
        return 1

    def sample_rays(self, n_rays):
        # deterministic sampling (Fibonacci sphere)
        i = np.arange(n_rays)
        phi = (i * (np.pi * (3 - np.sqrt(5)))) % (2 * np.pi)
        cos_theta = 1 - (2 * i + 1) / n_rays
        theta = np.arccos(cos_theta)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        directions = np.vstack((x, y, z)).T

        gains = self.get_response(phi, theta, degrees=False)
        gains = np.abs(gains)

        return directions, gains
    # -----------------------------------------------------    
    
    def sn3d_normalization(self):
        """
        Compute the N3D normalization factor for a given degree n and order m.
        
        Parameters:
        -----------
        n : int
            Degree of the spherical harmonic.
        m : int
            Order of the spherical harmonic (can be negative or positive).
        
        Returns:
        --------
        float
            SN3D normalization factor.
        """
        return 1/np.sqrt(2*self._n + 1) 

    def complex_to_real_sph_harm(self, m, n, azimuth, colatitude):
        """
        Convert complex spherical harmonics to real spherical harmonics.

        Parameters
        ----------
        m : int
            Order of the spherical harmonic.
        n : int
            Degree of the spherical harmonic.
        azimuth : float or array_like
            Azimuth angle(s) in radians.
        colatitude : float or array_like
            Colatitude angle(s) in radians.

        Returns
        -------
        Y_real : float or array_like
            Real spherical harmonics value(s).
        """
        Y_complex = sph_harm(m, n, azimuth, colatitude)

        if m > 0:
            Y_real = np.sqrt(2) * (-1)**m * np.real(Y_complex)
        elif m < 0:
            Y_real = np.sqrt(2) * (-1)**abs(m) * np.imag(Y_complex)
        else:  # m == 0
            Y_real = np.real(Y_complex)

        return Y_real

    '''
    def get_response(
        self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True
    ):
        """
        Get response for provided angles using spherical harmonics.

        Parameters
        ----------
        azimuth : array_like
            Azimuth in degrees or radians.
        colatitude : array_like, optional
            Colatitude in degrees or radians.
        magnitude : bool, optional
            Whether to return the magnitude of the response.
        frequency : float, optional
            For which frequency to compute the response (not used for spherical harmonics).
        degrees : bool, optional
            Whether provided angles are in degrees.

        Returns
        -------
        resp : :py:class:`~numpy.ndarray`
            Spherical harmonic response at provided angles.
        """
        if colatitude is not None:
            assert len(azimuth) == len(colatitude)


        if degrees:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)

        # Default to elevation if colatitude is not provided
        if colatitude is None:
            colatitude = np.pi / 2

        colatitude = np.clip(colatitude, a_min=0, a_max=np.pi)

        #if azimuth.min()<0:
        #    azimuth = azimuth + np.pi
        azimuth = azimuth % (2*np.pi)

        assert colatitude.max() <= np.pi and colatitude.min()>=0

        assert azimuth.max() <= 2*np.pi and azimuth.max() >= 0

        resp = self.complex_to_real_sph_harm(self._m, self._n, azimuth, colatitude) #*0.5#* self.sn3d_normalization() 

        if magnitude:
            return np.abs(resp)
        else:
            return resp
    '''    

    def get_response(self, azimuth, colatitude=None, magnitude=False, frequency=None, degrees=True):

        azimuth = np.asarray(azimuth)

        if degrees:
            azimuth = np.radians(azimuth)

        if colatitude is None:
            colatitude = np.pi / 2 * np.ones_like(azimuth)
        else:
            colatitude = np.asarray(colatitude)
            if degrees:
                colatitude = np.radians(colatitude)

        # Express the world-frame arrival azimuth in the rotated array frame.
        # A positive array rotation means the local pattern is rotated counter-clockwise
        # around +z, hence the inverse rotation is applied to incoming directions.
        azimuth = (np.asarray(azimuth) - self._azimuth_rotation) % (2 * np.pi)
        colatitude = np.asarray(colatitude)

        # ACN + N3D real SH matrix
        Y = sh_matrix(
            self._n,
            azimuth,
            colatitude,
            sh_type="real"
        )

        resp = Y[:, self._acn]

        return np.abs(resp) if magnitude else resp
    
    def plot_response(
        self, azimuth, colatitude=None, degrees=True, ax=None, offset=None, axis_at_zero=False
    ):
        """
        Plot spherical harmonic directivity response at specified angles.

        Parameters
        ----------
        azimuth : array_like
            Azimuth values for plotting.
        colatitude : array_like, optional
            Colatitude values for plotting. If not provided, 2D plot.
        degrees : bool
            Whether provided values are in degrees (True) or radians (False).
        ax : axes object, optional
        offset : list, optional
            3-D coordinates of the point where the response needs to be plotted.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`
            The axes object with the plot.
        """
        import matplotlib.pyplot as plt

        if offset is not None:
            x_offset = offset[0]
            y_offset = offset[1]
            z_offset = offset[2] if len(offset) > 2 else 0
        else:
            x_offset = y_offset = z_offset = 0

        # Convert angles to radians if necessary
        if degrees:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)

        # 3D plot if colatitude is provided, otherwise 2D plot
        if colatitude is not None:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection="3d")

            # Compute response over all combinations of azimuth and colatitude
            spher_coord = all_combinations(azimuth, colatitude)
            azi_flat = spher_coord[:, 0]
            col_flat = spher_coord[:, 1]

            # Calculate the spherical harmonic response
            resp = self.get_response(azimuth=azi_flat, colatitude=col_flat, magnitude=True, degrees=False)

            RESP = resp.reshape(len(azimuth), len(colatitude))

            # Convert spherical coordinates to Cartesian for plotting
            AZI, COL = np.meshgrid(azimuth, colatitude)
            X = RESP.T * np.sin(COL) * np.cos(AZI) + x_offset
            Y = RESP.T * np.sin(COL) * np.sin(AZI) + y_offset
            Z = RESP.T * np.cos(COL) + z_offset


            ax.plot_surface(X, Y, Z)
            

            if axis_at_zero:
                val = [1,0,0]
                for v in range(3):
                    x = [val[v-0], -val[v-0]]
                    y = [val[v-1], -val[v-1]]
                    z = [val[v-2], -val[v-2]]
                    ax.plot(x,y,z,'k-', linewidth=0.25)
            ax.axis('off')
            ax.grid(b=None)

            ax.set_title(f"Spherical Harmonic Order {self._n}, Degree {self._m}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        else:
            if ax is None:
                fig = plt.figure()
                ax = plt.subplot(111)

            # 2D plot in the XY plane (azimuth-only)
            resp = self.get_response(azimuth=azimuth, magnitude=True, degrees=False)
            X = resp * np.cos(azimuth) + x_offset
            Y = resp * np.sin(azimuth) + y_offset
            ax.plot(X, Y)

            ax.set_title(f"Spherical Harmonic Order {self._n}, Degree {self._m}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        return ax
