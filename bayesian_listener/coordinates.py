import sofar
import numpy as np
import matplotlib.pyplot as plt
from bayesian_listener import metrics as mt
import warnings

class Coordinates:
    """
    A class for handling spatial coordinates in different coordinate systems.

    Supports three coordinate systems with specific angle ranges:
    - **Cartesian**: (x, y, z) - no angle restrictions
    - **Spherical**: (azimuth, elevation, radius)
        - azimuth: [-180°, 180°] or [-π, π] rad
        - elevation: [-90°, 90°] or [-π/2, π/2] rad
    - **Horizontal-polar**: (lateral, polar, radius)
        - lateral: [-90°, 90°] or [-π/2, π/2] rad
        - polar: [-90°, 270°) or [-π/2, 3π/2) rad

    Note: The constructor automatically wraps angles to the correct ranges
    for each coordinate system.
    SOFA files that use azimuth in [0°, 360°] will be wrapped to [-180°, 180°].
    """

    def __init__(self,
                 sofa_file = None,
                 positions = [],
                 convention = 'cartesian',
                 units = 'rad'):
        if sofa_file is not None:
            # handle sofa input
            if isinstance(sofa_file, str):
                self.sofa_file = sofa_file
                self.sofa_data = sofar.read_sofa(sofa_file, verbose = False)
            elif isinstance(sofa_file, sofar.Sofa):
                self.sofa_file = None
                self.sofa_data = sofa_file
            else:
                raise ValueError(
                    'sofa must be a string containing the path to a '
                    'sofa file or a sofar.Sofa object')

            self.positions = self.sofa_data.SourcePosition
            if np.any(abs(self.positions[:, 0]) > np.pi):
                self.positions[:, (0,1)] = np.deg2rad(self.positions[:, (0,1)])
            self.positions[:, 2] = 1.0 # force distance to be unitary
            self.convention = 'spherical'
        else:
            if convention not in [
                'cartesian',
                'spherical',
                'horizontal-polar',
                ]:
                raise ValueError(
                    f'Specified "convention " is not supported: {convention}')

            # Convert to numpy array if needed
            positions = np.asarray(positions)

            # Validate that positions is an Nx3 array
            if positions.size > 0:  # Only validate if not empty
                if positions.ndim == 1:
                    if positions.shape[0] == 3:
                        # Single position as 1D array, reshape to (1, 3)
                        positions = positions.reshape(1, 3)
                    else:
                        raise ValueError(
                            f"1D positions array must have exactly "
                            f"3 elements, got {positions.shape[0]}")
                elif positions.ndim == 2:
                    if positions.shape[1] != 3:
                        raise ValueError(
                            f"positions must be an Nx3 array, "
                            f"got shape {positions.shape}")
                else:
                    raise ValueError(
                        f"positions must be 1D (size 3) or 2D (Nx3), "
                        f"got {positions.ndim}D array")

            if (units == 'rad') & (np.abs(positions) > 2. * np.pi).any():
                warnings.warn('Coordinates ask for radiants!', stacklevel=2)

            if units == 'deg':
                positions[:, :-1] = np.deg2rad(positions[:, :-1])

            self.positions = positions
            self.convention = convention

        # Wrap angles to correct range for the coordinate system
        if (
            self.convention in ['spherical', 'horizontal-polar']
            and np.size(self.positions) > 0
        ):
            self.positions = Coordinates._wrap_angles(self.positions,
                                                      self.convention)



    @classmethod
    def with_repetitions(cls, estimations, convention = 'cartesian'):
        """
        Create a Coordinates instance from estimations with repetitions.

        Parameters
        ----------
        estimations : np.ndarray
            A 3D array of shape (N, reps, 3) containing N positions,
            each with 'reps' repetitions of 3 coordinates.
        convention : str
            The coordinate system convention of the estimations.
            Options: 'cartesian', 'spherical', 'horizontal-polar'.

        Returns
        -------
        Coordinates : Coordinates
            A Coordinates instance with the collapsed estimations.
        """
        # Collaps the repetitions dimension (from (N, reps, 3) to (N*reps, 3))
        assert estimations.ndim == 3, \
            "Estimations should be a 3D array with shape (N, reps, 3)"
        assert estimations.shape[2] == 3, \
            "Estimations should have 3 coordinates in the last dimension"
        estimations = estimations.reshape(-1, 3)

        # Create a new instance of Coordinates
        # with the given positions and convention
        return cls(positions=estimations, convention=convention)

    def normalise(self):
        """
        Normalize the positions by setting the radius to 1
        in spherical coordinates.
        """
        positions = self.convert('spherical')
        positions[:, 2] = 1
        self.positions = positions
        self.convention = 'spherical'

    def convert(self, to_convention):
        """
        Convert the coordinates to a different convention.

        Parameters
        ----------
        to_convention : str
            The target coordinate system convention.
            Options: 'cartesian', 'spherical', 'horizontal-polar'.

        Returns
        -------
        np.ndarray : The converted coordinates.
        """
        if to_convention not in [
            'cartesian',
            'spherical',
            'horizontal-polar',
            ]:
            raise ValueError(
                f"Specified 'to_convention' is not supported: "
                f"{to_convention }")

        temp = self.positions.copy()
        if temp.ndim == 1:
            temp = np.expand_dims(temp, axis=0)

        if self.convention != to_convention:
            if self.convention in ['spherical']:
                temp =  Coordinates.sph2cart(temp[:, 0],
                                             temp[:, 1],
                                             temp[:, 2])
            elif self.convention == 'horizontal-polar':
                hor2sph_result = Coordinates.hor2sph(temp[:, 0], temp[:, 1])
                az = hor2sph_result[:, 0]
                el = hor2sph_result[:, 1]
                temp = Coordinates.sph2cart(az, el, temp[:, 2])

            if to_convention in ['spherical', 'geodesic']:
                temp = Coordinates.cart2sph(temp[:, 0], temp[:, 1], temp[:, 2])
            elif to_convention == 'horizontal-polar':
                temp = Coordinates.cart2sph(temp[:, 0], temp[:, 1], temp[:, 2])

                sph2hor_result = Coordinates.sph2hor(temp[:, 0], temp[:, 1])
                temp[:, 0] = sph2hor_result[:, 0]
                temp[:, 1] = sph2hor_result[:, 1]

        return temp

    def sph(self):
        """
        Get the spherical coordinates in degrees.

        Returns
        -------
        np.ndarray : The spherical coordinates
        (azimuth, elevation) in degrees.
        """
        positions = self.convert('spherical')
        positions[:, 0] = np.mod(positions[:, 0] + np.pi, 2 * np.pi) - np.pi
        return np.rad2deg(positions[:, (0, 1)])

    def hpo(self):
        """
        Get the horizontal-polar coordinates in degrees.

        Returns
        -------
        np.ndarray : The horizontal-polar coordinates
        (latitude, polar angle) in degrees.
        """
        positions = self.convert('horizontal-polar')
        positions[:, 0] = np.mod(positions[:, 0] + np.pi, 2 * np.pi) - np.pi
        return np.rad2deg(positions[:, (0, 1)])

    def az(self):
        """
        Get the azimuth angles in radians.

        Returns
        -------
        np.ndarray : The azimuth angles in radians.
        """
        dirs = self.sph()
        return dirs[:, 0]

    def el(self):
        """
        Get the elevation angles in radians.

        Returns
        -------
        np.ndarray : The elevation angles in radians.
        """
        dirs = self.sph()
        return dirs[:, 1]

    def lat(self):
        """
        Get the latitude angles in radians.

        Returns
        -------
        np.ndarray : The latitude angles in radians.
        """
        dirs = self.hpo()
        return dirs[:, 0]

    def pol(self):
        """
        Get the polar angles in radians.

        Returns
        -------
        np.ndarray : The polar angles in radians.
        """
        dirs = self.hpo()
        return dirs[:, 1]

    def find(self, coords_search):
        """
        Find the closest coordinates in the current set of coordinates.

        Parameters
        ----------
        coords_search : Coordinates
            The coordinates to search for.

        Returns
        -------
        coords_found : Coordinates
            The closest coordinates found.
        idx : np.ndarray
            The indices of the closest coordinates found.
        """
        pos = self.convert('cartesian')
        pos_search = coords_search.convert('cartesian')

        idx = np.zeros(pos_search.shape[0], dtype=int)

        for ii in range(pos_search.shape[0]):
            dist = np.sum((pos - pos_search[ii, :]) ** 2, axis=1)
            idx[ii] = np.argmin(dist)

        coords_found = Coordinates(positions = self.positions[idx, :],
                                   convention = self.convention)
        return coords_found, idx

    def plot(self, values = [], points = None):
        """
        Visualize the positions as a 3D scatter plot.

        Parameters
        ----------
        values : array-like, optional
            Numerical values used to color the scatter points.
            If None or empty, all points are plotted with the same color.
        points : array-like of shape (3,), optional
            Optional 3D point to plot as an estimated direction
            from the origin.
        """
        if values is None or len(values) == 0:
            values = np.ones(self.positions.shape[0])
        else:
            assert(len(values) == self.positions.shape[0])

        r = self.convert('cartesian')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(r[:,0],
                   r[:, 1],
                   r[:, 2],
                   c = values,
                   s=20,
                   alpha = .5,
                   label='Log posterior',
                   )
        ax.plot([0, 1], [0,0],zs=[0,0], c='red', label='Front direction')

        if points is not None:
            ax.plot(xs=[0, points[0]],
                    ys=[0, points[1]],
                    zs=[0, points[2]],
                    c='blue',
                    label='Estimated direction',
                    )

        ax.view_init(elev=0, azim=0)
        ax.set_box_aspect([1, 1, 1])
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Values')
        ax.legend()
        plt.show()

    def print(self):
        """Print the spherical coordinates in a grid format."""
        grid = self.sph()

        print("Positions (spherical):")
        for row in grid:
            for item in row:
                print(f"{item:2}", end=" ")  # Format with width 2
            print()  # New line after each row


    def localization_error(self, estimations, metric, auxiliary_output=False):
        """
        Compute the localization error between the current coordinates
        and the provided estimations using the specified metric.

        Parameters
        ----------
        estimations : Coordinates
            The estimated coordinates to compare against.
        metric : str or callable
            The metric to use for error computation.
            If a string, it should be a registered metric name.
            If callable, it should be a function that takes two
            Coordinates instances and returns the error value.
        auxiliary_output : bool, optional
            If True, return auxiliary output from the metric function.
            Default is False.

        Returns
        -------
        float or tuple :
            The computed localization error.
            If auxiliary_output is True, returns a tuple
            (error_value, auxiliary_data).
        """
        if not isinstance(estimations, Coordinates):
            raise ValueError(
                "estimations must be an instance of Coordinates class")
        if self.positions.shape != estimations.positions.shape:
            raise ValueError("Shape mismatch")

        # Case 1: metric is a custom function
        if callable(metric):
            return metric(self.positions, estimations.positions)

        # Case 2: metric is a string, but not registered in METRIC_FUNCTIONS
        if metric not in mt.METRIC_FUNCTIONS:
            raise ValueError(
                f"Unknown metric: {metric}. Available metrics are: "
                f"{list(mt.METRIC_FUNCTIONS.keys())}")

        # Case 3: metric is a string and registered in METRIC_FUNCTIONS
        expected_coord_convention = \
            mt.get_metric_metadata(metric)['coord_convention']
        expected_unit = mt.get_metric_metadata(metric)['input_unit']

        # Convert both self and estimations
        # to the expected coordinate convention
        converted_self = self.convert(expected_coord_convention)
        converted_estim = estimations.convert(expected_coord_convention)

        # TODO: Evaluate the unit if necessary

        value, aux_out = \
            mt.METRIC_FUNCTIONS[metric](converted_self, converted_estim)

        return (value, aux_out) if auxiliary_output else value


    @staticmethod
    def help_on_metric(name=None):
        """
        Print help information about a specific metric
        or list all available metrics if no name is provided.
        """
        mt.describe_metrics(name)

    @staticmethod
    def get_metric_metadata(name):
        """
        Get metadata for a specific metric.
        """
        return mt.get_metric_metadata(name)


    @staticmethod
    def _wrap_angles(positions, coord_convention ):
        """
        Wrap angles to the correct range for the specified coordinate system.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 3) containing positions in
            the specified coordinate system
        coord_convention : str
            convention of coordinate system:
            'spherical' or 'horizontal-polar'

        Returns
        -------
        np.ndarray : positions with angles wrapped to the correct ranges
        """
        positions = positions.copy()

        if coord_convention == 'spherical':
            # Spherical: azimuth in [-pi, pi], elevation in [-pi/2, pi/2]
            positions[:, 0] = np.mod(positions[:, 0] + np.pi, 2*np.pi) - np.pi
            # Elevation is naturally constrained
            # by arcsin in cart2sph, but clip for safety
            positions[:, 1] = np.clip(positions[:, 1], -np.pi/2, np.pi/2)

        elif coord_convention == 'horizontal-polar':
            # Horizontal-polar:
            # lateral in [-pi/2, pi/2], polar in [-pi/2, 3*pi/2)
            positions[:, 0] = np.clip(positions[:, 0], -np.pi/2, np.pi/2)
            positions[:, 1] = \
                np.mod(positions[:, 1] + np.pi/2, 2*np.pi) - np.pi/2

        return positions

    @staticmethod
    def sph2cart(azimuth, elevation, r):
        """
        Transform spherical to cartesian coordinates.

        Parameters
        ----------
        azimuth : np.ndarray
            azimuth (in radians)
        elevation : np.ndarray
            elevation (in radians)
        r : np.ndarray
            radius (in meters)

        Returns
        -------
        x, y, z : np.ndarray
            cartesian coordinates (in meters)
        """
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        return np.array([x, y, z]).T

    @staticmethod
    def cart2sph(x, y, z):
        """
        Transform cartesian to spherical coordinates.

        Parameters
        ----------
        x : np.ndarray
            x coordinate (in meters)
        y : np.ndarray
            y coordinate (in meters)
        z : np.ndarray
            z coordinate (in meters)

        Returns
        -------
        azimuth : np.ndarray
            azimuth (in radians)
        elevation : np.ndarray
            elevation (in radians)
        r : np.ndarray
            radius (in meters)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.atan2(z, np.sqrt(x**2 + y**2))
        return np.array([azimuth, elevation, r]).T

    @staticmethod
    def sph2hor(azi, ele):
        """
        Transform spherical to horizontal-polar coordinates.

        Parameters
        ----------
        azi : np.ndarray
            azimuth (in radians)
        ele : np.ndarray
            elevation (in radians)

        Returns
        -------
        lat : np.ndarray
            lateral angle (-pi/2 <= lat <= pi/2)
        pol : np.ndarray
            polar angle (-pi/2 <= pol < 3*pi/2)
        """
        # Convert spherical to cartesian
        x, y, z = Coordinates.sph2cart(azi, ele, np.ones_like(azi)).T

        # Remove noise below eps
        x[np.abs(x) < np.finfo(float).eps] = 0
        y[np.abs(y) < np.finfo(float).eps] = 0
        z[np.abs(z) < np.finfo(float).eps] = 0

        # Interpret horizontal polar format as rotated spherical coordinates
        # with negative azimuth direction
        pol, nlat, r = Coordinates.cart2sph(x, z, -y).T
        lat = -nlat

        # Adjust polar angle range from [-pi, pi] to [-pi/2, 3*pi/2)
        pol = np.mod(pol + np.pi/2, 2*np.pi) - np.pi/2

        return np.array([lat, pol]).T

    @staticmethod
    def hor2sph(lat, pol):
        """
        Transform horizontal-polar to spherical coordinates.

        Parameters
        ----------
        lat : np.ndarray
            lateral angle (-pi/2 <= lat <= pi/2, in radians)
        pol : np.ndarray
            polar angle (-pi/2 <= pol < 3*pi/2, in radians)

        Returns
        -------
        azi : np.ndarray
            azimuth (-pi <= azi <= pi, in radians)
        ele : np.ndarray
            elevation (-pi/2 <= ele <= pi/2, in radians)
        """
        x, nz, y = Coordinates.sph2cart(-pol, lat, np.ones_like(lat)).T

        # Convert back to spherical coordinates
        azi, ele, r = Coordinates.cart2sph(x, y, -nz).T

        # Azimuth is already in [-pi, pi] from cart2sph
        return np.array([azi, ele]).T
