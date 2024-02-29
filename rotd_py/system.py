import numpy as np
from enum import Enum
from ase import Atoms, units


class MolType(Enum):
    MONOATOMIC = 1
    LINEAR = 2
    NONLINEAR = 3


class SampTag(Enum):
    SAMP_ATOMS_CLOSE = 1
    SAMP_FACE_OUT = 2
    SAMP_SUCCESS = 3


class FluxTag(Enum):
    FLUX_TAG = 1
    SURF_TAG = 2
    STOP_TAG = 3


class Surface(object):
    """Class used for defining the dividing surface for 2 fragments system.

    Parameters
    ----------
    pivot_points : dictionary
        Dictionary of the position of the pivot point for each fragment.
        key: the index of fragment, value: the relative coordinates for each
        pivot point for that fragment
        eg:     pivot_points = {'0': [[0., 0., -0.5],
                                    [0., 0., 0.5]],
                             '1': [[0., 1., 0.]]}
    distances : 2-d numpy array
        distances[i][j] is the distance between the i-th pivot point
        of frag_1 and the j-th pivot point of frag_2

    Attributes
    ----------
    num_pivot_point : 1-D numpy array
        The number of pivot points for each fragments. (eg: [2,1])
    num_face : int
        The total number of pivot point pairs of two fragments, (eg: 2)
    curr_face : int
        The index of the face which is being calculated.
    pivot_points
    distances

    """

    def __init__(self, pivotpoints=None, distances=None):
        if len(pivotpoints) != len(distances.shape):
            raise ValueError("The dimension of fragment and \
                             distance does not consistent")

        self.pivotpoints = pivotpoints
        self.distances = distances
        self.num_pivot_points = [len(self.pivotpoints[f'{i}'])
                                 for i in range(0, len(self.pivotpoints))]
        self.num_faces = 1
        for i in range(0, len(self.num_pivot_points)):
            self.num_faces *= self.num_pivot_points[i]
        self._curr_face = 0
        self.pot_var = np.inf
        self.vol_var = np.inf

    def get_num_faces(self):
        """Return the total number of facets for this system. """
        return self.num_faces

    def pivot_index(self, face):
        """Get the pivot point index for each fragment for the input face. """
        if face < 0 or face >= self.num_faces:
            raise ValueError("Invalid face index")
        i = face % self.num_pivot_points[0]
        j = face // self.num_pivot_points[0]
        return i, j

    def dist(self, i, j):
        """Return the distance between the i-th pivot point of fragment 1 and
            j-th pivot point of fragment 2"""
        return self.distances[i][j]

    def get_dist(self, face):
        """Return pivot points distance for given face index """
        i, j = self.pivot_index(face)
        return self.dist(i, j)

    def set_face(self, face):
        """Set the current target face for flux calculation. """
        if face < 0 or face >= self.num_faces:
            raise ValueError("Invalid face index")
        self._curr_face = face

    def get_curr_face(self):
        """ Return the current face index """
        return self._curr_face

    def get_pivot_point(self, frag_index, face):
        """Return the pivot point coordinates for fragment "frag_index"
            at "face" index.
        """

        i, j = self.pivot_index(face)
        #print(i)
        #print(j)
        if frag_index == 0:
            return self.pivotpoints['0'][i]
        else:
            return self.pivotpoints['1'][j]
