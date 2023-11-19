min_flux = 1e-99


class FluxBase(object):
    """This class is used for user to define the parameters used in the
    flux calculation

    Parameters
    ----------
    temp_grid : 1-D numpy array
        Temperature grid (in kelvin) for canonical flux calculation.
    energy_grid : 1-D numpy array
        Energy grid (in kelvin) for microcanonical flux calculation.
    angular_grid : 1-D numpy array
        The angular momentum for e-j resolved flux calculation.
    flux_type : String
        The target flux calculation type: CANONICAL, MICROCANONICAL, EJ-RESOLVED.
    flux_parameter : dictionary for define the sampling and accuracy of the flux
        calculation. will including:
        'flux_rel_err': the tolerance of flux relative error
        'pot_smp_max': the maximum number of sampling for each facet
        'pot_smp_min': the minimum number of sampling for each facet
        'tot_smp_max': the maximum number of sampling in total
        'tot_smp_min': the minimum number of sampling in total
        'smp_len': the number of sampling points for each call of a slave
    Attributes
    ----------
    flux_type
    temp_grid
    energy_grid
    angular_grid

    """

    def __init__(self, temp_grid=None, energy_grid=None, angular_grid=None,
                 flux_type=None, flux_parameter=None):

        self.flux_type = flux_type
        self.temp_grid = temp_grid
        self.energy_grid = energy_grid
        self.angular_grid = angular_grid
        self._flux_parameter = flux_parameter
        self._acct_num = 0
        self._fail_num = 0
        self._fake_num = 0
        self._close_num = 0
        self._face_num = 0

    def acct_smp(self):
        """successful sampling"""
        return self._acct_num

    def fail_smp(self):
        """failed sampling"""
        return self._fail_num

    def fake_smp(self):
        return self._fake_num

    def close_smp(self):
        """too close sampling, counted as space sampling"""
        return self._close_num

    def face_smp(self):
        """face out sampling, counted as space sampling"""
        return self._face_num

    def pot_smp(self):
        return self._acct_num + self._fail_num + self._fake_num

    def space_smp(self):
        return self._face_num + self._close_num

    def tot_smp(self):
        return self.pot_smp() + self.space_smp()

    def add_acct_smp(self, n):
        self._acct_num += n

    def add_fail_smp(self, n):
        self._fail_num += n

    def add_face_smp(self, n):
        self._face_num += n

    def add_close_smp(self, n):
        self._close_num += n

    def tol(self):
        """Return the flux accuracy defined by user"""

        return self._flux_parameter['flux_rel_err']

    def pot_max(self):
        """return the maximum number of sampling for each facet"""

        return self._flux_parameter['pot_smp_max']

    def pot_min(self):
        """return the minimum number of sampling for each facet"""

        return self._flux_parameter['pot_smp_min']

    def tot_max(self):
        """return the maximum number of total sampling"""
        return self._flux_parameter['tot_smp_max']

    def tot_min(self):
        """return the minimum number of total sampling"""
        return self._flux_parameter['tot_smp_min']

    def samp_len(self):
        return self._flux_parameter['smp_len']
