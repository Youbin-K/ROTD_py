from rotd_py.fragment.nonlinear import Nonlinear
from rotd_py.fragment.linear import Linear
from rotd_py.fragment.monoatomic import Monoatomic
from rotd_py.fragment.slab import Slab
from rotd_py.system import Surface
from rotd_py.sample.multi_sample import MultiSample
from rotd_py.flux.fluxbase import FluxBase
from ase.constraints import FixAtoms
from ase import Atoms, atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from amp import Amp
import numpy as np

# from rotd_py.multi import Multi
from rotd_py.new_multi import new_Multi
from rotd_py.multi import Multi

def generate_grid(start, interval, factor, num_point):
    """Return a grid needed for the simulation of length equal to num_point

    @param start:
    @param interval:
    @param factor:
    @param num_point:
    @return:
    """
    i = 1
    grid = [start]
    for i in range(1, num_point):
        start += interval
        grid.append(start)
        interval = interval * factor
    return np.array(grid)


# temperature, energy grid and angular momentum grid
temperature = generate_grid(10, 10, 1.05, 51)
energy = generate_grid(0, 10, 1.05, 175)
angular_mom = generate_grid(0, 1, 1.1, 100)

# fragment info
ch3_1 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
                                 [1.0788619988,0.0000000000,0.0000000000],
                                [-0.5394309994,-0.9343218982,0.0000000000],
                                [-0.5394309994,0.9343218982,0.0000000000]])

ch3_2 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
                                 [1.0788619988,0.0000000000,0.0000000000],
                                [-0.5394309994,-0.9343218982,0.0000000000],
                                [-0.5394309994,0.9343218982,0.0000000000]])

# Setting the dividing surfaces
# This creates an output file, as much as the given dividing surface.
# The output file contains temperature and the flux based on the temperature.
# 
divid_surf = [Surface({'0':np.array([[0.0,0.0,1.0],
                                      [0.0,0.0,-1.0]]),
                       '1':np.array([[0.0,0.0,1.0],
                                    [0.0,0.0,-1.0]])},
                        distances=np.array([[6.5,6.5],
                                            [6.5,6.5]])),]


# how to sample the two fragments
# calc = 'amp.amp'
calc = {
'code': 'ch3ch3_test.amp',
'method': 'caspt2(2,2)',
'basis': 'cc-pvtz',
'mem': 300,
'scratch': '/users/ykim219/',
'processors': 1,
'queue': 'slurm',
'max_jobs': 1400}


inf_energy = -79.478906090636
ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
                         inf_energy=inf_energy,
                         energy_size=1,
                         min_fragments_distance=0.1,
                         name='example')

selected_faces = [range(1) for i in range(len(divid_surf))]
faces_weights = [[4,0,0,0], [4,0,0,0], [4,0,0,0], [4,0,0,0], [4,0,0,0], [4,0,0,0], [4,0,0,0], [4,0,0,0], [1], [1], [1], [1], [1], [1], [1], [1]]

# the flux info per surface
#flux_rel_err: flux accuracy in 'nu' (1=90% certitude, 2=99%, ...)
#pot_smp_max: maximum number of sampling for each facet
#pot_smp_min: minimum number of sampling for each facet
#rtot_smp_max: maximum number of total sampling
#tot_smp_min: minimum number of total sampling

flux_parameter = {'pot_smp_max': 100, 'pot_smp_min': 10, #per facet
                  'tot_smp_max': 100, 'tot_smp_min': 10, 
                  'flux_rel_err': 20, 'smp_len': 1}


flux_base = FluxBase(temp_grid=temperature,
                     energy_grid=energy,
                     angular_grid=angular_mom,
                     flux_type='MICROCANONICAL',
                     flux_parameter=flux_parameter)

parallel = True
if parallel == True:
    multi = Multi(sample=ch3_sample, dividing_surfaces=divid_surf, fluxbase=flux_base, calculator=calc)
    multi.run()
else:
    multi = new_Multi(sample=ch3_sample, dividing_surfaces=divid_surf,
              fluxbase=flux_base, calculator=calc, selected_faces=selected_faces)
    multi.run()
    multi.print_results(dynamical_correction=1.0,
                    faces_weights=faces_weights)

# # start the final run
# multi = Multi(sample=ch3_sample, dividing_surfaces=divid_surf,
#               fluxbase=flux_base, calculator=calc, selected_faces=selected_faces) # , from_rslt=True
# multi.run()
# multi.print_results(dynamical_correction=1.0,
#                     faces_weights=faces_weights)
