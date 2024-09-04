from rotd_py.fragment.nonlinear import Nonlinear
from rotd_py.fragment.linear import Linear
from rotd_py.fragment.monoatomic import Monoatomic
#from rotd_py.fragment.updated_slab import Slab
from rotd_py.fragment.slab import Slab
from rotd_py.system import Surface
from rotd_py.multi import Multi
from rotd_py.sample.multi_sample import MultiSample
from rotd_py.flux.fluxbase import FluxBase
from ase.constraints import FixAtoms
from ase import Atoms, atoms
from ase.io import write
from ase.io.trajectory import Trajectory
import numpy as np
from amp import Amp

# Some things to note about the Slab version.
# 1. First fragment should always be the adsorbate
# 2. Initial position shold be set as how it actually looks like, 
#    unlike the gas version where everything is set based on (0,0,0)
# 3. Slab should always have the first atom on the (0,0,0) position so that visualization works.
# 4. For the visualization of the random rotation, this code uses Pt36 for base. 
#    If the structure or number of slab atoms change, 
#    go to fragment.slab and change the slab_straightening_matrix_calculation function.
# 5. For the Surface case, unlike the gas phase pivot point, the pivot points should stay on it's original coordinate.
#    However, for the gas phase, the pivot points are considered as molframe since the gas initial coordinate itself is the molframe.
# 6. Initial distance given is in Bohr units. So, for Ang, should be divided by 2


def generate_grid(start, interval, factor, num_point):
    #return the grid needed for simulation
    #interval += interval * factor
    #start += interval
    #return an numpy array with length of num_point
    i = 1
    grid = [start]
    for i in range(1, num_point):
        start += interval
        grid.append(start)
        interval = interval*factor
    return np.array(grid)

# temperature, energy grid and angular momentum grid
temperature = generate_grid(10, 10, 1.05, 51)
energy = generate_grid(0, 10, 1.05, 169)
angular_mom = generate_grid(0, 1, 1.1, 40)


# ch3_1 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
#                                  [1.0788619988,0.0000000000,0.0000000000],
#                                 [-0.5394309994,-0.9343218982,0.0000000000],
#                                 [-0.5394309994,0.9343218982,0.0000000000]])

# ch3_1 = Nonlinear('CH3', positions=[[1.0, 1.0, 1.0],
#                                  [2.0788619988, 1.0000000000, 1.0000000000],
#                                 [0.461, 0.166, 1.0000000000],
#                                 [0.461, 1.9343218982, 1.0000000000]])

# ch3_1 = Linear('CH', positions=[[-0.0, .0, 0.0],
#                                  [1.0788619988,0.0000000000,0.0000000000]])

# ch3_2 = Linear('CH', positions=[[-0.0, .0, 0.0],
#                                  [1.0788619988,0.0000000000,0.0000000000]])

# ch3_2 = Linear('CH', positions=[[2.0, 2.0, 2.0],
#                                  [3.0788619988, 2.0000000000, 2.0000000000]])


# ch3_2 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
#                                  [1.0788619988,0.0000000000,0.0000000000],
#                                 [-0.5394309994,-0.9343218982,0.0000000000],
#                                 [-0.5394309994,0.9343218982,0.0000000000]])

# ch3_2 = Nonlinear('CH3', positions=[[1.0, 1.0, 1.0],
#                                  [2.0788619988, 1.0000000000, 1.0000000000],
#                                 [0.461, 0.166, 1.0000000000],
#                                 [0.461, 1.9343218982, 1.0000000000]])


# ch3_2 = Nonlinear('CH3', positions=[[2.0, 2.0, 2.0],
#                                  [3.0788619988, 2.0000000000, 2.0000000000],
#                                 [1.461, 1.166, 2.0000000000],
#                                 [1.461, 2.9343218982, 2.0000000000]])


#fragment info
# ch3_1 = Linear('CO', positions=[[-0.0, 0.0, 0.0],           
#                                  [1.0788619988,0.0000000000,0.0000000000]]) # position of each atom
# co_pos = [[-0.0, 0.0, 0.0],[1.0788619988,0.0000000000,0.0000000000]]

co_pos = [[3.3, 2.344, 9.777],[4.347, 2.344, 9.777]]
ch3_1 = Linear('CO', positions= co_pos) # position of each atom

# #ch3_1 = Linear(orig_mfo=ch3_2.frag_array['orig_mfo'],symbols='CO' , positions= co_pos) # position of each atom


pos = [
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+01],
 #[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [ 2.76655528e+00,  0.00000000e+00,  0.00000000e+01],
 [ 5.53311056e+00,  0.00000000e+00,  0.00000000e+01],
 [ 1.38327764e+00,  2.39590715e+00,  0.00000000e+01],
 [ 4.14983292e+00,  2.39590715e+00,  0.00000000e+01],
 [ 6.91638820e+00,  2.39590715e+00,  0.00000000e+01],
 [ 2.76655528e+00,  4.79181431e+00,  0.00000000e+01],
 [ 5.53311056e+00,  4.79181431e+00,  0.00000000e+01],
 [ 8.29966584e+00,  4.79181431e+00,  0.00000000e+01],
 [ 1.38327764e+00,  7.98635718e-01,  0.22588829e+01],
 [ 4.14983292e+00,  7.98635718e-01,  0.22588829e+01],
 [ 6.91638820e+00,  7.98635718e-01,  0.22588829e+01],
 [ 2.76655528e+00,  3.19454287e+00,  0.22588829e+01],
 [ 5.53311056e+00,  3.19454287e+00,  0.22588829e+01],
 [ 8.29966584e+00,  3.19454287e+00,  0.22588829e+01],
 [ 4.14983292e+00,  5.59045003e+00,  0.22588829e+01],
 [ 6.91638820e+00,  5.59045003e+00,  0.22588829e+01],
 [ 9.68294348e+00,  5.59045003e+00,  0.22588829e+01],
 [-5.11915562e-17,  1.59727144e+00,  0.45177659e+01],
 [ 2.76655528e+00,  1.59727144e+00,  0.45177659e+01],
 [ 5.53311056e+00,  1.59727144e+00,  0.45177659e+01],
 [ 1.38327764e+00,  3.99317859e+00,  0.45177659e+01],
 [ 4.14983292e+00,  3.99317859e+00,  0.45177659e+01],
 [ 6.91638820e+00,  3.99317859e+00,  0.45177659e+01],
 [ 2.76655528e+00,  6.38908575e+00,  0.45177659e+01],
 [ 5.53311056e+00,  6.38908575e+00,  0.45177659e+01],
 [ 8.29966584e+00,  6.38908575e+00,  0.45177659e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.67766488e+01],
 [ 2.76655528e+00,  0.00000000e+00,  0.67766488e+01],
 [ 5.53311056e+00,  0.00000000e+00,  0.67766488e+01],
 [ 1.38327764e+00,  2.39590715e+00,  0.67766488e+01],
 [ 4.14983292e+00,  2.39590715e+00,  0.67766488e+01],
 [ 6.91638820e+00,  2.39590715e+00,  0.67766488e+01],
 [ 2.76655528e+00,  4.79181431e+00,  0.67766488e+01],
 [ 5.53311056e+00,  4.79181431e+00,  0.67766488e+01],
 [ 8.29966584e+00,  4.79181431e+00,  0.67766488e+01], ####### added from here
]

ch3_2 = Slab('Pt36', positions = pos)
#ch3_2 = Slab(orig_mfo=ch3_1.frag_array['orig_mfo'], symbols='Pt36', positions = pos)

#unit_cell_334_pt = [[8.3, 0.000, 0.000],
#                    [4.15, 7.188, 0.000],
#                    [0.000, 0.000, 26.777]]

# For Surface
# divid_surf = [Surface({'0':np.array([[0.61624056, 0., 0.], #  point i
#                                     [0.61624056, 0., 0.]]), # Pivot point j
# # divid_surf = [Surface({'0':np.array([[4.347, 2.344, 11.777], #  point i
# #                                     [4.347, 2.344, 11.777]]), # Pivot point j                                    
#                        '1':np.array([[4.150, 2.396, 6.777],
#                                     [4.150, 2.396, 6.777]])},
#                         distances=np.array([[7.0, 7.0],
#                                             [7.0, 7.0]]))]

# divid_surf = [Surface({'0':np.array([[3.89804115, 2.344, 10.777], #  point i # 이건 CO의 COM
#                                     [3.89804115, 2.344, 10.777]]), # Pivot point j
divid_surf = [Surface({'0':np.array([[3.3, 2.344, 9.777], #  point i # 이건 그냥 C
                                    [3.3, 2.344, 9.777]]), # Pivot point j        
                        '1':np.array([[4.150, 2.396, 6.777 ],
                                    [4.150, 2.396, 6.777]])}, # Pt surface middle atom
                        # '1':np.array([[2.638, 1.574, 8.523], 
                        #             [2.638, 1.574, 8.523]])}, # fcc-hollow site
#                         '1':np.array([[4.150, 2.396, 7.777 ],
#                                     [4.150, 2.396, 7.777]])}, # A top site
                        # distances=np.array([[7.0, 7.0], 
                        #                     [7.0, 7.0]]))] # 
                        distances=np.array([[7.0, 7.0], 
                                            [7.0, 7.0]]))] # Pivot 이 C 와 중간 Pt 로 주어질경우, 3.5 이하로 내려가면 안됨.
                                                            # 이 주어지는 거리의 경우 Bohr 단위로 주어짐. Ang = /2

# For Gas
# divid_surf = [Surface({'0':np.array([[0.0, 0.0, 1.0],
#                                       [0.0, 0.0, -1.0]]),
#                        '1':np.array([[0.0, 0.0, 1.0],
#                                     [0.0, 0.0, -1.0]])},
#                         distances=np.array([[0.5, 0.5],
#                                             [0.5, 0.5]]))]

# divid_surf = [Surface({'0':np.array([[0.0,0.0,0.0],
#                                       [0.0,0.0,0.0]]),
#                        '1':np.array([[0.0, 0.0, 0.0],
#                                     [0.0, 0.0, 0.0]])},
#                         distances=np.array([[7, 7],
#                                             [7, 7]]))] # 여기 distance 값은 bohr 임.

# divid_surf = [Surface({'0':np.array([[0.0,0.0,0.0],
#                                       [0.0,0.0,0.0]]),
#                        '1':np.array([[1.0, 1.0, 1.0],
#                                     [1.0, 1.0, 1.0]])},
#                         distances=np.array([[7, 7],
#                                             [7, 7]]))] # 여기 distance 값은 bohr 임.


#how to sample the two fragments
calc = 'PtCO.amp'
# calc = 'ch3ch3_test.amp'

r_inf= -79.47971696 #RS2/cc-pvtz
ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
                inf_energy=r_inf, energy_size=1, min_fragments_distance=0.01)

#the flux info
#flux_parameter={'pot_smp_max':2000, 'pot_smp_min': 100,
#            'tot_smp_max' : 10000, 'tot_smp_min' :50,
#            'flux_rel_err' : 10.0, 'smp_len' : 1}

# the flux info per surface
#flux_rel_err: flux accuracy in 'nu' (1=90% certitude, 2=99%, ...)
#pot_smp_max: maximum number of sampling for each facet
#pot_smp_min: minimum number of sampling for each facet
#tot_smp_max: maximum number of total sampling
#tot_smp_min: minimum number of total sampling

flux_parameter={'pot_smp_max':100, 'pot_smp_min': 10,
            'tot_smp_max' : 100, 'tot_smp_min' :10,
            'flux_rel_err' : 20.0, 'smp_len' : 1}


### X3 TQQQ
# flux_parameter={'pot_smp_max':6000, 'pot_smp_min': 300,
#            'tot_smp_max' : 30000, 'tot_smp_min' :150,
#            'flux_rel_err' : 30.0, 'smp_len' : 1}

flux_base = FluxBase(temp_grid = temperature,
                 energy_grid = energy,
                 angular_grid = angular_mom,
                flux_type = 'MICROCANONICAL',
                flux_parameter=flux_parameter)

#start the final run
multi = Multi(sample=ch3_sample, dividing_surfaces=divid_surf, fluxbase=flux_base, calculator=calc)
multi.run()
#multi.total_flux['0'].flux_array[0].run(50)
#multi.total_flux['0'].save_file(0)

# print ('end of copt.py')
# print('this is not printed why..?',multi.total_flux['0'].flux_array[0].acct_smp())
# print(multi.total_flux['0'].flux_array[0].fail_smp())
# print(multi.total_flux['0'].flux_array[0].face_smp())
# print(multi.total_flux['0'].flux_array[0].close_smp())

# print(multi.total_flux['1'].flux_array[0].acct_smp())
# print(multi.total_flux['1'].flux_array[0].fail_smp())
# print(multi.total_flux['1'].flux_array[0].face_smp())
# print(multi.total_flux['1'].flux_array[0].close_smp())

#print ("example total flux", total_flux)