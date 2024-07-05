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

# ch3_2 = Nonlinear('CH3', positions=[[-0.0, .0, 0.0],
#                                  [1.0788619988,0.0000000000,0.0000000000],
#                                 [-0.5394309994,-0.9343218982,0.0000000000],
#                                 [-0.5394309994,0.9343218982,0.0000000000]])


#fragment info
#ch3_1 = Linear('CO', positions=[[-0.0, 0.0, 0.0],           
                                #  [1.0788619988,0.0000000000,0.0000000000]]) # position of each atom
# co_pos = [[-0.0, 0.0, 0.0],[1.0788619988,0.0000000000,0.0000000000]]

co_pos = [[3.3, 2.344, 9.777],[4.347, 2.344, 9.777]]


ch3_1 = Linear('CO', positions= co_pos) # position of each atom
#ch3_1 = Linear(orig_mfo=ch3_2.frag_array['orig_mfo'],symbols='CO' , positions= co_pos) # position of each atom


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

#  [ 8.3, 0, 0 ],
#  [ 4.15, 7.188, 0],
#  [ 12.45, 7.188, 0], # 이거 3개랑 하나는 0.0.0
 
 # [ 0,0,26.777 ],
  #[ 4.15, 7.188, 26.777],
  #[ 8.3, 0, 26.777],
  #[ 12.45, 7.188, 26.777],
]

#ch3_2 = Slab('Pt36', positions = pos)
ch3_2 = Slab('Pt36', positions = pos)
#ch3_2 = Slab(orig_mfo=ch3_1.frag_array['orig_mfo'], symbols='Pt36', positions = pos)

#unit_cell_334_pt = [[8.3, 0.000, 0.000],
#                    [4.15, 7.188, 0.000],
#                    [0.000, 0.000, 26.777]]

# For Surface
# divid_surf = [Surface({'0':np.array([[0.0, 0.0, 1.0], #  point i
#                                     [0.0, 0.0, -1.0]]), # Pivot point j
#                        '1':np.array([[2, 7.5, 0],
#                                     [2, 7.5, 0]])},
#                         distances=np.array([[15.0,15.0],
#                                             [15.0,15.0]]))]

divid_surf = [Surface({'0':np.array([[4.2, 2.344, 9], #  point i
                                    [4.2, 2.344, 9]]), # Pivot point j
                    #    '1':np.array([[3.8, 1.835, 17],
                    #                 [3.8, 1.835, 17]])}, # 이 피봇이냐 아래 피봇이냐에 따라 샘플이 나오고 안나오고
                    #    '1':np.array([[0, 0, 7 ],
                    #                 [0, 0, 7]])}, # 가까운게 lf에서 종종 보임
                        '1':np.array([[4.2, 2.4, 7 ],
                                    [4.2, 2.4, 7]])}, # 가까운게 lf에서 종종 보임
                        distances=np.array([[2.0,2.0], 
                                            [2.0,2.0]]))]

# For Gas
# divid_surf = [Surface({'0':np.array([[0.0,0.0,1.0],
#                                       [0.0,0.0,-1.0]]),
#                        '1':np.array([[0.0,0.0,1.0],
#                                     [0.0,0.0,-1.0]])},
#                         distances=np.array([[15,15],
#                                             [15,15]]))]



#how to sample the two fragments
calc = 'PtCO.amp'
#calc = 'ch3ch3_test.amp'

r_inf= -79.47971696 #RS2/cc-pvtz
ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
                inf_energy=r_inf, energy_size=1, min_fragments_distance=0.5)

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

flux_parameter={'pot_smp_max':4, 'pot_smp_min': 1,
            'tot_smp_max' : 4, 'tot_smp_min' :1,
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

#print(multi.total_flux['0'].flux_array[0].acct_smp())
#print(multi.total_flux['0'].flux_array[0].fail_smp())
#print(multi.total_flux['0'].flux_array[0].face_smp())
#print(multi.total_flux['0'].flux_array[0].close_smp())

#print(multi.total_flux['1'].flux_array[0].acct_smp())
#print(multi.total_flux['1'].flux_array[0].fail_smp())
#print(multi.total_flux['1'].flux_array[0].face_smp())
#print(multi.total_flux['1'].flux_array[0].close_smp())

#print ("example total flux", total_flux)