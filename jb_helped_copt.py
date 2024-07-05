from rotd_py.fragment.nonlinear import Nonlinear
from rotd_py.fragment.linear import Linear
from rotd_py.fragment.monoatomic import Monoatomic
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
# ch3_1 = Linear('CO', positions=[[-0.0, 0.0, 0.0],           
#                                  [1.0788619988,0.0000000000,0.0000000000]]) # position of each atom
#co_pos = [[-0.0, 0.0, 0.0],[1.0788619988,0.0000000000,0.0000000000]]

co_pos = [[7.581, 2.703, 20.777],[6.789, 1.659, 20.777]] # This position is on a few A above Slab

# co_pos = [[0.3269, 0.5639, 2.0777],[0.4347, 0.5639, 2.0777]] # This position is on a few A above Slab

ch3_1 = Linear('CO', positions= co_pos) # position of each atom

pos = [
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+01],
 [ 2.76655528e+00,  0.00000000e+00,  1.00000000e+01],
 [ 5.53311056e+00,  0.00000000e+00,  1.00000000e+01],
 [ 1.38327764e+00,  2.39590715e+00,  1.00000000e+01],
 [ 4.14983292e+00,  2.39590715e+00,  1.00000000e+01],
 [ 6.91638820e+00,  2.39590715e+00,  1.00000000e+01],
 [ 2.76655528e+00,  4.79181431e+00,  1.00000000e+01],
 [ 5.53311056e+00,  4.79181431e+00,  1.00000000e+01],
 [ 8.29966584e+00,  4.79181431e+00,  1.00000000e+01],
 [ 1.38327764e+00,  7.98635718e-01,  1.22588829e+01],
 [ 4.14983292e+00,  7.98635718e-01,  1.22588829e+01],
 [ 6.91638820e+00,  7.98635718e-01,  1.22588829e+01],
 [ 2.76655528e+00,  3.19454287e+00,  1.22588829e+01],
 [ 5.53311056e+00,  3.19454287e+00,  1.22588829e+01],
 [ 8.29966584e+00,  3.19454287e+00,  1.22588829e+01],
 [ 4.14983292e+00,  5.59045003e+00,  1.22588829e+01],
 [ 6.91638820e+00,  5.59045003e+00,  1.22588829e+01],
 [ 9.68294348e+00,  5.59045003e+00,  1.22588829e+01],
 [-5.11915562e-17,  1.59727144e+00,  1.45177659e+01],
 [ 2.76655528e+00,  1.59727144e+00,  1.45177659e+01],
 [ 5.53311056e+00,  1.59727144e+00,  1.45177659e+01],
 [ 1.38327764e+00,  3.99317859e+00,  1.45177659e+01],
 [ 4.14983292e+00,  3.99317859e+00,  1.45177659e+01],
 [ 6.91638820e+00,  3.99317859e+00,  1.45177659e+01],
 [ 2.76655528e+00,  6.38908575e+00,  1.45177659e+01],
 [ 5.53311056e+00,  6.38908575e+00,  1.45177659e+01],
 [ 8.29966584e+00,  6.38908575e+00,  1.45177659e+01],
 [ 0.00000000e+00,  0.00000000e+00,  1.67766488e+01],
 [ 2.76655528e+00,  0.00000000e+00,  1.67766488e+01],
 [ 5.53311056e+00,  0.00000000e+00,  1.67766488e+01],
 [ 1.38327764e+00,  2.39590715e+00,  1.67766488e+01],
 [ 4.14983292e+00,  2.39590715e+00,  1.67766488e+01],
 [ 6.91638820e+00,  2.39590715e+00,  1.67766488e+01],
 [ 2.76655528e+00,  4.79181431e+00,  1.67766488e+01],
 [ 5.53311056e+00,  4.79181431e+00,  1.67766488e+01],
 [ 8.29966584e+00,  4.79181431e+00,  1.67766488e+01], ####### added from here

# [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#  [ 8.3, 0, 0 ],
#  [ 4.15, 7.188, 0],
#  [ 12.45, 7.188, 0], # 이거 3개랑 하나는 0.0.0
 
#  [ 0,0,26.777 ],
#   [ 4.15, 7.188, 26.777],
#   [ 8.3, 0, 26.777],
#   [ 12.45, 7.188, 26.777],
]

ch3_2 = Slab(orig_mfo=ch3_1.frag_array['orig_mfo'], symbols='Pt36', positions = pos)

# For Surface
# divid_surf = [Surface({'0':np.array([[0.0, 0.0, 1.0], #  point i # 1,1,1 = numpy not iterable
#                                     [0.0, 0.0, -1.0]]), # Pivot point j
#                        '1':np.array([[2, 7.5, 0],
#                                     [2, 7.5, 0]])}, # No sampling found
#                         distances=np.array([[15.0,15.0],
#                                             [15.0,15.0]]))]

#co_pos = [[3.269, 5.639, 20.777],[4.347, 5.639, 20.777]] # This position is on a few A above Slab

# For test
divid_surf = [Surface({'0':np.array([[6.948, 2.139, 20.], #  point i
                                    [6.948, 2.139, 19.]]), # Pivot point j
                    #    '1':np.array([[3.8, 1.835, 17],
                    #                 [3.8, 1.835, 17]])}, # 이 피봇이냐 아래 피봇이냐에 따라 샘플이 나오고 안나오고
                    #    '1':np.array([[0, 0, 7 ],
                    #                 [0, 0, 7]])}, # 가까운게 lf에서 종종 보임
                        '1':np.array([[4.2, 2.4, 18 ],
                                    [4.2, 2.4, 18]])}, # 가까운게 lf에서 종종 보임
                        distances=np.array([[0.0,0.0], 
                                            [0.0,0.0]]))]


# For Gas
# divid_surf = [Surface({'0':np.array([[0.0,0.0,1.0],
#                                       [0.0,0.0,-1.0]]),
#                        '1':np.array([[0.0,0.0,1.0],
#                                     [0.0,0.0,-1.0]])},
#                         distances=np.array([[15,15],
#                                             [15,15]]))]



#how to sample the two fragments
calc = 'PtCO.amp'
# calc = 'ch3ch3_test.amp'

r_inf= -79.47971696 #RS2/cc-pvtz
ch3_sample = MultiSample(fragments=[ch3_1, ch3_2],
                inf_energy=r_inf, energy_size=1, min_fragments_distance=1.1) #originally 2.1

#the flux info
# flux_parameter={'pot_smp_max':2000, 'pot_smp_min': 100,
#            'tot_smp_max' : 10000, 'tot_smp_min' :50,
#            'flux_rel_err' : 10.0, 'smp_len' : 1}

# the flux info per surface
#flux_rel_err: flux accuracy in 'nu' (1=90% certitude, 2=99%, ...)
#pot_smp_max: maximum number of sampling for each facet
#pot_smp_min: minimum number of sampling for each facet
#tot_smp_max: maximum number of total sampling
#tot_smp_min: minimum number of total sampling

flux_parameter={'pot_smp_max':100, 'pot_smp_min': 50,
            'tot_smp_max' : 100, 'tot_smp_min' :50,
            'flux_rel_err' : 10.0, 'smp_len' : 1}


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