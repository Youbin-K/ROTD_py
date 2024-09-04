from rotd_py.system import MolType
import rotd_py.rotd_math as rotd_math
import numpy as np
import ase
from abc import ABCMeta, abstractmethod
from ase.atoms import Atoms, Atom
from ase.io.trajectory import Trajectory
from amp import Amp
from ase.constraints import FixAtoms
import math


class Sample(object):
    """Class used for generating random configuration of given fragments and
    dividing surface.

    Parameters
    ----------
    fragments : List of Fragment
        The fragments in the system
    dividing_surface : Surface
        Current calculated surface considering all facets
    calculator: Calculator (ASE)
        Ase object calculator for calculating the potential
    min_fragments_distance : float
        Too close atoms distance threshold.
    Attributes
    ----------
    dividing_surface : Surface
    close_dist : float
    configuration : Atoms objects which will be used to store the combination of
                   two fragments (the sampled configuration) in Angstrom.
    dof_num : degree of freedom of the reaction system.
    weight : the geometry weight for the generated configuration.
    inf_energy: the energy of the configuration at infinite separation in Hartree
    """

    def __init__(self, fragments=None, dividing_surface=None,
                 min_fragments_distance=None, inf_energy=0.0, energy_size=1):
        __metaclass__ = ABCMeta
        # list of fragments

        self.surf_level = None

        self.fragments = fragments
        if fragments is None or len(fragments) < 2:
            raise ValueError("For sample class, \
                            at least two None fragments are needed")
        self.div_surface = dividing_surface
        self.close_dist = min_fragments_distance
        self.weight = 0.0
        self.inf_energy = inf_energy
        self.energy_size = energy_size
        self.energies = np.array([0.] * self.energy_size)
        self.ini_configuration()
        self.set_dof()

    def get_dividing_surface(self):
        return self.div_surface

    def set_dividing_surface(self, surface):
        self.div_surface = surface

    def set_dof(self):
        """Set up the degree of freedom of the system for calculation

        """
        self.dof_num = 3
        for frag in self.fragments:
            if frag.molecule_type == MolType.MONOATOMIC:
                continue
            elif frag.molecule_type == MolType.LINEAR:
                self.dof_num += 2
            elif frag.molecule_type == MolType.NONLINEAR:
                self.dof_num += 3
            elif frag.molecule_type == MolType.SLAB:
                # continue
                self.dof_num -= 3
                #print ('self dof num slab', self.dof_num)

    def ini_configuration(self):
        #Initialize the total system with ASE class Atoms
        # Random orientation comes after setting this function
        # ini_config -> generate_config
        
        new_atoms = []
        new_positions = []

        for frag in self.fragments:
            #print ("initial frag: ", frag) # CO, Pt one by one
                #frag.translate([0., 0., surf_level])
                #frag.translate([0.0, 0.0, 30])
                #print ("IF AND surf_level")
            new_atoms += frag.get_chemical_symbols()
            #print ("new_atoms: ", new_atoms)
            for pos in frag.get_positions():
                #print ("get_positions", frag.get_positions())
                #print ("position: ", pos) 
                #frag.translate([0.0, 0.0, 100])
                new_positions.append(pos)
                initial_coms = frag.get_center_of_mass()
                # print ('initial com', initial_coms)

        

        #print ("new_positions", new_positions)
        self.configuration = Atoms(new_atoms, new_positions)
        
        self.test_configuration = Atoms(new_atoms, new_positions)
        
        self.initial_user_configuration = Atoms(new_atoms, new_positions)
        self.surface_labframe_configuration = Atoms(new_atoms + ['B','N', 'F', 'He'], new_positions + [[0,0,0],[0,0,0], [0,0,0], [0,0,0]])
        self.visual_configuration = Atoms(new_atoms+ ['B','N', 'F'], new_positions + [[0,0,0],[0,0,0], [0,0,0]])
        self.gas_visual_configuration = Atoms(new_atoms + ['B','N', 'F', 'He'], new_positions + [[0,0,0],[0,0,0], [0,0,0], [0,0,0]])
        self.gas_labframe_configuration = Atoms(new_atoms + ['B','N','F', 'He'], new_positions + [[0,0,0],[0,0,0], [0,0,0], [0,0,0]])

        initial_user_position = new_positions.copy()
        self.initial_user_configuration.set_positions(initial_user_position)

        test_traj = Trajectory('initial_user_config.traj', 'w', self.initial_user_configuration)
        test_traj.write()
        test_traj.close()
        #print ("len config: ", len(self.configuration)) #### Total 38
        
        # if frag.molecule_type == MolType.SLAB:
        #     # c = FixAtoms(indices = [atom.index for atom in self.configuration if atom.symbol =='Pt'])
        #     # self.configuration.set_constraint(c)
        
        #     unit_cell_334_pt = [[8.3, 0.000, 0.000],
        #                         [4.15, 7.188, 0.000],
        #                         [0.000, 0.000, 26.777]]

        #     self.configuration.set_cell(unit_cell_334_pt)
            #self.labframe_configuration.set_cell(unit_cell_334_pt)
        

        """
        new_atoms = []
        new_positions = []
        for frag in self.fragments:
            new_atoms += frag.get_chemical_symbols()
            for pos in frag.get_positions():
                new_positions.append(pos)
        self.configuration = Atoms(new_atoms, new_positions)
        c = FixAtoms(indices = [atom.index for atom in self.configuration if atom.symbol =='Pt'])
        self.configuration.set_constraint(c) 
        """

    def get_dof(self):
        """return degree of freedom

        """
        return self.dof_num

    def get_canonical_factor(self):
        """return canonical factor used in the flux calculation

        """
        return 2.0 * np.sqrt(2.0 * np.pi)

    def get_microcanonical_factor(self): #느낌상 Gamma function 부터 ~ prod of sqrt.2pi.I 까지.
        """return microcanonical factor used in the flux calculation

        """
        #print ("Getting canonical factor")
        #print ("55555")
        # Called out after run in flux.py
        return rotd_math.M_2_SQRTPI * rotd_math.M_SQRT1_2 / 2.0 / \
            rotd_math.gamma_2(self.get_dof() + 1)

    def get_ej_factor(self):
        """return e-j resolved ensemble factor used in the flux calculation
            return 1/pi/Gamma(DOF/2 -1) DOF=nu??
        """
        #print ("Get ej factor")
        #print ("66666") # 여기서 끝남..
        # called out after get_microcanonical_factor
        return rotd_math.M_1_PI / rotd_math.gamma_2(self.get_dof() - 2)

    def get_tot_inertia_moments(self):
        """Return the inertia moments for the sampled configuration

        """
        
        # if (frag.molecule_type == MolType.SLAB for frag in self.fragments):
        #     #print ('passing slab tim')
        #     test = frag.get_chemical_symbols()
        #     print ('symbols ', test)
        #     pass
        # else:
        #     print ('I m not slab')
        #     return self.configuration.get_moments_of_inertia() * \
        #             rotd_math.mp / rotd_math.Bohr**2
        #print ('configuration: ', self.configuration) # COPt36 으로 나옴.
        #print ('symbols, ', self.configuration.get_chemical_symbols())
        #print('tim COM ',self.configuration.get_center_of_mass())
        #print ('tim positions ', self.configuration.get_positions())
        
        #여기여기여기여기 프린트 어케함?
        #print ('ASE tim get_inertia.get_positions: ', self.configuration.get_positions())
        #print ('inside tim')
        return self.configuration.get_moments_of_inertia() * \
                    rotd_math.mp / rotd_math.Bohr**2

    def get_weight(self):
        return self.weight

    def get_calculator(self):
        return self.calculator

    def if_fragments_too_close(self):
        """Check the distance among atoms between the two fragments.

        Returns
        -------
        type
            Boolean

        """
        lb_pos_0 = self.fragments[0].get_labframe_positions()
        #print ("Is this list? ", lb_pos_0) #This is lf position array of C and O
        #fragments[0] = CO & fragments[1] = Whole Pt
        #print ("len frag 0: ", len(lb_pos_0)) # 2 comes out
        lb_pos_1 = self.fragments[1].get_labframe_positions()
        #print ("len frag 1: ", len(lb_pos_1)) # 36 comes out
        #print ("self.fragment[1].get_lf_position", lb_pos_1)
        # This gets the position of whole 36 atoms of Pt
        for i in range(0, len(lb_pos_0)):
            for j in range(0, len(lb_pos_1)):
                dist = np.linalg.norm(lb_pos_0[i] - lb_pos_1[j])
                # print ('def if_fragments_too_close', dist)
                # print ('self.close_dist', self.close_dist)
                if dist < self.close_dist:
                    return True                    
        return False
    
    def if_fragments_too_close_for_surface(self):
        """Check the distance among atoms between the two fragments.

        Returns
        -------
        type
            Boolean

        """
        lb_pos_0 = self.fragments[0].get_labframe_positions()
        lb_pos_1 = self.fragments[1].get_labframe_positions()
        for i in range(0, len(lb_pos_0)):
            for j in range(0, len(lb_pos_1)):
                dist = np.linalg.norm(lb_pos_0[i] - lb_pos_1[j])
                # print ('def if_fragments_too_close', dist)
                # print ('self.close_dist', self.close_dist)
                if dist < 0.001:
                    return True                    
        return False

    """ 
    def if_slab_and_molecule_too_far(self):
        #lb_pos_co = self.fragments[0].get_labframe_positions()
        #lb_pos_platinum = self.fragments[1].get_labframe_positions()
        #Labframe 으로 따지면 복잡함. labframe = ASE.get_positions - center_of_mass
		#그러므로, absolute value 가져와서 그거로 com 잡아서 계산.
        # -9:-1 과 같음! : 는 한국의 ~와 같은 의미를 가짐. 즉, -9번째부터 끝까지
        # :2 는 0~2 까지 
        top_slab = Atoms(self.configuration.get_chemical_symbols()[-9:],
                         self.configuration.get_positions()[-9:]) 
        top_slab_com = top_slab.get_center_of_mass()
     
        new_co = Atoms(self.configuration.get_chemical_symbols()[:2],
                       self.configuration.get_positions()[:2]) 
        new_co_com = new_co.get_center_of_mass()
        #print (top_slab_com)

        #print ("Bohr 1st: ",lb_pos_platinum[34])
        #test5 = np.array(lb_pos_platinum)
        #test5 *= rotd_math.Bohr
        #print ("Units of ANG, platinum whole", test5)
        #print ("Ang 1st test5[0]: ", test5[34])
        #print(self.fragments[1].get_positions())

        #for i in range(0, len(lb_pos_co)):
            #dist = np.linalg.norm(lb_pos_co[i] - lb_pos_platinum[34])
            #print ("calculating the distance,,,,", dist)
            #if dist > 15:
                #print ("distance is longer than 10")
                #return True
        #return False
        dist = np.linalg.norm(top_slab_com - new_co_com)
        #print ("calculating the distance,,,,", dist)
        return dist > 4.5 #6.5 보다 멀리있는애는 버림
    
    
    def if_slab_and_molecule_too_close(self):
        top_slab = Atoms(self.configuration.get_chemical_symbols()[-9:], 
                         self.configuration.get_positions()[-9:])
        top_slab_com = top_slab.get_center_of_mass()

        new_co = Atoms(self.configuration.get_chemical_symbols()[:2],
                       self.configuration.get_positions()[:2])
        new_co_com = new_co.get_center_of_mass()

        dist = np.linalg.norm(top_slab_com - new_co_com)
        return dist < 2 # 2보다 가까운애는 버림
    """

    """
    def if_molecule_lower_than_slab(self):
        #print ("Does This Work?",new_positions[0][2]) new_position not defined
        top_layer = Atoms(self.configuration.get_chemical_symbols()[-1:],
                          self.configuration.get_positions()[-1:])
        top_layer_com = top_layer.get_center_of_mass()
        #print ("THIS IS TOP LAYEr COM", top_layer_com)
        #print ("TEST", top_layer_com[-1])

        input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2],
                               self.configuration.get_positions()[:2])
        input_molecule_com = input_molecule.get_center_of_mass()
        #print("COM COORD", input_molecule_com)
        #print ("-1-1", input_molecule_com[-1])
        z_position = input_molecule_com[-1] - top_layer_com[-1]
       
        return z_position < 2.5 # 2 보다 작은애 버림 
    """
    
    
    def check_molecule_z_coordinate(self):
        top_layer = Atoms(self.configuration.get_chemical_symbols()[-1:],
                          self.configuration.get_positions()[-1:])
        #print ('top layer', top_layer)
        top_layer_com = top_layer.get_center_of_mass()
        #print ('top layer COM ', top_layer_com)

        input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2],
                               self.configuration.get_positions()[:2])
        input_molecule_com = input_molecule.get_center_of_mass()
        #print ('input molecule ', input_molecule)
        #print ('input molecule COM ', input_molecule_com)

        z_position = input_molecule_com[-1] - top_layer_com[-1]

        if input_molecule_com[-1] < top_layer_com[-1] + 2.0:
            return True #버린다
        #print ('top layer COM ', top_layer_com)
        #print ('input molecule COM ', input_molecule_com)
        return False #남긴다

    def check_molecule_z_coordinate_in_bohr(self):
        top_layer = Atoms(self.configuration.get_chemical_symbols()[-1:],
                          self.configuration.get_positions()[-1:])
        #print ('top layer', top_layer)
        top_layer_com = top_layer.get_center_of_mass()
        #print ('top layer COM ', top_layer_com)

        input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2],
                               self.configuration.get_positions()[:2])
        input_molecule_com = input_molecule.get_center_of_mass()
        #print ('input molecule ', input_molecule)
        #print ('input molecule COM ', input_molecule_com)

        z_position = input_molecule_com[-1] - top_layer_com[-1]

        if input_molecule_com[-1] < top_layer_com[-1] + 2.0: # 1,2,3 & 37.xx works / 3 & 22 no works / 2 & 22 
            return True #버린다
        return False #남긴다
    



    def check_in_surface(self):

        def distance_between_two_points(point1, point2):
            # x1, y1, z1 = point1
            # x2, y2, z2 = point2
            # distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            distance = np.sqrt(np.sum((point2 - point1)**2))
            return distance
        
        right_bottom_molecule_position = (self.configuration.get_positions()[37]) # Pt 37
        right_top_molecule_position = (self.configuration.get_positions()[35]) # Pt 35
        left_bottom_molecule_position = (self.configuration.get_positions()[31]) # Pt 31
        left_top_molecule_position = (self.configuration.get_positions()[29]) # Pt 29
        
        length_of_bottom_x = distance_between_two_points(right_bottom_molecule_position, left_bottom_molecule_position)
        length_of_top_x = distance_between_two_points(right_top_molecule_position, left_top_molecule_position)
        length_of_left_y = distance_between_two_points(left_bottom_molecule_position, left_top_molecule_position)
        length_of_right_y = distance_between_two_points(right_bottom_molecule_position, right_top_molecule_position)

        input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2],
                               self.configuration.get_positions()[:2])
        input_molecule_com = input_molecule.get_center_of_mass()
        input_molecule_position = (self.configuration.get_positions()[0]) # Carbon

        square_ax1 = (right_top_molecule_position - right_bottom_molecule_position)/ distance_between_two_points(right_top_molecule_position, right_bottom_molecule_position)
        square_ax2 = (left_bottom_molecule_position -  right_bottom_molecule_position)/ distance_between_two_points(left_bottom_molecule_position,  right_bottom_molecule_position)
        square_ax3 = np.cross(square_ax1, square_ax2)
        test_point = input_molecule_com - right_bottom_molecule_position
        test_point_ax1 = np.dot(test_point, square_ax1)
        test_point_ax2 = np.dot(test_point, square_ax2)
        test_point_ax3 = np.dot(test_point, square_ax3)

        input_slab = Atoms(self.configuration.get_chemical_symbols()[2:38],
                               self.configuration.get_positions()[2:38])
        
        #print ('what is this?', input_slab)
        input_slab_com = input_slab.get_center_of_mass()
        #slab_com = self.fragments[1].get_center_of_mass() # This gives you initial position COM
        #print ('slab_com, ', slab_com) # [4.49565233 2.99488394 3.3883244 ]

        
        if (test_point_ax1 < length_of_right_y) and (test_point_ax1 > 2.5) and \
            (test_point_ax2 < length_of_bottom_x) and (test_point_ax2 > 2.5) and \
            (test_point_ax3 < 20) and (test_point_ax3 > 1.5): # 이게 아마 높이 조절하는값... 처음엔 2.5 도전은 1.0?
            #print ('is in square')
            # print ('right_bottom_molecule_position: ',right_bottom_molecule_position)
            # print ('Carbon position: ', input_molecule_position)
            #print ('slab aow COM', input_slab_com) # Correct COM comes out!!
            return False
            
        else: 
            return True
        
        
        """
        def is_inside_rectangle(point, rectangle):
            # 좌표를 2차원 평면 상의 좌표로 변환하여 정의된 사각형 내에 있는지 확인합니다.
            x, y = point[:2]  # x와 y 좌표만 추출합니다.
            x1, y1 = rectangle[0][:2]
            x2, y2 = rectangle[1][:2]
            x3, y3 = rectangle[2][:2]
            x4, y4 = rectangle[3][:2]
            
            # 사각형을 구성하는 각 변에 대한 방정식을 사용하여 사각형 내부에 있는지 확인합니다.
            if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1) and \
            (x1 <= x <= x4 or x4 <= x <= x1) and (y1 <= y <= y4 or y4 <= y <= y1):
                return True
            else:
                return False

        # 3차원 좌표를 정의합니다. 예를 들어, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] 형태로 입력합니다.
        coordinates_3d = [right_bottom_molecule_position_list, right_top_molecule_position_list,left_bottom_molecule_position_list, left_top_molecule_position_list ]
        # 특정 좌표를 정의합니다. 예를 들어, (0.5, 0.5) 형태로 입력합니다.
        point_to_check = input_molecule_com_list

        if is_inside_rectangle(point_to_check, coordinates_3d):
            #print("주어진 좌표의 x와 y 값은 사각형 안에 있습니다.")
            return False # 남김
        else:
            #print("주어진 좌표의 x와 y 값은 사각형 밖에 있습니다.")
            return True # 버림
        """

        """
        def is_inside_rectangle(point, rectangle):
            # 좌표를 좌표 평면의 방정식을 사용하여 정의된 사각형 내에 있는지 확인합니다.
            x, y, z = point
            x1, y1, z1 = rectangle[0]
            x2, y2, z2 = rectangle[1]
            x3, y3, z3 = rectangle[2]
            x4, y4, z4 = rectangle[3]
            
            # 사각형을 구성하는 각 면에 대한 법선 벡터를 계산합니다.
            normal1 = np.cross(np.array([x2-x1, y2-y1, z2-z1]), np.array([x3-x1, y3-y1, z3-z1]))
            normal2 = np.cross(np.array([x3-x2, y3-y2, z3-z2]), np.array([x4-x2, y4-y2, z4-z2]))
            normal3 = np.cross(np.array([x4-x3, y4-y3, z4-z3]), np.array([x1-x3, y1-y3, z1-z3]))
            normal4 = np.cross(np.array([x1-x4, y1-y4, z1-z4]), np.array([x2-x4, y2-y4, z2-z4]))
            
            # 각 면의 법선 벡터와 주어진 점과의 내적을 계산하여 사각형 내부에 있는지 확인합니다.
            if np.dot(normal1, np.array([x-x1, y-y1, z-z1])) >= 0 and \
            np.dot(normal2, np.array([x-x2, y-y2, z-z2])) >= 0 and \
            np.dot(normal3, np.array([x-x3, y-y3, z-z3])) >= 0 and \
            np.dot(normal4, np.array([x-x4, y-y4, z-z4])) >= 0:
                return True # 들어가있음
            else:
                return False # 안들어가 있음

        # 사각형의 네 꼭지점 좌표를 정의합니다. 예를 들어, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] 형태로 입력합니다.
        rectangle_coordinates = [right_bottom_molecule_position_list, right_top_molecule_position_list,left_bottom_molecule_position_list, left_top_molecule_position_list ]
        # 특정 좌표를 정의합니다. 예를 들어, (0.5, 0.5, 0) 형태로 입력합니다.
        point_to_check = input_molecule_com_list

        if is_inside_rectangle(point_to_check, rectangle_coordinates):
            #print("주어진 좌표는 사각형 안에 있습니다.")
            return True # 버림
        else:
            # print("주어진 좌표는 사각형 밖에 있습니다.")
            return False # 남김
        """



   
        




    def check_in_rhombus(self):
        #print ("Checking the rhombus")
        input_molecule = Atoms(self.configuration.get_chemical_symbols()[:2],
                               self.configuration.get_positions()[:2])
        input_molecule_com = input_molecule.get_center_of_mass()

        uc_x=8.358
        uc_y=7.238

        # For testing BOHR
        # uc_x = 15.794331
        # uc_y = 13.677838
        
        """
        ################### for COM + @  stay in unit cell ###################
        y_upper_boundary = uc_y
        y_lower_boundary = 0

        x_upper_boundary = uc_x + input_molecule_com[1] *(1.0/np.sqrt(3))
        x_lower_boundary = input_molecule_com[1]*(1.0/np.sqrt(3))
        ######################################################################

        """

        # ################### For whole molecule in unit cell ##################
        y_upper_boundary = uc_y + 5 # 3 is original
        y_lower_boundary = 1.0

        x_upper_boundary = uc_x + input_molecule_com[1] *(1.0/np.sqrt(3)) + 5 #3 is original
        x_lower_boundary = input_molecule_com[1]*(1.0/np.sqrt(3)) + 5
        # ######################################################################

        ################### For testing BOHR ##################
        # y_upper_boundary = uc_y + 5.66918
        # y_lower_boundary = 5.66918

        # x_upper_boundary = uc_x + input_molecule_com[1] *(1.0/np.sqrt(3)) + 5.66918
        # x_lower_boundary = input_molecule_com[1]*(1.0/np.sqrt(3)) + 5.66918
        ######################################################################
        

        if input_molecule_com[1] > y_upper_boundary or input_molecule_com[1] < y_lower_boundary:
            #print ("Y happening")
            return True
        if input_molecule_com[0] > x_upper_boundary or input_molecule_com[0] < x_lower_boundary:
            #print ("X happening")
            return True
        return False

    @abstractmethod
    def generate_configuration(self):
        """This function is an abstract method for generating random
        configuration based relative to different sample schema.
        1. generate the new configuration.
        2. set the self.weight
        3. set the new positions for self.configuration.

        """
        pass

    def get_energies(self, calculator):

        # return absolute energy in e
        # This is a temporary fix specific for using Amp calculator as we are not able to
        # either 1) deepcopy the Amp calculator object or 2) using MPI send/receive Amp calculator object
        # Note: this may cause some performance issue as we need to load Amp calculator for each calculation.
        amp_calc = Amp.load(calculator)
        self.configuration.set_calculator(amp_calc)
        e = self.configuration.get_potential_energy()
        self.configuration.set_calculator(None)
        # TODO: need to fill the energy correction part

        self.energies[0] = e / rotd_math.Hartree - self.inf_energy
        return self.energies


# The following code can be helpful if already has some sampled configuration
# and want to do some tests. Otherwise, no need to use the following code.
class Geometry(object):
    def __init__(self, atoms=None, energy=0.0, weight=0.0, pivot_rot=None,
                 frag1_rot=None, frag2_rot=None, tot_im=None):
        self.atoms = atoms
        self.energy = np.array(energy)
        self.weight = weight
        self.tot_im = np.array(tot_im)
        self.pivot_rot = np.array(pivot_rot)
        self.frag1_rot = np.array(frag1_rot)
        self.frag2_rot = np.array(frag2_rot)

    def get_atoms(self):
        return self.atoms

    def get_energy(self):
        return self.energy

    def get_weight(self):
        return self.weight

    def get_pivot_rot(self):
        return self.pivot_rot

    def get_frag1_rot(self):
        return self.frag1_rot

    def get_frag2_rot(self):
        return self.frag2_rot

    def get_tot_im(self):
        return self.tot_im


def preprocess(file_name):
    file_lines = open(file_name).readlines()
    geometries = []
    atom_num = 8
    for i, line in enumerate(file_lines):

        if line.startswith('Geometry:'):
            line_index = i
            positions = None
            if len(line.split()) == 1:
                positions = np.zeros((atom_num, 3))
                for j in range(0, atom_num):
                    positions[j] = map(float, file_lines[i+1+j].split()[1:])
                line_index += atom_num

            energy = float(file_lines[line_index + 1].split()[1])
            weight = float(file_lines[line_index + 2].split()[1])
            tot_im = np.array(map(float, file_lines[line_index + 4].split()[:]))
            pivot_rot = map(float, file_lines[line_index + 6].split()[:])
            frag1_rot = map(float, file_lines[line_index + 8].split())
            frag2_rot = map(float, file_lines[line_index + 10].split())
            atoms = None
            if positions is not None:
                atoms = Atoms('CH3CH3', positions=positions)

            geom = Geometry(atoms=atoms, energy=energy, weight=weight,
                            tot_im=tot_im,
                            frag1_rot=frag1_rot,
                            frag2_rot=frag2_rot, pivot_rot=pivot_rot)
            geometries.append(geom)

    return geometries
