import math
import numpy as np
from ase import units
from scipy.integrate import quad


"""
# This module include all the essential math calculation and
# the constant I am going to use in other calculations.
# The units used:
    ASE: angstrom, ev, g/mol
    rotd_py: bohr, Hartree, amu

"""

M_PI_2 = np.pi/2.0
M_1_PI = 1.0/np.pi
M_2_PI = 2.0/np.pi
M_2_SQRTPI = 2.0/np.sqrt(np.pi)
M_SQRT2 = np.sqrt(2.0)
M_SQRT1_2 = 1.0/np.sqrt(2.0)
M_SQRTPI = np.sqrt(np.pi)
# convert Kelv to Hartree

Kelv = units.kB/units.Hartree
# convert Kcal to Hartree
Kcal = units.kcal/units.mol/units.Hartree
# convert angstrom to Bohr
Bohr = units.Bohr
# convert eV to Hartree
Hartree = units.Hartree
# convert Hartree to cm-1
Hartree2cm = 219474.63

# mass of  proton in amu
mp = 1.672621898e-27/9.10938356e-31
# Adopt from original Varecof convert the number of states to rate constant with unit
# of 10^11 cm^3/sec

conv_fac = 612.6


# def any useful function for calculation under below:



def gamma_2(n): #Gamma function =Gamma(n/2) 
    if n == 1:
        return M_SQRTPI # = np.sqrt(np.pi)

    elif n == 2:
        return 1.0
    elif n > 2:
        return (n-2.0)/2.0 * gamma_2(n-2)


    return 0 # n>2 일땐 자기 혼자 돔 + return 0은 값이 음수거나, 0일때.



def mc_stat_weight(kin_en, ang_mom, iner_mom, dof_num):
    """integration for calculating the kinematic weight for E-J resolved case

    Parameters
    ----------
    kin_en : float
        kinetic energy.
    ang_mom : float
        Description of parameter `ang_mom`.
    iner_mom : 1*3 numpy array
        Description of parameter `iner_mom`.
    dof_num : int
        the degree of freedom of the current system

    Returns
    -------
    float
        the angular momentum integration.

    """

    eps = 1.0e-14
    tol = 1.0e-4
    if kin_en <= 0.:
        return 0

  #Sandia_molpro
  #  if iner_mom[0] < 0.:
  # Xi version 
    if iner_mom[0] <= 0.:

        raise ValueError("mc_stat_weight: inertia moments are not positive")
    if iner_mom[0] > iner_mom[1] or iner_mom[1] > iner_mom[2]:
        raise ValueError("mc_stat_weight: inertia moments are not monotonic")

    global_pow_num = dof_num - 4
    global_rot_en = np.zeros(3)
    for i in range(0, 3):
        global_rot_en[i] = ang_mom**2 / (2. * iner_mom[i] * kin_en)

    if global_rot_en[2] >= 1. - eps: # Why?
        return 0
    en_fac = np.power(kin_en, global_pow_num/2) #kin_en^ (nu/2-2) ## Here, kin_en = E-V_q

    if global_pow_num % 2: #global_pow_num = DOF-4

        en_fac *= np.sqrt(kin_en)

    def theta_integral(a, b, n):

        if a <= 0:
            return 0
        if b < eps*a:

            res = np.power(a, n/2)

            if n % 2:
                res *= np.sqrt(a)
            return res

        if n < -1:
            raise ValueError("The integral: wrong n \n")

        if n == -1:
            if b < a:
                return np.arcsin(np.sqrt(b/a))/np.sqrt(b)
            else:
                return M_PI_2/np.sqrt(b)

        if n == 0:
            if b <= a:
                return 1.0
            else:
                return np.sqrt(a/b)

        res = a * n * theta_integral(a, b, n-2)
        ab = a-b
        if ab > 0:
            dtemp = np.power(ab, n/2)
            if n % 2:
                dtemp *= np.sqrt(ab)
            res += dtemp

        return res/float(n+1)


    def phi_integrand(y): #mc_stat_weight 안에 들어있음

        y2 = y**2
        y1 = 1.0-y2
        r = np.zeros(2)
        r[0] = global_rot_en[1] * y1 + global_rot_en[2] * y2
        r[1] = global_rot_en[1] * y2 + global_rot_en[2] * y1

        res = 0

        for i in range(0, 2): #아마도 phi sqrt안에 있는 거 같은데.. 
            res += theta_integral(1.0 - r[i], global_rot_en[0]-r[i], global_pow_num)
        return res/np.sqrt(y1) # y 가 define이 안됐는데 어떻게 나오는거지??????????

    res = 0
    if global_rot_en[1] <= 1.:
        res, err = quad(phi_integrand, 0., M_SQRT1_2) # Quad 는 phi_integrand를 0 에서 M_sqrt1_2까지 적분. M_sqrt1_2 = 1/루트 2. phi_integrand = 
    else:
        dtemp = (1. - global_rot_en[2]) / (global_rot_en[1] - global_rot_en[2]) #dtemp = 꺽쇠 오메가 J 적분 범위

        if dtemp <= 0.5:
            y_max = np.sqrt(dtemp)
            res, err = quad(phi_integrand, 0., y_max)
        else:
            y_max = np.sqrt(1.0 - dtemp)
            res, err = quad(phi_integrand, 0., y_max)
            res_2, err = quad(phi_integrand, y_max, M_SQRT1_2)
            res += res_2

    return M_2_PI * en_fac * res


def microcanonical_phi():
    return 


def random_orient(dimes):
    
    #This function is used to generate a unit vector in dimension of dimes

    
    vec = [np.random.normal() for i in range(dimes)]
    vec = np.array(vec)
    #print ("dimes", dimes) # Depends on the angle_size of fragment
    #print ("rotd_math.py rnd orient vec", vec) # 3 or 4 array followed by dimes
    magni = sum(x**2 for x in vec) ** 0.5
    test = vec/magni
    #print ("rotd_math.py rnd orient vec/magni", test) # 3 or 4 array followed by dimes
    return vec/magni

###########################################################################
# Due to the circlular import, this is not possible

"""
def random_orient(dimes):
    if frag.molecule_type == MolType.SLAB:
        c = FixAtoms(indices = [atom.index for atom in frag if atom.symbol =='Pt'])


        vec = [np.random.normal() for i in range(dimes)]
        vec = np.array(vec)
        #print ("dimes", dimes) # 3 or 4
        #print ("rotd_math.py rnd orient vec", vec) # 3 or 4 array followed by dimes
        magni = sum(x**2 for x in vec) ** 0.5
        test = vec/magni
        #print ("rotd_math.py rnd orient vec/magni", test) # 3 or 4 array followed by dimes
    else:
        vec = [np.random.normal() for i in range(dimes)]
        vec = np.array(vec)
        magni = sum(x**2 for x in vec) ** 0.5
        test = vec/magni
    return vec/magni
"""



def orthogonalize(vector, n):
    """Return a vector that orthogonal to n and in the same plane with vector
    CAUTION: make the change to vector directly
    """
    norm = sum(n**2)
    norm1 = sum(vector**2)
    project = np.dot(vector, n)
    vector -= project * n
    return project**2 * norm / norm1


def normalize(vector):
    """Normalize the vector, change to itself directly
    Return the magnitude
    """
    norm = np.sqrt(sum(vector**2))
    if norm == 0:

        return norm
    else:
        vector /= norm

    return norm


def quaternion_matrix_transformation(quat):
    """TODO: Return homogeneous rotation matrix from  quaternion
    Input: a normalized quaternion
    return 2-D numpy array
    """
    if abs(1.0-sum(quat**2)) > 1.0e-5:
        raise ValueError("Input is not normalized")
    a, b, c, d = quat[0], quat[1], quat[2], quat[3]
    return np.array([[a**2+b**2-c**2-d**2, 2*b*c+2*a*d, 2*b*d-2*a*c, ],
                     [2*b*c-2*a*d, a**2-b**2 + c**2-d**2, 2*c*d+2*a*b],
                     [2*b*d+2*a*c, 2*c*d-2*a*b, a**2-b**2-c**2+d**2]])
