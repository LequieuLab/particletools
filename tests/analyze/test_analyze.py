#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
from math import sqrt
from numpy.random import default_rng

class TestAnalyzeFunctions(unittest.TestCase):

    def test_img_flags_from_traj(self):
        
        # Define an orthogonal simulation box configuration.

        box_config = np.asarray([4, 6, 10, 0, 0, 0])

        # Define a wrapped particle-based trajectory with known image flags.

        traj_wrap = [[[ -2,   3,   0], [  2,  -3,   3]],
                     [[  1,  -2,   3], [ -1,   2, 4.5]],
                     [[ -1,   1,  -2], [  2,  -2, 4.8]],
                     [[  2,  -3, 0.5], [  0,  -1,  -4]]]
        img_flags = [[[  0,   0,   0], [  0,   0,   0]],
                     [[ -1,   1,   0], [  1,  -1,   0]],
                     [[ -1,   1,   0], [  0,   0,   0]],
                     [[ -2,   2,   0], [  0,   0,   1]]]
        traj_wrap = np.asarray(traj_wrap)
        img_flags = np.asarray(img_flags, dtype=np.int32)

        # Test get_img_flags to see if it returns the correct values.

        test_img_flags = pt.img_flags_from_traj(traj_wrap, box_config)
        self.assertTrue((test_img_flags == img_flags).all())

    def test_unwrap_traj(self):
        
        # Define an orthogonal simulation box configuration.

        box_config = np.asarray([4, 6, 10, 0, 0, 0])

        # Define a wrapped particle-based trajectory with known images flags
        # and unwrapped particle-based trajectory.

        traj_wrap = [  [[ -2,   3,   0], [  2,  -3,   3]],
                       [[  1,  -2,   3], [ -1,   2, 4.5]],
                       [[ -1,   1,  -2], [  2,  -2, 4.8]],
                       [[  2,  -3, 0.5], [  0,  -1,  -4]]]          
        img_flags = [  [[  0,   0,   0], [  0,   0,   0]],
                       [[ -1,   1,   0], [  1,  -1,   0]],
                       [[ -1,   1,   0], [  0,   0,   0]],
                       [[ -2,   2,   0], [  0,   0,   1]]]
        traj_unwrap = [[[ -2,   3,   0], [  2,  -3,   3]],
                       [[ -3,   4,   3], [  3,  -4, 4.5]],
                       [[ -5,   7,  -2], [  2,  -2, 4.8]],
                       [[ -6,   9, 0.5], [  0,  -1,   6]]]          
        traj_wrap = np.asarray(traj_wrap)
        img_flags = np.asarray(img_flags, dtype=np.int32)
        traj_unwrap = np.asarray(traj_unwrap)

        # Test unwrap_traj to see if it returns the correct values.

        test_traj_unwrap = pt.unwrap_traj(traj_wrap, box_config, img_flags)
        self.assertTrue((test_traj_unwrap == traj_unwrap).all())

    def test_mol_com_from_frame(self):
        
        # Define the positions, molecule IDs, and masses of the particles.

        pos =     [[  -1,   -1,    0],
                   [ 2.5,  2.5,  2.5],
                   [   0,    0,    0],
                   [  -2,   -2,   -1],
                   [ 3.6,  3.6,  3.6],
                   [   5,    5,    5],
                   [   0,    0,    1],
                   [   1,    1,    0]]
        molid =    [    5,    2,    2,    5,    0,    2,    5,    5]
        mass =     [    1,   10,  3.5,    1,   69,  3.5,    1,    1]
        pos = np.asarray(pos)
        molid = np.asarray(molid)
        mass = np.asarray(mass)

        # Define the center of mass and mass for each molecule.

        mol_com = [[ 3.6,  3.6,  3.6],
                   [ 2.5,  2.5,  2.5],
                   [-0.5, -0.5,    0]]
        mol_mass = [  69,   17,    4]
        mol_com = np.asarray(mol_com)
        mol_mass = np.asarray(mol_mass)

        # Test mol_com_from_frame to see if it returns the correct values.

        test_mol_com, test_mol_mass = pt.mol_com_from_frame(pos, molid, mass)
        self.assertTrue((test_mol_com == mol_com).all())
        self.assertTrue((test_mol_mass == mol_mass).all())

    def test_mol_com_from_traj(self):

        # Define the trajectory, molecule IDs, and masses of the particles.
     
        traj =         [[[  10,   -4,   -2],
                         [   9,   -3,   -1],
                         [   8,   -2,    0],
                         [   7,   -1,    1],
                         [   6,    0,    2],
                         [   1,    0,    1],
                         [   1,    1,    1]],
                        [[  10,    0,    4],
                         [12.5,   -1,    5],
                         [   5,   -3,    6],
                         [  15,    1,    7],
                         [17.5,    2,    8],
                         [   1,  -10,  5.7],
                         [   9,    2,  7.7]],
                        [[   8,    2,    0],
                         [   6,    2,    0],
                         [   0,    2,    0],
                         [  10,    2,    0],
                         [   4,    2,    0],
                         [  -5,    7,  100],
                         [   5,    5,  -82]]]
        molid =          [   1,    1,    1,    1,    1,   2,    2]
        mass =           [   1,    1,    3,    1,    1,  10,   10]
        traj = np.asarray(traj)
        molid = np.asarray(molid)
        mass = np.asarray(mass)

        # Define the center of mass and mass for each molecule across the 
        # trajectory.

        traj_mol_com = [[[   8,   -2,    0],
                         [   1,  0.5,    1]],
                        [[  10,   -1,    6],
                         [   5,   -4,  6.7]],
                        [[   4,    2,    0],
                         [   0,    6,    9]]]
        mol_mass = [7, 20]
        traj_mol_com = np.asarray(traj_mol_com)
        mol_mass = np.asarray(mol_mass)

        # Test mol_com_from_traj to see if it returns the correct values.

        test_traj_mol_com, test_mol_mass = pt.mol_com_from_traj(traj, molid, 
                                                                mass)
        self.assertTrue((test_traj_mol_com == traj_mol_com).all())
        self.assertTrue((test_mol_mass == mol_mass).all())

    def test_rg_from_frame(self):
        
        # Define the positions, molecule IDs, and masses of the particles.

        pos =     [[ 4.2,  4.2,  3.2],
                   [   0,    0,    0],
                   [   8,    4,    1],
                   [  -1,    0,    1],
                   [ -11,    0,    6],
                   [   9,    0,   -4]]
        molid =    [   1,    0,    0,    2,    2,    2]
        mass =     [  75,    1,    1,   10,    1,    1]
        pos = np.asarray(pos)
        molid = np.asarray(molid)
        mass = np.asarray(mass)

        # Define the center of mass and radius of gyration for each molecule.

        mol_com = [[   4,    2,  0.5],
                   [ 4.2,  4.2,  3.2],
                   [  -1,    0,    1]]
        rg      =  [4.5, 0, sqrt(250/3)]
        mol_com = np.asarray(mol_com)
        rg = np.asarray(rg)

        # Test mol_com_from_traj to see if it returns the correct values.

        test_rg = pt.rg_from_frame(pos, molid, mass, mol_com)
        self.assertTrue((test_rg == rg).all())

    def test_rg_from_traj(self):
        
        # Define the trajectory, molecule IDs, and masses of the particles.

        traj =    [[[   5,    5,  3.2],
                    [   0,    0,    0],
                    [  10,   -4,    6],
                    [  -5,    0,    6],
                    [   0,    0,    6],
                    [   5,    0,    6]],
                   [[ 4.5,  4.5,    7],
                    [   6,   -2,    0],
                    [  10,   -4,   10],
                    [   0,   -5,    8],
                    [   0,    0,    8],
                    [   0,    5,    8]],
                   [[   8,    9,    1],
                    [  -1,   -1,   -1],
                    [   1,    1,    1],
                    [  -2,    0,    8],
                    [   0,    1,    8],
                    [   2,    2,    8]]]
        molid =     [   1,    0,    0,    4,    4,    4]
        mass =      [  75,    7,    7,    5,   50,    5]
        traj = np.asarray(traj)
        molid = np.asarray(molid)
        mass = np.asarray(mass)

        # Define the center of mass and radius of gyration for each molecule.

        traj_mol_com = [[[   5,   -2,    3],
                         [   5,    5,  3.2],
                         [   0,    0,    6]],
                        [[   8,   -3,    5],
                         [ 4.5,  4.5,    7],
                         [   0,    0,    8]],
                        [[   0,    0,    0],
                         [   8,    9,    1],
                         [   0,    1,    8]]]
        traj_rg = [[sqrt(38), 0, sqrt(50/3)],
                   [sqrt(30), 0, sqrt(50/3)],
                   [ sqrt(3), 0, sqrt(10/3)]]
        traj_mol_com = np.asarray(traj_mol_com)
        traj_rg = np.asarray(traj_rg)

        # Test mol_com_from_traj to see if it returns the correct values.

        test_traj_rg = pt.rg_from_traj(traj, molid, mass, traj_mol_com) 
        self.assertTrue((test_traj_rg == traj_rg).all())

    def test_profile_density(self):

        # Define the positions, molecule IDs, and masses of particles.
        # TODO Resume: fix this -3.6 from going into -5 bin

        pos =     [[-3.4,    0,    0],  # mol 3
                   [2.25,    0,    0],  # mol 9
                   [   0,    0,    0],  # mol 8
                   [   1,    1,    0],  # mol 5
                   [   1,    0,    1],  # mol 5
                   [   2,    2,    0],  # mol 6
                   [   2,   -2,    0],  # mol 6
                   [   3,    0,    0],  # mol 7
                   [   4,    0,    0],  # mol 7
                   [  -4,    1,   50],  # mol 1
                   [  -4,    0,   50],  # mol 2
                   [  -3,    0,    0]]  # mol 9
        molid = [3, 9, 8, 5, 5, 6, 6, 7, 7, 1, 2, 9]
        mass =  [1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        pos = np.asarray(pos)
        molid = np.asarray(molid)
        mass = np.asarray(mass)

        # Define an orthogonal box configuration, a selection array, the bin
        # dimension, the number of bins, and the cluster cutoff.

        box_config = [10, 1, 50, 0, 0, 0]
        selection = [0, 2, 3, 4, 5, 6, 7, 8, 11]
        bin_dim = 0
        nbins = 10
        ccut = 2
        box_config = np.asarray(box_config)
        selection = np.asarray(selection)

        # For the 'NONE' centering method, define the density profile and test
        # density_from_frame to see if it returns the correct values.

        density_profile = [[  -5,   0], 
                           [  -4,   1], 
                           [  -3,   1], 
                           [  -2,   0], 
                           [  -1,   0], 
                           [   0,   1], 
                           [   1,   2], 
                           [   2,   2], 
                           [   3,   1], 
                           [   4,   1]]
        density_profile = np.asarray(density_profile, dtype=np.float64)
        density_profile[:, 1] *= 1 / 50
        test_density_profile = pt.density_from_frame(pos, molid, mass, 
                                                     box_config, selection, 
                                                     bin_dim, nbins, 'NONE', 
                                                     ccut)
        self.assertTrue((test_density_profile == density_profile).all())

        # For the 'SYSTEM' centering method, define the density profile and
        # test density_from_frame to see if it returns the correct values.

        density_profile = [[  -5,   0], 
                           [  -4,   2], 
                           [  -3,   0], 
                           [  -2,   0], 
                           [  -1,   1], 
                           [   0,   2], 
                           [   1,   2], 
                           [   2,   1], 
                           [   3,   1], 
                           [   4,   0]]
        density_profile = np.asarray(density_profile, dtype=np.float64)
        density_profile[:, 1] *= 1 / 50
        test_density_profile = pt.density_from_frame(pos, molid, mass, 
                                                     box_config, selection, 
                                                     bin_dim, nbins, 'SYSTEM', 
                                                     ccut)
        self.assertTrue((test_density_profile == density_profile).all())

        # For the 'SLAB' centering method, define the density profile and test
        # density_from_frame to see if it returns the correct values.

        density_profile = [[  -5,   2], 
                           [  -4,   0], 
                           [  -3,   0], 
                           [  -2,   1], 
                           [  -1,   2], 
                           [   0,   2], 
                           [   1,   1], 
                           [   2,   1], 
                           [   3,   0], 
                           [   4,   0]]
        density_profile = np.asarray(density_profile, dtype=np.float64)
        density_profile[:, 1] *= 1 / 50
        test_density_profile = pt.density_from_frame(pos, molid, mass, 
                                                     box_config, selection, 
                                                     bin_dim, nbins, 'SLAB', 
                                                     ccut)
        self.assertTrue((test_density_profile == density_profile).all())

if __name__ == '__main__':
    unittest.main()
