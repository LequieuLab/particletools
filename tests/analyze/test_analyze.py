#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
from math import ceil
from numpy.random import default_rng

class TestAnalyzeFunctions(unittest.TestCase):

    def test_img_flags_from_traj(self):
        
        # Define an orthogonal simulation box configuration.

        test_box_config = np.asarray([4, 6, 10, 0, 0, 0])

        # Define a wrapped particle-based trajectory with known image flags.

        test_traj_wrap = [[[ -2,   3,   0], [  2,  -3,   3]],
                          [[  1,  -2,   3], [ -1,   2, 4.5]],
                          [[ -1,   1,  -2], [  2,  -2, 4.8]],
                          [[  2,  -3, 0.5], [  0,  -1,  -4]]]
        test_img_flags = [[[  0,   0,   0], [  0,   0,   0]],
                          [[ -1,   1,   0], [  1,  -1,   0]],
                          [[ -1,   1,   0], [  0,   0,   0]],
                          [[ -2,   2,   0], [  0,   0,   1]]]
        test_traj_wrap = np.asarray(test_traj_wrap)
        test_img_flags = np.asarray(test_img_flags, dtype=np.int32)

        # Test get_img_flags to see if it returns the correct values.

        img_flags = pt.img_flags_from_traj(test_traj_wrap, test_box_config)
        self.assertTrue((img_flags == test_img_flags).all())

    def test_unwrap_traj(self):
        
        # Define an orthogonal simulation box configuration.

        test_box_config = np.asarray([4, 6, 10, 0, 0, 0])

        # Define a wrapped particle-based trajectory with known images flags
        # and unwrapped particle-based trajectory.

        test_traj_wrap = [  [[ -2,   3,   0], [  2,  -3,   3]],
                            [[  1,  -2,   3], [ -1,   2, 4.5]],
                            [[ -1,   1,  -2], [  2,  -2, 4.8]],
                            [[  2,  -3, 0.5], [  0,  -1,  -4]]]          
        test_img_flags = [  [[  0,   0,   0], [  0,   0,   0]],
                            [[ -1,   1,   0], [  1,  -1,   0]],
                            [[ -1,   1,   0], [  0,   0,   0]],
                            [[ -2,   2,   0], [  0,   0,   1]]]
        test_traj_unwrap = [[[ -2,   3,   0], [  2,  -3,   3]],
                            [[ -3,   4,   3], [  3,  -4, 4.5]],
                            [[ -5,   7,  -2], [  2,  -2, 4.8]],
                            [[ -6,   9, 0.5], [  0,  -1,   6]]]          
        test_traj_wrap = np.asarray(test_traj_wrap)
        test_img_flags = np.asarray(test_img_flags, dtype=np.int32)
        test_traj_unwrap = np.asarray(test_traj_unwrap)

        # Test unwrap_traj to see if it returns the correct values.

        traj_unwrap = pt.unwrap_traj(test_traj_wrap, 
                                     test_box_config, 
                                     test_img_flags)
        self.assertTrue((traj_unwrap == test_traj_unwrap).all())

    def test_mol_com_from_frame(self):
        
        # Define the positions, molecule IDs, and masses of the particles.

        test_pos =     [[  -1,   -1,    0],
                        [ 2.5,  2.5,  2.5],
                        [   0,    0,    0],
                        [  -2,   -2,   -1],
                        [ 3.6,  3.6,  3.6],
                        [   5,    5,    5],
                        [   0,    0,    1],
                        [   1,    1,    0]]
        test_molid =   [    5,    2,    2,    5,    0,    2,    5,    5]
        test_mass =    [    1,   10,  3.5,    1,   69,  3.5,    1,    1]
        test_mol_com = [[ 3.6,  3.6,  3.6],
                        [ 2.5,  2.5,  2.5],
                        [-0.5, -0.5,    0]]
        test_pos = np.asarray(test_pos)
        test_molid = np.asarray(test_molid)
        test_mass = np.asarray(test_mass)
        test_mol_com = np.asarray(test_mol_com)

        # Test mol_com_from_frame to see if it returns the correct values.

        mol_com = pt.mol_com_from_frame(test_pos, test_molid, test_mass)
        self.assertTrue((mol_com == test_mol_com).all())

    def test_mol_com_from_traj(self):

        # Define the trajectory, molecule IDs, and masses of the particles.
     
        test_traj =         [[[  10,   -4,   -2],
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
        test_molid =          [   1,    1,    1,    1,    1,   2,    2]
        test_mass =           [   1,    1,    3,    1,    1,  10,   10]
        test_traj_mol_com = [[[   8,   -2,    0],
                              [   1,  0.5,    1]],
                             [[  10,   -1,    6],
                              [   5,   -4,  6.7]],
                             [[   4,    2,    0],
                              [   0,    6,    9]]]
        test_traj = np.asarray(test_traj)
        test_molid = np.asarray(test_molid)
        test_mass = np.asarray(test_mass)
        test_traj_mol_com = np.asarray(test_traj_mol_com)

        # Test mol_com_from_frame to see if it returns the correct values.

        traj_mol_com = pt.mol_com_from_traj(test_traj, test_molid, test_mass)
        self.assertTrue((traj_mol_com == test_traj_mol_com).all())

    # def test_calc_rg(self):

    # def test_profile_density(self):


if __name__ == '__main__':
    unittest.main()
