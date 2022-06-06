#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
from math import ceil

class TestAnalyzeFunctions(unittest.TestCase):

    def test_img_flags_from_traj(self):
        
        # Define a random orthogonal simulation box configuration.

        rng = np.random.default_rng()
        lx, ly, lz = rng.random(size=3) * (100 - 1) + 1
        print(lx)
        print(ly)
        print(lz)
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
        
        # Define the positions, molecule IDs, and masses of particles.

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

    # def test_mol_com_from_traj(self):

    # def test_calc_rg(self):

    # def test_profile_density(self):


if __name__ == '__main__':
    unittest.main()
