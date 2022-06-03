#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
from math import ceil

class TestAnalyzeFunctions(unittest.TestCase):

    def test_get_img_flags(self):
        
        # Define an orthogonal simulation box configuration.

        box_config = np.asarray([4, 6, 10, 0, 0, 0])

        # Define a particle trajectory with known image flags.

        test_p_traj = [   [[ -2,   3,   0], [  2,  -3,   3]],
                          [[  1,  -2,   3], [ -2,   2, 4.5]],
                          [[ -1,   1,  -2], [  1,  -2, 4.8]],
                          [[  2,  -3, 0.5], [  0,  -1,  -4]]]          
        test_img_flags = [[[  0,   0,   0], [  0,   0,   0]],
                          [[ -1,   1,   0], [  1,  -1,   0]],
                          [[ -1,   1,   0], [  0,   0,   0]],
                          [[ -2,   2,   0], [  0,   0,   1]]]
        test_p_traj = np.asarray(test_p_traj)
        test_img_flags = np.asarray(test_img_flags, dtype=np.int32)

        # Test get_img_flags to see if it returns the correct values.

        img_flags = pt.get_img_flags(test_p_traj, box_config)
        self.assertTrue((img_flags == test_img_flags).all())

    # def test_get_mol_com(self):
        
    # def test_calc_rg(self):

    # def test_profile_density(self):


if __name__ == '__main__':
    unittest.main()
