#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
from math import sqrt
from numpy.random import default_rng

# TODO Add unit tests for calc_rdf and calc_S. 

class TestAnalyzeFunctions(unittest.TestCase):

    # def test_img_flags_from_traj(self):
    #     
    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([4, 6, 10, 0, 0, 0])

    #     # Define a wrapped particle-based trajectory with known image flags.

    #     traj_wrap = [[[ -2,   3,   0], [  2,  -3,   3]],
    #                  [[  1,  -2,   3], [ -1,   2, 4.5]],
    #                  [[ -1,   1,  -2], [  2,  -2, 4.8]],
    #                  [[  2,  -3, 0.5], [  0,  -1,  -4]]]
    #     img_flags = [[[  0,   0,   0], [  0,   0,   0]],
    #                  [[ -1,   1,   0], [  1,  -1,   0]],
    #                  [[  0,   0,   1], [  0,   0,   0]],
    #                  [[ -1,   1,   1], [  1,   0,   1]]]
    #     traj_wrap = np.asarray(traj_wrap)
    #     img_flags = np.asarray(img_flags, dtype=np.int32)

    #     # Test get_img_flags to see if it returns the expected values.

    #     test_img_flags = pt.img_flags_from_traj(traj_wrap, box_config)
    #     self.assertTrue((test_img_flags == img_flags).all())

    # def test_unwrap_traj(self):
    #     
    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([4, 6, 10, 0, 0, 0])

    #     # Define a wrapped particle-based trajectory with known images flags
    #     # and unwrapped particle-based trajectory.

    #     traj_wrap = [  [[ -2,   3,   0], [  2,  -3,   3]],
    #                    [[  1,  -2,   3], [ -1,   2, 4.5]],
    #                    [[ -1,   1,  -2], [  2,  -2, 4.8]],
    #                    [[  2,  -3, 0.5], [  0,  -1,  -4]]]          
    #     img_flags = [  [[  0,   0,   0], [  0,   0,   0]],
    #                    [[ -1,   1,   0], [  1,  -1,   0]],
    #                    [[ -1,   1,   0], [  0,   0,   0]],
    #                    [[ -2,   2,   0], [  0,   0,   1]]]
    #     traj_unwrap = [[[ -2,   3,   0], [  2,  -3,   3]],
    #                    [[ -3,   4,   3], [  3,  -4, 4.5]],
    #                    [[ -5,   7,  -2], [  2,  -2, 4.8]],
    #                    [[ -6,   9, 0.5], [  0,  -1,   6]]]          
    #     traj_wrap = np.asarray(traj_wrap)
    #     img_flags = np.asarray(img_flags, dtype=np.int32)
    #     traj_unwrap = np.asarray(traj_unwrap)

    #     # Test unwrap_traj to see if it returns the expected values.

    #     test_traj_unwrap = pt.unwrap_traj(traj_wrap, box_config, img_flags)
    #     self.assertTrue((test_traj_unwrap == traj_unwrap).all())

    # def test_mol_com_from_frame(self):
    #     
    #     # Define the positions, molecule IDs, and masses of the particles.

    #     pos =     [[  -1,   -1,    0],
    #                [ 2.5,  2.5,  2.5],
    #                [   0,    0,    0],
    #                [  -2,   -2,   -1],
    #                [ 3.6,  3.6,  3.6],
    #                [   5,    5,    5],
    #                [   0,    0,    1],
    #                [   1,    1,    0]]
    #     molid =    [    5,    2,    2,    5,    0,    2,    5,    5]
    #     mass =     [    1,   10,  3.5,    1,   69,  3.5,    1,    1]
    #     pos = np.asarray(pos)
    #     molid = np.asarray(molid)
    #     mass = np.asarray(mass)

    #     # Define the center of mass and mass for each molecule.

    #     mol_com = [[ 3.6,  3.6,  3.6],
    #                [ 2.5,  2.5,  2.5],
    #                [-0.5, -0.5,    0]]
    #     mol_mass = [  69,   17,    4]
    #     mol_com = np.asarray(mol_com)
    #     mol_mass = np.asarray(mol_mass)

    #     # Test mol_com_from_frame to see if it returns the expected values.

    #     test_mol_com, test_mol_mass = pt.mol_com_from_frame(pos, molid, mass)
    #     self.assertTrue((test_mol_com == mol_com).all())
    #     self.assertTrue((test_mol_mass == mol_mass).all())

    # def test_mol_com_from_traj(self):

    #     # Define the trajectory, molecule IDs, and masses of the particles.
    #  
    #     traj =         [[[  10,   -4,   -2],
    #                      [   9,   -3,   -1],
    #                      [   8,   -2,    0],
    #                      [   7,   -1,    1],
    #                      [   6,    0,    2],
    #                      [   1,    0,    1],
    #                      [   1,    1,    1]],
    #                     [[  10,    0,    4],
    #                      [12.5,   -1,    5],
    #                      [   5,   -3,    6],
    #                      [  15,    1,    7],
    #                      [17.5,    2,    8],
    #                      [   1,  -10,  5.7],
    #                      [   9,    2,  7.7]],
    #                     [[   8,    2,    0],
    #                      [   6,    2,    0],
    #                      [   0,    2,    0],
    #                      [  10,    2,    0],
    #                      [   4,    2,    0],
    #                      [  -5,    7,  100],
    #                      [   5,    5,  -82]]]
    #     molid =          [   1,    1,    1,    1,    1,   2,    2]
    #     mass =           [   1,    1,    3,    1,    1,  10,   10]
    #     traj = np.asarray(traj)
    #     molid = np.asarray(molid)
    #     mass = np.asarray(mass)

    #     # Define the center of mass and mass for each molecule across the 
    #     # trajectory.

    #     traj_mol_com = [[[   8,   -2,    0],
    #                      [   1,  0.5,    1]],
    #                     [[  10,   -1,    6],
    #                      [   5,   -4,  6.7]],
    #                     [[   4,    2,    0],
    #                      [   0,    6,    9]]]
    #     mol_mass = [7, 20]
    #     traj_mol_com = np.asarray(traj_mol_com)
    #     mol_mass = np.asarray(mol_mass)

    #     # Test mol_com_from_traj to see if it returns the expected values.

    #     test_traj_mol_com, test_mol_mass = pt.mol_com_from_traj(traj, molid, 
    #                                                             mass)
    #     self.assertTrue((test_traj_mol_com == traj_mol_com).all())
    #     self.assertTrue((test_mol_mass == mol_mass).all())

    # def test_calc_rg(self):

    #     # Define the positions and masses of the molecule's particles.

    #     pos =     [[  -5,    1,    3],  
    #                [   0,    1,    2],  
    #                [   5,    0,   -9]]  
    #     pos = np.asarray(pos)
    #     mass = [1, 7, 2]
    #     mass = np.asarray(mass)

    #     # Test calc_rg to see if it returns the expected value.

    #     test_calc_rg = pt.calc_rg(pos, mass)
    #     calc_rg = sqrt(27.3)
    #     self.assertTrue((test_calc_rg == calc_rg))

    # def test_density_from_frame(self):

    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([10, 1, 50, 0, 0, 0])

    #     # Define the positions, molecule IDs, and masses of the particles.

    #     pos =     [[-3.4,    0,    0],  
    #                [2.25,    0,    0],  
    #                [   0,    0,    0],  
    #                [   1,    1,    0],  
    #                [   1,    0,    1],  
    #                [   2,    2,    0],  
    #                [   2,   -2,    0],  
    #                [   3,    0,    0],  
    #                [   4,    0,    0],  
    #                [  -4,    1,   50],  
    #                [  -4,    0,   50],  
    #                [  -3,    0,    0]]  
    #     molid = [3, 9, 8, 5, 5, 6, 6, 7, 7, 1, 2, 9]
    #     mass =  [1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #     pos = np.asarray(pos)
    #     molid = np.asarray(molid)
    #     mass = np.asarray(mass)

    #     # Define a selection array, the bin axis, the number of bins, and the 
    #     # cluster cutoff.

    #     box_config = [10, 1, 50, 0, 0, 0]
    #     selection = [0, 2, 3, 4, 5, 6, 7, 8, 11]
    #     bin_axis = 0
    #     nbins = 10
    #     ccut = 2
    #     box_config = np.asarray(box_config)
    #     selection = np.asarray(selection)

    #     # For the 'NONE' centering method, define the density profile and test
    #     # density_from_frame to see if it returns the expected values.

    #     density_profile = [[  -5,   0], 
    #                        [  -4,   1], 
    #                        [  -3,   1], 
    #                        [  -2,   0], 
    #                        [  -1,   0], 
    #                        [   0,   1], 
    #                        [   1,   2], 
    #                        [   2,   2], 
    #                        [   3,   1], 
    #                        [   4,   1]]
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= 50
    #     test_density_profile = pt.density_from_frame(pos, molid, mass, 
    #                                                  box_config, selection, 
    #                                                  bin_axis, nbins, 'NONE', 
    #                                                  ccut)
    #     self.assertTrue((test_density_profile == density_profile).all())

    #     # For the 'SYSTEM' centering method, define the density profile and
    #     # test density_from_frame to see if it returns the expected values.

    #     density_profile = [[  -5,   0], 
    #                        [  -4,   2], 
    #                        [  -3,   0], 
    #                        [  -2,   0], 
    #                        [  -1,   1], 
    #                        [   0,   2], 
    #                        [   1,   2], 
    #                        [   2,   1], 
    #                        [   3,   1], 
    #                        [   4,   0]]
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= 50
    #     test_density_profile = pt.density_from_frame(pos, molid, mass, 
    #                                                  box_config, selection, 
    #                                                  bin_axis, nbins, 'SYSTEM', 
    #                                                  ccut)
    #     self.assertTrue((test_density_profile == density_profile).all())

    #     # For the 'SLAB' centering method, define the density profile and test
    #     # density_from_frame to see if it returns the expected values.

    #     density_profile = [[  -5,   2], 
    #                        [  -4,   0], 
    #                        [  -3,   0], 
    #                        [  -2,   1], 
    #                        [  -1,   2], 
    #                        [   0,   2], 
    #                        [   1,   1], 
    #                        [   2,   1], 
    #                        [   3,   0], 
    #                        [   4,   0]]
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= 50
    #     test_density_profile = pt.density_from_frame(pos, molid, mass, 
    #                                                  box_config, selection, 
    #                                                  bin_axis, nbins, 'SLAB', 
    #                                                  ccut)
    #     self.assertTrue((test_density_profile == density_profile).all())

    # def test_density_from_traj(self):

    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([3, 20, 50, 0, 0, 0])

    #     # Define the trajectory, molecule IDs, and masses of the particles.

    #     traj =    [[[   0,   -5,   40],  
    #                 [   0,    5,    0],  
    #                 [   0,    7,    0],  
    #                 [   0,    7,    2],  
    #                 [   0,    9,    0]],

    #                [[   0,   -8,   30],  
    #                 [   0,    3,    0],  
    #                 [   0,    5,    0],  
    #                 [   0,    9,    0],  
    #                 [   0,    7,    0]]]
    #     molid = [ 5,  1,  1,  3,  3] 
    #     mass =  [96,  1,  1,  1,  1] 
    #     traj = np.asarray(traj)
    #     molid = np.asarray(molid)
    #     mass = np.asarray(mass)

    #     # Define a selection array, the bin axis, the number of bins, and the 
    #     # cluster cutoff.

    #     selection = [1, 2, 3]
    #     bin_axis = 1
    #     nbins = 10
    #     ccut = 10
    #     selection = np.asarray(selection)

    #     # For the 'NONE' centering method, define the density profile and test
    #     # density_from_traj to see if it returns the expected values. 

    #     density_profile = [[ -10,   0], 
    #                        [  -8,   0], 
    #                        [  -6,   0], 
    #                        [  -4,   0], 
    #                        [  -2,   0],  
    #                        [   0,   0], 
    #                        [   2, 0.5], 
    #                        [   4,   1], 
    #                        [   6,   1], 
    #                        [   8, 0.5]] 
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= (3 * 50 * 20 / 10)
    #     test_traj_density_profile = pt.density_from_traj(traj, molid, mass,
    #                                                      box_config, selection,
    #                                                      bin_axis, nbins, 
    #                                                      'NONE', ccut)
    #     test_density_profile = np.average(test_traj_density_profile, axis=0)
    #     self.assertTrue((test_density_profile == density_profile).all())

    #     # For the 'SYSTEM' centering method, define the density profile and
    #     # test density_from_traj to see if it returns the expected values.

    #     density_profile = [[ -10, 1.5], 
    #                        [  -8, 0.5], 
    #                        [  -6,   0], 
    #                        [  -4, 0.5], 
    #                        [  -2,   0], 
    #                        [   0,   0], 
    #                        [   2,   0], 
    #                        [   4,   0], 
    #                        [   6,   0], 
    #                        [   8, 0.5]] 
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= (3 * 50 * 20 / 10)
    #     test_traj_density_profile = pt.density_from_traj(traj, molid, mass,
    #                                                      box_config, selection,
    #                                                      bin_axis, nbins, 
    #                                                      'SYSTEM', ccut)
    #     test_density_profile = np.average(test_traj_density_profile, axis=0)
    #     self.assertTrue((test_density_profile == density_profile).all())

    #     # For the 'SLAB' centering method, define the density profile and test
    #     # density_from_traj to see if it returns the expected values.

    #     density_profile = [[ -10,   0],  
    #                        [  -8,   0], 
    #                        [  -6,   0], 
    #                        [  -4, 0.5], 
    #                        [  -2,   1], 
    #                        [   0,   1], 
    #                        [   2, 0.5], 
    #                        [   4,   0], 
    #                        [   6,   0], 
    #                        [   8,   0]] 
    #     density_profile = np.asarray(density_profile, dtype=np.float64)
    #     density_profile[:, 1] /= (3 * 50 * 20 / 10)
    #     test_traj_density_profile = pt.density_from_traj(traj, molid, mass,
    #                                                      box_config, selection,
    #                                                      bin_axis, nbins, 
    #                                                      'SLAB', ccut)
    #     test_density_profile = np.average(test_traj_density_profile, axis=0)
    #     self.assertTrue((test_density_profile == density_profile).all())

    # def test_meshgrid3D(self):

    #     # Define the x-axis, y-axis, and z-axis values.

    #     x = np.linspace( 0, 1, 2)
    #     y = np.linspace(-1, 1, 3)
    #     z = np.linspace( 0, 3, 4)

    #     # Define the gridpoint values of each axis in the 3D mesh.

    #     xv = [[[ 0,  0,  0,  0],
    #            [ 0,  0,  0,  0],
    #            [ 0,  0,  0,  0]],
    #           [[ 1,  1,  1,  1],
    #            [ 1,  1,  1,  1],
    #            [ 1,  1,  1,  1]]]
    #     yv = [[[-1, -1, -1, -1],
    #            [ 0,  0,  0,  0],
    #            [ 1,  1,  1,  1]],
    #           [[-1, -1, -1, -1],
    #            [ 0,  0,  0,  0],
    #            [ 1,  1,  1,  1]]]
    #     zv = [[[ 0,  1,  2,  3],
    #            [ 0,  1,  2,  3],
    #            [ 0,  1,  2,  3]],
    #           [[ 0,  1,  2,  3],
    #            [ 0,  1,  2,  3],
    #            [ 0,  1,  2,  3]]]
    #     grid = np.stack((xv, yv, zv), axis=0)

    #     # Test meshgrid3D to see if it returns the expected values.

    #     test_grid = pt.meshgrid3D(x, y, z)
    #     self.assertTrue((test_grid == grid).all())

    # def test_rijcnt_from_frame(self):

    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([8, 6, 8, 0, 0, 0])

    #     # Define the positions of particles.

    #     pos =  [[    3,    -2,     3],
    #             [    3,     2,     1],
    #             [    2,    -1,     0],
    #             [   -3,     2,  0.01]]
    #     pos = np.asarray(pos)
    #             
    #     # Define the rij vector cutoff and the number of grind points per axis.

    #     rcut = 400
    #     ngpoints = np.asarray([5, 7, 3])

    #     # Define the rij grid and the average number of particles at rij.

    #     xv = [[[ -4, -4, -4],
    #            [ -4, -4, -4],
    #            [ -4, -4, -4],
    #            [ -4, -4, -4],
    #            [ -4, -4, -4],
    #            [ -4, -4, -4],
    #            [ -4, -4, -4]],
    #           [[ -2, -2, -2],
    #            [ -2, -2, -2],
    #            [ -2, -2, -2],
    #            [ -2, -2, -2],
    #            [ -2, -2, -2],
    #            [ -2, -2, -2],
    #            [ -2, -2, -2]],
    #           [[  0,  0,  0],
    #            [  0,  0,  0],
    #            [  0,  0,  0],
    #            [  0,  0,  0],
    #            [  0,  0,  0],
    #            [  0,  0,  0],
    #            [  0,  0,  0]],
    #           [[  2,  2,  2],
    #            [  2,  2,  2],
    #            [  2,  2,  2],
    #            [  2,  2,  2],
    #            [  2,  2,  2],
    #            [  2,  2,  2],
    #            [  2,  2,  2]],
    #           [[  4,  4,  4],
    #            [  4,  4,  4],
    #            [  4,  4,  4],
    #            [  4,  4,  4],
    #            [  4,  4,  4],
    #            [  4,  4,  4],
    #            [  4,  4,  4]]]
    #     yv = [[[ -3, -3, -3],
    #            [ -2, -2, -2],
    #            [ -1, -1, -1],
    #            [  0,  0,  0],
    #            [  1,  1,  1],
    #            [  2,  2,  2],
    #            [  3,  3,  3]],
    #           [[ -3, -3, -3],
    #            [ -2, -2, -2],
    #            [ -1, -1, -1],
    #            [  0,  0,  0],
    #            [  1,  1,  1],
    #            [  2,  2,  2],
    #            [  3,  3,  3]],
    #           [[ -3, -3, -3],
    #            [ -2, -2, -2],
    #            [ -1, -1, -1],
    #            [  0,  0,  0],
    #            [  1,  1,  1],
    #            [  2,  2,  2],
    #            [  3,  3,  3]],
    #           [[ -3, -3, -3],
    #            [ -2, -2, -2],
    #            [ -1, -1, -1],
    #            [  0,  0,  0],
    #            [  1,  1,  1],
    #            [  2,  2,  2],
    #            [  3,  3,  3]],
    #           [[ -3, -3, -3],
    #            [ -2, -2, -2],
    #            [ -1, -1, -1],
    #            [  0,  0,  0],
    #            [  1,  1,  1],
    #            [  2,  2,  2],
    #            [  3,  3,  3]]]
    #     zv = [[[ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4]],
    #           [[ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4]],
    #           [[ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4]],
    #           [[ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4]],
    #           [[ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4],
    #            [ -4,  0,  4]]]
    #     rijgrid = np.stack((xv, yv, zv), axis=0)
    #     rijcnt = [[[  0,  0,  0], 
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0]], 
    #               [[  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  1,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0]], 
    #               [[  0,  0,  0],  
    #                [  1,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  1],  
    #                [  0,  0,  0]], 
    #               [[  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  1,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0]], 
    #               [[  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0],  
    #                [  0,  0,  0]]] 
    #     rijcnt = np.asarray(rijcnt, dtype=np.float64)

    #     # Test rijcnt_from_frame to see if it returns the expected values. 

    #     test_rijcnt = pt.rijcnt_from_frame(pos, box_config, rijgrid, rcut)
    #     self.assertTrue((test_rijcnt == rijcnt).all())
    #     
    # def test_rijcnt_from_traj(self):

    #     # Define an orthogonal simulation box configuration.

    #     box_config = np.asarray([13, 7, 9, 0, 0, 0])

    #     # Define the positions of particles.

    #     traj = [[[  2.5,     0,     1],
    #              [  3.5,     1,     1],
    #              [    7,    -1,     0]],
    #             [[  5.5,     1,     3],
    #              [ -6.5,     1,     4],
    #              [    0,     0,     0]]]
    #     traj = np.asarray(traj)
    #             
    #     # Define the rij vector cutoff and the number of grind points per axis.

    #     rcut = 2
    #     ngpoints = np.asarray([4, 2, 5])

    #     # Define the rij grid and the average number of particles at rij.

    #     xv =      [[[     -6.5,      -6.5,      -6.5,      -6.5,      -6.5],
    #                 [     -6.5,      -6.5,      -6.5,      -6.5,      -6.5]],
    #                [[-6.5+13/3, -6.5+13/3, -6.5+13/3, -6.5+13/3, -6.5+13/3],
    #                 [-6.5+13/3, -6.5+13/3, -6.5+13/3, -6.5+13/3, -6.5+13/3]],
    #                [[-6.5+26/3, -6.5+26/3, -6.5+26/3, -6.5+26/3, -6.5+26/3],
    #                 [-6.5+26/3, -6.5+26/3, -6.5+26/3, -6.5+26/3, -6.5+26/3]],
    #                [[-6.5+39/3, -6.5+39/3, -6.5+39/3, -6.5+39/3, -6.5+39/3],
    #                 [-6.5+39/3, -6.5+39/3, -6.5+39/3, -6.5+39/3, -6.5+39/3]]]
    #     yv =      [[[     -3.5,      -3.5,      -3.5,      -3.5,      -3.5],
    #                 [      3.5,       3.5,       3.5,       3.5,       3.5]],
    #                [[     -3.5,      -3.5,      -3.5,      -3.5,      -3.5],
    #                 [      3.5,       3.5,       3.5,       3.5,       3.5]],
    #                [[     -3.5,      -3.5,      -3.5,      -3.5,      -3.5],
    #                 [      3.5,       3.5,       3.5,       3.5,       3.5]],
    #                [[     -3.5,      -3.5,      -3.5,      -3.5,      -3.5],
    #                 [      3.5,       3.5,       3.5,       3.5,       3.5]]]
    #     zv =      [[[     -4.5,     -2.25,         0,      2.25,       4.5],
    #                 [     -4.5,     -2.25,         0,      2.25,       4.5]],
    #                [[     -4.5,     -2.25,         0,      2.25,       4.5],
    #                 [     -4.5,     -2.25,         0,      2.25,       4.5]],
    #                [[     -4.5,     -2.25,         0,      2.25,       4.5],
    #                 [     -4.5,     -2.25,         0,      2.25,       4.5]],
    #                [[     -4.5,     -2.25,         0,      2.25,       4.5],
    #                 [     -4.5,     -2.25,         0,      2.25,       4.5]]]
    #     rijgrid = np.stack((xv, yv, zv), axis=0)
    #     traj_rijcnt = [[[[       0,        0,        0,        0,        0],
    #                      [       0,        0,        0,        0,        0]],
    #                     [[       0,        0,        1,        0,        0],
    #                      [       0,        0,        0,        0,        0]],
    #                     [[       0,        0,        0,        0,        0],
    #                      [       0,        0,        1,        0,        0]],
    #                     [[       0,        0,        0,        0,        0],
    #                      [       0,        0,        0,        0,        0]]],
    #                    [[[       0,        0,        0,        0,        0],
    #                      [       0,        0,        0,        0,        0]],
    #                     [[       0,        0,        1,        0,        0],
    #                      [       0,        0,        0,        0,        0]],
    #                     [[       0,        0,        1,        0,        0],
    #                      [       0,        0,        0,        0,        0]],
    #                     [[       0,        0,        0,        0,        0],
    #                      [       0,        0,        0,        0,        0]]]]
    #     traj_rijcnt = np.asarray(traj_rijcnt, dtype=np.float64)

    #     # Test rijcnt_from_traj to see if it returns the expected values. 

    #     test_traj_rijcnt = pt.rijcnt_from_traj(traj, box_config, rijgrid, rcut)
    #     self.assertTrue((test_traj_rijcnt == traj_rijcnt).all())

    # def test_calc_msd(self):

    #     # Define the trajectory.

    #     traj_unwrap = [[[   0,   -5,   40],  
    #                     [   0,    5,    0],  
    #                     [   0,    7,    0],  
    #                     [   0,    7,    2],  
    #                     [   0,    9,    0]],

    #                    [[   0,   -5,   30],  
    #                     [   0,   15,    0],  
    #                     [ -10,    7,    0],  
    #                     [   0,   15,   -4],  
    #                     [   6,    9,    8]]]
    #     traj_unwrap = np.asarray(traj_unwrap, dtype=np.float64)

    #     # Define msd to be returned.

    #     msd = [0, 100]
    #     msd = np.asarray(msd)

    #     # Test msd to see if it returns the expected values.

    #     test_msd = pt.calc_msd(traj_unwrap)
    #     self.assertTrue((test_msd == msd).all())

    # def test_calc_rgt(self):

    #     # Define the positions and masses of the molecule's particles.

    #     pos =     [[  -5,    1,    3],  
    #                [   0,    1,    2],  
    #                [   5,    0,   -9]]  
    #     pos = np.asarray(pos)
    #     mass = [1, 7, 2]
    #     mass = np.asarray(mass)

    #     # Test calc_rg to see if it returns the expected value.

    #     test_calc_rgt = pt.calc_rgt(pos, mass).round(6)
    #     calc_rgt = np.asarray([[ 7.25,  -0.9, -10.45],
    #                            [ -0.9,  0.16,   1.78],
    #                            [-10.45, 1.78,  19.89]])
    #     self.assertTrue((test_calc_rgt == calc_rgt).all())

    def test_center_traj(self):

        # Define the positions and mass.

        traj =   [[[  -3,    0,   -6],  
                   [   0,    0,   -4],  
                   [   3,    0,   42]],
                  [[  -6,    1,    2],
                   [   0,    2,    3],
                   [   6,   -1,    5]]]
        traj = np.asarray(traj)
        molid = [0, 1, 2]
        molid = np.asarray(molid)
        mass = [1, 9, 5]
        mass = np.asarray(mass)
        box_config = [20, 5, 85, 0, 0, 0]
        box_config = np.asarray(box_config)

        # Test center_traj to see if it returns the expected values for the 
        # SYSTEM centering.

        test_traj_centered = pt.center_traj(traj,
                                            molid,
                                            mass,
                                            box_config,
                                            0,
                                            method='SYSTEM')
        test_traj_centered = test_traj_centered.round(6)
        traj_centered = [[[-3.8,  0.,  -6.],
                          [-0.8,  0.,  -4.],
                          [ 2.2,  0.,  42.]],
                         [[-7.6,  1.,   2.],
                          [-1.6,  2.,   3.],
                          [ 4.4, -1.,   5.]]]
        traj_centered = np.asarray(traj_centered)
        self.assertTrue((test_traj_centered == traj_centered).all())

        # Test center_traj to see if it returns the expected values for the 
        # SLAB centering.

        test_traj_centered = pt.center_traj(traj,
                                            molid,
                                            mass,
                                            box_config,
                                            2,
                                            method='SLAB',
                                            ccut=15)
        test_traj_centered = test_traj_centered.round(6)
        traj_centered = [[[ -3.,    0.,   -1.8],
                          [  0.,    0.,    0.2],
                          [  3.,    0.,  -38.8]],
                         [[ -6.,    1.,   -1.6],
                          [  0.,    2.,   -0.6],
                          [  6.,   -1.,    1.4]]]
        traj_centered = np.asarray(traj_centered)
        self.assertTrue((test_traj_centered == traj_centered).all())

if __name__ == '__main__':
    unittest.main()


