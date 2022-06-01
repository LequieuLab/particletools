#!/usr/bin/env python3

import unittest
import particletools as pt
import numpy as np
import mdtraj as md

class TestAnalyzeFunctions(unittest.TestCase):

    # def test_get_img_flags(self):

    def test_get_mol_com(self):
        xyz = np.load('box_xyz.npy')
        beads_per_mol = np.load('box_beads_per_mol.npy')
        box_dims = np.load('box_box_dims.npy')
        img_flags = np.load('box_img_flags.npy')
        mdata = np.load('box_mdata.npy')
        mol_com = pt.get_mol_com(xyz, beads_per_mol, box_dims, img_flags, 
                                 mdata)
        test_mol_com = np.load('box_mol_com.npy')
        error = np.abs(test_mol_com - mol_com)
        self.assertTrue((error < 1e-6).all())
        
    # def test_calc_rg(self):

    # def test_profile_density(self):


if __name__ == '__main__':
    unittest.main()
