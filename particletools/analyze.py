#!/usr/bin/env python3

'''
Functions to analyze a monte carlo (MC) or molecular dynamics (MD) trajectory.
'''

import numpy as np
from math import sqrt
from numba import jit

@jit(nopython=True)
def get_mol_com(xyz, beads_per_mol, box_dims, img_flags, mass_data):
    """
    Get the center of mass for each molecule at every frame of a trajectory.
    
    Args:
        xyz: the coordinates of a trajectory stored as a 3D numpy array, where
             the dimensions are (in order) simulation frame, atom number, and
             xyz coordinates.
        beads_per_mol: the number of particles per molecule stored as a 1D
                       numpy array where the index equals the molecule number.
        box_dims: the xyz values of the simulation box stored as a 1D numpy
                  array, where index 0 is the x-dimension, index 1 is the
                  y-dimension, and index 2 is the z-dimension.
        img_flags: the trajectory's image flags stored as a 3D numpy array ... 
                   (finish and test documentation later).
        mass_data: 1D numpy array of particle masses where the index is particle_ID.

    Returns:
        mol_com: and me!
    """

    # TODO Finish and test docstring using Sphinx.
    # TODO Better name than beads for particles? And are comments appropiate?
    # TODO Importantly, there is a better way to structure these calculations
    # by having optional arguments, so that unwrapping and building of mol_data
    # is not done multiple times when going from get_mol -> calc_rg.

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = xyz.shape[0]
    nmolec = beads_per_mol.shape[0]
    largest_mol = np.amax(beads_per_mol)
    mol_data = np.zeros((4, largest_mol))
    mol_com = np.zeros((nframes, nmolec, 3))

    # Loop through each frame.

    for frame in range(nframes):
        
        # Loop through each molecule.

        mol_start = 0
        for mol_num in range(nmolec):

            # Find the starting bead number for this molecule and the next.

            mol_end = beads_per_mol[mol_num] + mol_start 

            # Loop through each bead in the current molecule.

            for bead_num in range(mol_start, mol_end):

                # Unwrap the bead's coordinates and find its mass. 

                x = xyz[frame][bead_num][0] +\
                    np.sum(img_flags[:frame + 1, bead_num, 0]) *\
                    box_dims[0]
                y = xyz[frame][bead_num][1] +\
                    np.sum(img_flags[:frame + 1, bead_num, 1]) *\
                    box_dims[1]
                z = xyz[frame][bead_num][2] +\
                    np.sum(img_flags[:frame + 1, bead_num, 2]) *\
                    box_dims[2]
                mass = mass_data[bead_num]

                # Store the bead's unwrapped coordinates and mass with 
                # other beads of the same molecule.

                mol_data[0][bead_num - mol_start] = x
                mol_data[1][bead_num - mol_start] = y
                mol_data[2][bead_num - mol_start] = z
                mol_data[3][bead_num - mol_start] = mass

            # Calculate and store the molecule's center of mass.

            mass_weighted_x = np.sum(mol_data[0] * mol_data[3])
            mass_weighted_y = np.sum(mol_data[1] * mol_data[3])
            mass_weighted_z = np.sum(mol_data[2] * mol_data[3])
            total_mass = np.sum(mol_data[3])
            mol_com[frame][mol_num][0] = mass_weighted_x / total_mass
            mol_com[frame][mol_num][1] = mass_weighted_y / total_mass
            mol_com[frame][mol_num][2] = mass_weighted_z / total_mass

            # Update for the next molecule.

            mol_start = mol_end
            mol_data.fill(0)

    return mol_com


@jit(nopython=True)
def calc_rg(xyz, mol_com, beads_per_mol, box_dims, img_flags, mass_data):
    """
    Get the center of mass for each molecule at every frame of a trajectory.
    
    Args:
        xyz: the coordinates of a trajectory stored as a 3D numpy array, where
             the dimensions are (in order) simulation frame, atom number, and
             xyz coordinates.
        beads_per_mol: the number of particles per molecule stored as a 1D
                       numpy array where the index equals the molecule number.
        box_dims: the xyz values of the simulation box stored as a 1D numpy
                  array, where index 0 is the x-dimension, index 1 is the
                  y-dimension, and index 2 is the z-dimension.
        img_flags: the trajectory's image flags stored as a 3D numpy array ... 
                   (finish and test documentation later).
        mass_data: 1D numpy array of particle masses where the index is particle_ID.
        mol_com: now me!

    Returns:
        calc_rg: and me!
    """

    # TODO Finish and test docstring using Sphinx.
    # TODO Better name than beads for particles? And are comments appropiate?
    # TODO Better name than rg_data?

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = xyz.shape[0]
    nmolec = beads_per_mol.shape[0]
    largest_mol = np.amax(beads_per_mol)
    bead_xyz = np.zeros(3)
    rg_data = np.zeros((nframes, nmolec))

    # Loop through each frame.

    for frame in range(nframes):
        
        # Loop through each molecule.

        mol_start = 0
        for mol_num in range(nmolec):

            # Find the starting bead number for this molecule and the next.

            mol_end = beads_per_mol[mol_num] + mol_start
        
            # Loop through each bead in the current molecule.

            rg_sum = 0
            for bead_num in range(mol_start, mol_end):

                # Unwrap the bead's coordinates. 

                bead_xyz[0] = xyz[frame][bead_num][0] +\
                              np.sum(img_flags[:frame + 1, bead_num, 0]) *\
                              box_dims[0]
                bead_xyz[1] = xyz[frame][bead_num][1] +\
                              np.sum(img_flags[:frame + 1, bead_num, 1]) *\
                              box_dims[1]
                bead_xyz[2] = xyz[frame][bead_num][2] +\
                              np.sum(img_flags[:frame + 1, bead_num, 2]) *\
                              box_dims[2]

                # Calculate the bead's squared distance from the molecule's
                # center of mass.

                dr = bead_xyz - mol_com[frame][mol_num]
                rg_sum += np.dot(dr, dr)
            
            # Calculate the molecule's radius of gyration for the frame.

            rg_data[frame][mol_num] = sqrt(rg_sum / beads_per_mol[mol_num])

            # Update for the next molecule.

            mol_start = mol_end 

    return rg_data
