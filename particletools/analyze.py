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
        mol_com: 2D numpy array with dimensions of frame and molecule ID, holds the
                 value of the molecule's center of mass for a given frame.
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
        mol_com: 2D numpy array with dimensions of frame and molecule ID, holds the
                 value of the molecule's center of mass for a given frame.

    Returns:
        calc_rg: 2D numpy array with dimensions of frame and molecule ID, holds the
                 value of the molecule's radius of gyration for a given frame.
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


@jit(nopython=True)
def bead_heatmap(xyz, bead_list, fixed_dim, rcut, nbins, box_dims, img_flags):
    """
    Create a 2D heatmap showing the relative density of nearby beads. The
    resolution of the heatmap cover the rij vector (which connects bead 1 to
    bead 2) to a maximum of rcut, having free rotation in every dimension but
    the fixed dimension given by fixed_dim.
    
    Args:
        xyz: the coordinates of a trajectory stored as a 3D numpy array, where
             the dimensions are (in order) simulation frame, atom number, and
             xyz coordinates.
        bead_list: 1D numpy array with the index of every bead of interest.
        fixed_dim: Dimension to not allow rij to rotate in. Either x, y, or z.
                   In the rij vector, it sets that dimension's component to 0.
        rcut: maximum length for the rij vector.
        nbins: number of bins for a dimension of the rij vector.
        box_dims: the xyz values of the simulation box stored as a 1D numpy
                  array, where index 0 is the x-dimension, index 1 is the
                  y-dimension, and index 2 is the z-dimension.
        img_flags: the trajectory's image flags stored as a 3D numpy array ... 
                   (finish and test documentation later).

    Returns:
        fig_heatmap, ax_heatmap: matplotlib figure and axes of the heatmap
    """
    
    # TODO Make an rij matrix that is rij_matrix[ri_value][rj_value].
    # TODO Go through each rij in the simulation and bin into a bin matrix (of
    #      shape equal to rij_matrix). Find closest point.
    # TODO Generate a heatmap from the bin matrix.







@jit(nopython=True)
def get_img_flags(xyz, box_dims):
    """
    Calculate the image flags of a trajectory. This assumes that the dump
    frequency is sufficiently high such that beads never travel more than half
    the length of a box dimension.
    
    Args:
        xyz: the coordinates of a trajectory stored as a 3D numpy array, where
             the dimensions are (in order) simulation frame, atom number, and
             xyz coordinates.
        box_dims: the xyz values of the simulation box stored as a 1D numpy
                  array, where index 0 is the x-dimension, index 1 is the
                  y-dimension, and index 2 is the z-dimension.
    Returns:
        img_flags: the trajectory's image flags stored as a 3D numpy array ... 
                   (finish and test documentation later).
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = xyz.shape[0]
    nbeads = xyz.shape[1]
    img_flags = np.zeros(xyz.shape)

    # Loop through each frame but the first, since changes in position between
    # frames are needed to calculate image flags.

    for frame in range(1, nframes):

        # Loop through each bead.

        for bead_num in range(nbeads):

            # Get the bead's change in position from the last frame

            del_x = (xyz[frame][bead_num][0] -\
                     xyz[frame - 1][bead_num][0])
            del_y = (xyz[frame][bead_num][1] -\
                     xyz[frame - 1][bead_num][1])
            del_z = (xyz[frame][bead_num][2] -\
                     xyz[frame - 1][bead_num][2])

            # Store any periodic boundary crossings.

            if del_x > box_dims[0] / 2:
                img_flags[frame][bead_num][0] -= 1
            if del_y > box_dims[1] / 2:
                img_flags[frame][bead_num][1] -= 1
            if del_z > box_dims[2] / 2:
                img_flags[frame][bead_num][2] -= 1
            if del_x < box_dims[0] / -2:
                img_flags[frame][bead_num][0] += 1
            if del_y < box_dims[1] / -2:
                img_flags[frame][bead_num][1] += 1
            if del_z < box_dims[2] / -2:
                img_flags[frame][bead_num][2] += 1

    # Return the image flags.

    return img_flags


