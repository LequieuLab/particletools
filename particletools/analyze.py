#!/usr/bin/env python3

'''
Functions to analyze a particle based trajectory.
'''

# TODO Expand functions that use box_config as an argument to use tilt factors.

import numpy as np
from math import sqrt
from numba import jit

@jit(nopython=True)
def img_flags_from_traj(traj_wrap, box_config):
    """
    Calculate the image flags of a wrapped particle-based trajectory. This 
    assumes that the dump frequency is sufficiently high such that particles 
    never travel more than half the length of a box dimension, otherwise image 
    flags may be incorrect. Molecules that are spread across a periodic 
    boundary will have incorrect image flags, potentially introducing errors in
    other calculations. Assumes that the box dimensions are constant in time.
    
    Args:
        traj_wrap: The wrapped trajectory of each particle stored as a 3D 
                   numpy array with dimensions 'frame (ascending order) by 
                   particle ID (ascending order) by particle position 
                   (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz).
    Returns:
        img_flags: The image flags of the particles across the trajectory
                   stored as a 3D numpy array with dimensions 'frame (ascending
                   order) by particle ID (ascending order) by particle image 
                   flag (ix, iy, iz)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = traj_wrap.shape[0]
    nparticles = traj_wrap.shape[1]
    lx, ly, lz = box_config[0:3]
    img_flags = np.zeros(traj_wrap.shape, dtype=np.int32)

    # Loop through each frame but the first, since changes in position between
    # frames are needed to calculate image flags.

    for frame in range(1, nframes):

        # Loop through each particle's index (index = ID - 1).

        for idx in range(nparticles):

            # Get the particle's change in position from the last frame

            del_x = traj_wrap[frame, idx, 0] - traj_wrap[frame - 1, idx, 0]
            del_y = traj_wrap[frame, idx, 1] - traj_wrap[frame - 1, idx, 1]
            del_z = traj_wrap[frame, idx, 2] - traj_wrap[frame - 1, idx, 2]

            # Store any periodic boundary crossings.

            if del_x > lx / 2:
                img_flags[frame:, idx, 0] -= 1
            if del_y > ly / 2:
                img_flags[frame:, idx, 1] -= 1
            if del_z > lz / 2:
                img_flags[frame:, idx, 2] -= 1
            if del_x < lx / -2:
                img_flags[frame:, idx, 0] += 1
            if del_y < ly / -2:
                img_flags[frame:, idx, 1] += 1
            if del_z < lz / -2:
                img_flags[frame:, idx, 2] += 1

    # Return the image flags.

    return img_flags


@jit(nopython=True)
def unwrap_traj(traj_wrap, box_config, img_flags):
    """
    Calculate the unwrapped trajectory of a particle-based trajectory using the
    trajectory's image flags and the simluation box configuration. Assumes that
    the box dimensions are constant in time.
    
    Args:
        traj_wrap: The wrapped trajectory of each particle stored as a 3D numpy
                   array with dimensions 'frame (ascending order) by particle 
                   ID (ascending order) by particle position (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.
        img_flags: The image flags of the particles across the trajectory
                   stored as a 3D numpy array with dimensions 'frame (ascending
                   order) by particle ID (ascending order) by particle image 
                   flag (ix, iy, iz)'.
    Returns:
        traj_unwrap: The unwrapped trajectory of each particle stored as a 3D 
                     numpy array with dimensions 'frame (ascending order) by 
                     particle ID (ascending order) by particle position 
                     (x, y, z)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = traj_wrap.shape[0]
    nparticles = traj_wrap.shape[1]
    lx, ly, lz = box_config[0:3]
    traj_unwrap = np.zeros(traj_wrap.shape)

    # Loop through each frame. 

    for frame in range(nframes):

        # Loop through each particle's index (index = ID - 1).

        for idx in range(nparticles):

            # Unwrap the particle's position based on the image flags and box
            # configuration.

            traj_unwrap[frame, idx, 0] = traj_wrap[frame, idx, 0] +\
                                         img_flags[frame, idx, 0] * lx
            traj_unwrap[frame, idx, 1] = traj_wrap[frame, idx, 1] +\
                                         img_flags[frame, idx, 1] * ly
            traj_unwrap[frame, idx, 2] = traj_wrap[frame, idx, 2] +\
                                         img_flags[frame, idx, 2] * lz

    # Return the unwrapped trajectory.

    return traj_unwrap


@jit(nopython=True)
def mol_com_from_frame(pos, molid, mass):
    """
    Calculate the center of mass and mass of each molecule for a single frame 
    of their trajectory.
    
    Args:
        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:
        mol_com: The center of mass of each molecule stored as a 2D numpy array
                 with dimensions 'molecule ID (ascending order) by molecule 
                 center of mass (x, y, z)'.
        mol_mass: The mass of each molecule stored as a 1D numpy array with
                  dimension 'molecule ID (asecnding order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    mol_com = np.zeros((mols.shape[0], 3))
    mol_mass = np.zeros(mols.shape[0])

    # Calculate the mass weighted position of each particle.

    wt_pos = (pos.T * mass).T

    # Loop through each molecule, find the corresponding particle indices, and
    # then calculate the center of mass of the molecule.

    for mol in mols:
        indices = np.where(molid == mol)
        mol_idx = np.where(mols == mol)
        mol_mass[mol_idx] = np.sum(mass[indices])
        mol_com[mol_idx] = np.sum(wt_pos[indices], axis=0) / mol_mass[mol_idx]

    # Return the molecules' center of masses and masses.

    return mol_com, mol_mass


@jit(nopython=True)
def mol_com_from_traj(traj, molid, mass):
    """
    Calculate the center of mass and mass of each molecule across their 
    trajectory.
    
    Args:
        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:
        traj_mol_com: The center of mass of each molecule across their 
                      trajectory stored as a 2D numpy array with dimensions 
                      'frame (ascending order) by molecule ID (ascending order)
                      by molecule center of mass (x, y, z)'.
        mol_mass: The mass of each molecule stored as a 1D numpy array with
                  dimension 'molecule ID (asecnding order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    nmols = np.unique(molid).shape[0]
    traj_mol_com = np.zeros((nframes, nmols, 3))

    # Loop through each frame and get the center of mass of each molecule.

    for frame in range(nframes):
        mol_com, mol_mass = mol_com_from_frame(traj[frame], molid, mass)
        traj_mol_com[frame] = mol_com

    # Return the molecules' center of masses over the trajectory and masses.

    return traj_mol_com, mol_mass


@jit(nopython=True)
def rg_from_frame(pos, molid, mass, mol_com):
    """
    Calculate the radius of gyration of each molecule for a single frame of
    their trajectory. The radius of gyration is the average distance between
    the particles of a molecule and the molecule's center of mass.
    
    Args:
        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.
        mol_com: The center of mass of each molecule stored as a 2D numpy array
                 with dimensions 'molecule ID (ascending order) by molecule 
                 center of mass (x, y, z)'.

    Returns:
        rg: The radius of gyration of each molecule stored as a 1D numpy array
            with dimension 'molecule ID (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    rg = np.zeros(mols.shape[0])

    # Loop through each molecule, find the corresponding particle indices, and
    # then calculate the radius of gyration of the molecule.

    for mol in mols:
        indices = np.where(molid == mol)[0]
        mol_idx = np.where(mols == mol)
        dr = (pos[indices] - mol_com[mol_idx]).flatten()
        sq_dist = np.dot(dr, dr)
        N = indices.shape[0]
        sq_rg = sq_dist / N
        rg[mol_idx] = sqrt(sq_rg)
        
    # Return the molecules' radii of gyration.

    return rg


@jit(nopython=True)
def rg_from_traj(traj, molid, mass, traj_mol_com):
    """
    Calculate the radius of gyration of each molecule across their trajectory. 
    The radius of gyration is the average distance between the particles of a 
    molecule and the molecule's center of mass.
    
    Args:
        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.
        traj_mol_com: The center of mass of each molecule across their
                      trajectory stored as a 3D numpy array with dimensions 
                      'frame (ascending order) by molecule ID (ascending order)
                      by molecule center of mass (x, y, z)'.

    Returns:
        traj_rg: The radius of gyration of each molecule across their 
                 trajectory stored as a 2D numpy array with dimensions 'frame 
                 (ascending order) by molecule ID (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    nmols = np.unique(molid).shape[0]
    traj_rg = np.zeros((nframes, nmols))

    # Loop through each frame and get the radius of gyration of each molecule.

    for frame in range(nframes):
        traj_rg[frame] = rg_from_frame(traj[frame], molid, mass, 
                                       traj_mol_com[frame])

    # Return the molecules' radii of gyration over the trajectory.

    return traj_rg


@jit(nopython=True)
def density_from_frame(pos, molid, mass, box_config, selection, bin_ax, nbins, 
                       centering='NONE', ccut=350):
    """
    Calculate the density profile of the selected particles along a given axis
    for a single frame.
    
    Args:
        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.
        selection: The selected particles chosen for this calculation stored as
                   a 1D numpy array with dimension 'particle ID' (ascending 
                   order). For example, for density_from_frame, the particles
                   making up the density returned are just the selected 
                   particles.
        bin_ax: The axis along which bins are generated for counting particles.
                In most cases, the bin_ax can have a value of 0 (x-axis), 1
                (y-axis), or 2 (z-axis).
        nbins: The number of bins to generate for counting particles.
        centering: The centering method used when calculating the density
                   profile. Centering can have values of 'NONE' (no centering
                   is performed), 'SYSTEM' (all particle positions are shifted
                   so that the system's center of mass is at the center of the
                   profile), or 'SLAB' (all particle positions are shifted so
                   that the largest cluster of particles is at the center of
                   the profile).
        ccut: The cluster cutoff used for determining clusters in the 'SLAB'
              centering method. For efficiency, molecules are clustered
              together instead of particles, and ccut is the maximum
              distance a molecule can be from the closest molecule in the same
              cluster.

    Returns:
        density_profile: The density profile of the selected particles along a
                         given axis stored as a 2D numpy array with dimensions
                         'bin index by bin properties (position along the axis,
                         density at that position).
    """

    # Calculate the cross section along the bin dimension.

    cross_section = 1
    for dim in range(3):
        if dim == bin_ax:
            continue
        cross_section *= box_config[dim]

    # Create histogram bins along the bin dimension.

    bin_range = box_config[bin_ax]
    bin_lo = bin_range / -2
    bin_hi = bin_range / 2
    bin_width = bin_range / nbins
    bin_vol = bin_width * cross_section
    bin_pos = np.arange(bin_lo, bin_hi, bin_width)
    bin_val = np.zeros(nbins)

    # Define the offset to apply for centering.

    offset = np.zeros(3)
    
    # If the centering method is 'SYSTEM', then calculate the entire system's
    # center of mass and set it as the offset.

    if centering == 'SYSTEM':
        wt_pos = (pos.T * mass).T
        offset = np.sum(wt_pos, axis=0) / np.sum(mass)

    # If the centering method is 'SLAB', then calculate the center of mass of
    # the largest cluster (referred to as the slab) and set it as the offset.
    
    if centering == 'SLAB':

        # Get the center of mass and mass of each molecule.

        mol_com, mol_mass = mol_com_from_frame(pos, molid, mass)

        # Create a contact map of the molecules, which details whether a
        # molecule is within the contact cutoff (ccut) of other molecules.

        nmol = np.unique(molid).shape[0]
        contact_map = np.zeros((nmol, nmol))
        for i in range(nmol):                           
            mol_i_pos = mol_com[i]                     
            for j in range(i, nmol):                                                
                if i == j:
                    contact_map[i, j] += 1  # Contact of a mol with itself.
                    continue
                mol_j_pos = mol_com[j]
                dr = mol_i_pos - mol_j_pos
                dist = sqrt(np.dot(dr, dr))
                if dist <= ccut:                        
                    contact_map[i, j] += 1
                    contact_map[j, i] += 1

        # Compress the contact map by combining rows with shared contacts. This
        # causes each row to represent a unique cluster.
        
        for row in range(contact_map.shape[0]):
            new_contacts = True
            while new_contacts:
                new_contacts = False
                for col in range(contact_map.shape[1]):
                    if row == col:  # Skip contacts of a mol with itself.
                        continue
                    if (contact_map[row, col] != 0 and
                        np.any(contact_map[col])):  # Contact of non-zero row.
                        new_contacts = True
                        contact_map[row] += contact_map[col]
                        contact_map[col].fill(0)

        # From the compressed contact map, find the largest cluster (i.e. the
        # slab) and set the offset to the slab's center of mass.

        cluster_sizes = np.count_nonzero(contact_map, axis=1)
        slab = np.argmax(cluster_sizes)
        slab_mols = np.nonzero(contact_map[slab])
        wt_mol_com = (mol_com.T * mol_mass).T
        offset = np.sum(wt_mol_com[slab_mols], axis=0) /\
                 np.sum(mol_mass[slab_mols])


    # Filter the particle positions based on the selection array. 
    
    pos_sel = pos[selection, :]

    # For each selected particle, apply the center of mass offset and bin it.
    # Periodic boundary conditions are applied after the offset.

    for i in range(pos_sel.shape[0]):
            pos_i = pos_sel[i, bin_ax] - offset[bin_ax]
            if pos_i >= bin_hi:
                pos_i -= bin_range
            if pos_i < bin_lo:
                pos_i += bin_range
            bin_idx = int(((pos_i / bin_range + 1 / 2) * nbins))
            if bin_idx < 0 or bin_idx >= nbins:
                print('Error: invalid bin index created.')
                bin_val.fill(0)
                break
                # TODO Put in a Numba friendly way of exiting the code instead
                # of filling the bin_val array with zeros.
                # exit(1)
            bin_val[bin_idx] += 1

    # Convert the binned particles to a density profile.

    bin_conc = bin_val / bin_vol
    density_profile = np.column_stack((bin_pos, bin_conc))

    # Return the density profile across the desired dimension of the box.

    return density_profile


@jit(nopython=True)
def density_from_traj(traj, molid, mass, box_config, selection, bin_ax, nbins,
                      centering='NONE', ccut=350):
    """
    Calculate the average density profile of the selected particles along a 
    given axis over their trajectory.
    
    Args:
        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.
        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.
        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.
        selection: The selected particles chosen for this calculation stored as
                   a 1D numpy array with dimension 'particle ID' (ascending 
                   order). For example, for density_from_frame, the particles
                   making up the density returned are just the selected 
                   particles.
        bin_ax: The axis along which bins are generated for counting particles.
                In most cases, the bin_ax can have a value of 0 (x-axis), 1
                (y-axis), or 2 (z-axis).
        nbins: The number of bins to generate for counting particles.
        centering: The centering method used when calculating the density
                   profile. Centering can have values of 'NONE' (no centering
                   is performed), 'SYSTEM' (all particle positions are shifted
                   so that the system's center of mass is at the center of the
                   profile), or 'SLAB' (all particle positions are shifted so
                   that the largest cluster of particles is at the center of
                   the profile).
        ccut: The cluster cutoff used for determining clusters in the 'SLAB'
              centering method. For efficiency, molecules are clustered
              together instead of particles, and ccut is the maximum
              distance a molecule can be from the closest molecule in the same
              cluster.

    Returns:
        density_profile: The density profile of the selected particles along a
                         given axis stored as a 2D numpy array with dimensions
                         'bin index by bin properties (position along the axis,
                         density at that position).
    """
    # Get the number of simulation frames and preallocate a density profile.
    
    nframes = traj.shape[0]
    density_profile = np.zeros((nbins, 2))

    # Loop through each frame and sum the density profiles.

    for frame in range(nframes):
        pos = traj[frame]
        density_profile += density_from_frame(pos, molid, mass, box_config,
                                              selection, bin_ax, nbins,
                                              centering, ccut)

    # Return the average density profile over the frames.

    density_profile /= nframes
    return density_profile 
