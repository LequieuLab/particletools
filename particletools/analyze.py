#!/usr/bin/env python3

'''
Functions to analyze a particle based trajectory.
'''

# TODO Expand functions that use box_config as an argument to use tilt factors.
# TODO Add functions that take write psfs and trajectories from data.
# TODO Merge functions that have from traj and from pos to be only one of them.
# TODO Rewrite and recope functions to minimize arguments.
# TODO Reformat in general and get sphinx documents working.
# TODO Add demos that use these functions on toy systems.

import numpy as np
from math import sqrt
from math import floor
from math import pi
from numba import jit

@jit(nopython=True)
def img_flags_from_traj(traj_wrap, box_config):
    """Estimate image flags from a trajectory.

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
        The image flags of the particles across the trajectory 
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

            img_flags[frame:, idx, 0] += int(-2 * del_x / lx)
            img_flags[frame:, idx, 1] += int(-2 * del_y / ly)
            img_flags[frame:, idx, 2] += int(-2 * del_z / lz)

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

        The unwrapped trajectory of each particle stored as a 3D 
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

        A tuple where the first element is the center of mass of each molecule
        stored as a 2D numpy array with dimensions 'molecule ID (ascending 
        order) by molecule center of mass (x, y, z)'. The second element is the
        mass of each molecule stored as a 1D numpy array with dimension 
        'molecule ID (asecnding order)'.
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
    Calculate the center of mass and mass of each molecule for every frame.
    
    Args:

        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.

        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.

        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:

        A tuple where the first element is the center of mass of each molecule 
        for every frame stored as a 2D numpy array with dimensions 'frame 
        (ascending order) by molecule ID (ascending order) by molecule center 
        of mass (x, y, z)'. The second element is the mass of each molecule 
        stored as a 1D numpy array with dimension 'molecule ID (asecnding 
        order)'.
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
def calc_rg(pos, mass):
    """
    Calculate the radius of gyration for a molecule. The radius of gyration is 
    the mass-averaged distance between the particles of a molecule and the 
    molecule's center of mass.
    
    Args:

        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.

        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:

        The radius of gyration of the molecule stored as a float.
    """
    
    # Find the number of particles.

    nparticles = pos.shape[0]

    # Find the total mass of the molecule.

    m_total = np.sum(mass)

    # Calculate the center of mass for the molecule.

    com = np.sum(pos * mass.reshape(nparticles, 1), axis=0) / m_total

    # Calculate the distance between each particle and the molecule's com.

    s = pos - com
    s2 = np.sum(s * s, axis=1)

    # Calculate Rg.

    rg = sqrt(np.sum(s2 * mass, axis=0) / m_total)

    # Return Rg.
    
    return rg


@jit(nopython=True)
def center_traj(traj, molid, mass, box_config, axis, method='SYSTEM', 
                ccut=350):
    """
    Center a trajectory along a given axis using the given method.
    
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

        axis: The axis along which to center the trajectory.

        method: The centering method used when calculating the density profile.
                Method can have values of 'SYSTEM' (all particle positions are
                shifted so that the entire system's center of mass is at the 
                center of the chosen axis) or 'SLAB' (all particle positions 
                are shifted so that the largest cluster of particles' center of
                mass is at the center of the chosen axis).

        ccut: The cluster cutoff used for determining clusters in the 'SLAB'
              centering method. For computational tractability, molecules are 
              clustered together instead of particles, and ccut is the maximum
              distance a molecule can be from the closest molecule in the same
              cluster.

    Returns:

        The centered trajectory of each particle stored as a 3D numpy array 
        with dimensions 'frame (ascending order) by particle ID (ascending
        order) by particle position (x, y, z)'.
    """

    # Preallocate memory for traj_centered and the offset to apply.

    traj_centered = np.zeros(traj.shape)
    offset = np.zeros(3)

    # Loop through each frame.

    for frame in range(traj.shape[0]): 
        offset.fill(0)
        pos = traj[frame]
    
        # If the centering method is 'SYSTEM', then calculate the entire 
        # system's center of mass and set it as the offset.

        if method == 'SYSTEM':
            wt_pos = (pos.T * mass).T
            offset = np.sum(wt_pos, axis=0) / np.sum(mass)

        # If the centering method is 'SLAB', then calculate the center of mass 
        # of the largest cluster (referred to as the slab) and set it as the 
        # offset.
        
        if method == 'SLAB':

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

            # Compress the contact map by combining rows with shared contacts. 
            # This causes each row to represent a unique cluster.
            
            for row in range(contact_map.shape[0]):
                new_contacts = True
                while new_contacts:
                    new_contacts = False
                    for col in range(contact_map.shape[1]):
                        if row == col:  # Skip contacts of a mol with itself.
                            continue
                        if (contact_map[row, col] != 0 and
                            np.any(contact_map[col])):  # Contact non-zero row.
                            new_contacts = True
                            contact_map[row] += contact_map[col]
                            contact_map[col].fill(0)

            # From the compressed contact map, find the largest cluster (i.e. 
            # the slab) and set the offset to the slab's center of mass.

            cluster_sizes = np.count_nonzero(contact_map, axis=1)
            slab = np.argmax(cluster_sizes)
            slab_mols = np.nonzero(contact_map[slab])
            wt_mol_com = (mol_com.T * mol_mass).T
            offset = np.sum(wt_mol_com[slab_mols], axis=0) /\
                     np.sum(mol_mass[slab_mols])

        # Apply the offset to the trajectory and apply the periodic boundary
        # condition.

        for i in range(pos.shape[0]):
            pos_centered = pos[i, axis] - offset[axis]
            if pos_centered >= box_config[axis] / 2:
                pos_centered -= box_config[axis]
            if pos_centered < box_config[axis] / -2:
                pos_centered += box_config[axis]
            traj_centered[frame, i] = pos[i]
            traj_centered[frame, i, axis] = pos_centered

    # Return the centered trajectory.

    return traj_centered


@jit(nopython=True)
def density_from_frame(pos, molid, mass, box_config, selection, bin_axis, 
                       nbins, centering='NONE', ccut=350):
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

        bin_axis: The axis along which bins are generated for counting 
                  particles. In most cases, the bin_axis can have a value of 0 
                  (x-axis), 1 (y-axis), or 2 (z-axis).

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

        The density profile of the selected particles along a given axis stored
        as a 2D numpy array with dimensions 'bin index by bin properties 
        (position along the axis, density at that position).
    """

    # Calculate the cross section along the bin dimension.

    cross_section = 1
    for dim in range(3):
        if dim == bin_axis:
            continue
        cross_section *= box_config[dim]

    # Create histogram bins along the bin dimension.

    bin_range = box_config[bin_axis]
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
            pos_i = pos_sel[i, bin_axis] - offset[bin_axis]
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
def density_from_traj(traj, molid, mass, box_config, selection, bin_axis, 
                      nbins, centering='NONE', ccut=350):
    """
    Calculate the average density profile of the selected particles along a 
    given axis for every frame.
    
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

        bin_axis: The axis along which bins are generated for counting 
                  particles. In most cases, the bin_axis can have a value of 0 
                  (x-axis), 1 (y-axis), or 2 (z-axis).

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

        The density profile of the selected particles along a given axis per 
        frame stored as a 3D numpy array with dimensions 'frame (ascending 
        order) by bin index by bin properties (position along the axis, density
        at that position)'.
    """
    # Get the number of simulation frames and preallocate a density profile.
    
    nframes = traj.shape[0]
    traj_density_profile = np.zeros((nframes, nbins, 2))

    # Loop through each frame and sum the density profiles.

    for frame in range(nframes):
        pos = traj[frame]
        traj_density_profile[frame] = density_from_frame(pos, molid, mass, 
                                                         box_config, selection, 
                                                         bin_axis, nbins, 
                                                         centering, ccut)

    # Return the density profile per frame.

    return traj_density_profile 


@jit(nopython=True)
def meshgrid3D(x, y, z):
    """
    Create a 3D mesh and return the gridpoints of that mesh for the x, y, and z
    axes. Analagous to np.meshgrid(x, y, z, indexing='ij'), except meshgrid3D
    is compatible with Numba's jit compilation in nopython mode.
    
    Args:

        x: The x-axis values of the 3D mesh stored as a 1D numpy array with
           dimension 'x-index'.

        y: The y-axis values of the 3D mesh stored as a 1D numpy array with
           dimension 'y-index'.

        z: The z-axis values of the 3D mesh stored as a 1D numpy array with
           dimension 'z-index'.

    Returns:

        The gridpoint values of each axis in the 3D mesh stored as a 4D numpy
        array with dimension 'axis (x, y, z) by x-index by y-index by z-index'.
    """
    
    # TODO Replace this with an n-dimensional version.

    # Preallocate each 3D numpy array.

    shape = (x.size, y.size, z.size)
    xv = np.zeros(shape)
    yv = np.zeros(shape)
    zv = np.zeros(shape)

    # Store the values of each axis at each gridpoint in their respective 3D
    # array, and then store the arrays together as a 4D array.

    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                xv[i, j, k] = x[i] 
                yv[i, j, k] = y[j]
                zv[i, j, k] = z[k]
    grid = np.stack((xv, yv, zv), axis=0)

    # Return the gridpoint values of each axis in the 3D mesh.

    return grid


@jit(nopython=True)
def rijcnt_from_frame(pos, box_config, rijgrid, rcut):
    """
    Count the number of particles separated by a vector rij for a single frame.
    rij is equal to the position vector of particle j minus the position vector
    of particle i.
    
    Args:

        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.

        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.

        rijgrid: The gridpoint values of each axis in the 3D mesh, that the rij
                 vector is placed on, stored as a 4D numpy array with dimension
                 'axis (x, y, z) by x-index by y-index by z-index'.

        rcut: The rij vector cutoff used to determine the maximum length the
              rij vector can be along a single axis. Due to the minimum image 
              convention, rcut cannot be greater than the shortest simulation 
              box length.

    Returns:

        The number of particles separated by a vector rij stored as a 3D numpy
        array with dimensions 'x-index by y-index by z-index', where these 
        indices are the indices of the rij vector placed on the 3D mesh defined
        by rijgrid.
    """
    
    # TODO Allow gridpoints to be uneven. Can prolly do this by doing binning
    # with np.argmin(np.abs(difference between value and available bins).

    # Get simulation parameters from the arguments, limit rcut if needed, and
    # preallocate rijcnt.

    lx, ly, lz = box_config[0:3]
    rcut_lim = min(box_config[0:3]) / 2
    if rcut > rcut_lim:
        rcut = rcut_lim
    x_lo = np.amin(rijgrid[0])
    y_lo = np.amin(rijgrid[1])
    z_lo = np.amin(rijgrid[2])
    ngpoints = rijgrid[0].shape
    rijcnt = np.zeros(ngpoints)

    # Loop through each pair of particles and place the rij vector on the grid.

    nparticles = pos.shape[0]
    for i in range(nparticles - 1):
        for j in range(i + 1, nparticles):
            
            # Get the rij vector and apply the minimum image convention.
            
            rij = pos[j] - pos[i]
            rij[0] += int(-2 * rij[0] / lx) * lx
            rij[1] += int(-2 * rij[1] / ly) * ly
            rij[2] += int(-2 * rij[2] / lz) * lz

            # Check if the length of rij is greater than rcut.

            if sqrt(np.dot(rij, rij)) > rcut:
                continue

            # Place the rij vector on the nearest gridpoint.

            x_ind = round((rij[0] - x_lo) * (ngpoints[0] - 1) / lx)
            y_ind = round((rij[1] - y_lo) * (ngpoints[1] - 1) / ly)
            z_ind = round((rij[2] - z_lo) * (ngpoints[2] - 1) / lz)
            rijcnt[x_ind, y_ind, z_ind] += 1

            # # Place the rji vector (which represents the reverse order in
            # # pairing) on the nearest gridpoint.

            rji = -1 * rij
            x_ind = round((rji[0] - x_lo) * (ngpoints[0] - 1) / lx)
            y_ind = round((rji[1] - y_lo) * (ngpoints[1] - 1) / ly)
            z_ind = round((rji[2] - z_lo) * (ngpoints[2] - 1) / lz)
            rijcnt[x_ind, y_ind, z_ind] += 1

    # Return the number of particles at rij.

    return rijcnt

 
@jit(nopython=True)
def rijcnt_from_traj(traj, box_config, rijgrid, rcut):
    """
    Count the number of particles separated by a vector rij for each frame. rij
    is equal to the position vector of particle j minus the position vector of 
    particle i.
    
    Args:

        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.

        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.

        rijgrid: The gridpoint values of each axis in the 3D mesh, that the rij
                 vector is placed on, stored as a 4D numpy array with dimension
                 'axis (x, y, z) by x-index by y-index by z-index'.

        rcut: The rij vector cutoff used to determine the maximum length the
              rij vector can be along a single axis. Due to the minimum image 
              convention, rcut cannot be greater than the shortest simulation 
              box length.

    Returns:

        The number of particles separated by a vector rij per frame stored as a
        4D numpy array with dimensions 'frame (ascending order) by x-index by 
        y-index by z-index', where these indices are the indices of the rij 
        vector placed on the 3D mesh defined by rijgrid.
    """
    
    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    traj_rijcnt = np.zeros((nframes, rijgrid[0].shape[0], rijgrid[0].shape[1],
                           rijgrid[0].shape[2]))

    # Loop through each frame and get the number of particles at rij per frame.

    for frame in range(nframes):
        traj_rijcnt[frame] = rijcnt_from_frame(traj[frame], box_config, 
                                               rijgrid, rcut)

    # Return the number of particles at rij per frame.

    return traj_rijcnt

@jit(nopython=True)
def calc_msd(traj_unwrap):
    """
    Count the number of particles separated by a vector rij for each frame. rij
    is equal to the position vector of particle j minus the position vector of 
    particle i.
    
    Args:

        traj_unwrap: The unwrapped trajectory of each particle stored as a 3D 
                     numpy array with dimensions 'frame (ascending order) by 
                     particle ID (ascending order) by particle position 
                     (x, y, z)'.

    Returns:

        The mean-squared displacement of an unwrapped trajectory stored as a 1D
        numpy array with dimension frame (ascending order).
    """
    
    # Get the number of frames and particles.

    nframes = traj_unwrap.shape[0]
    nparticles = traj_unwrap.shape[1]
    
    # Preallocate sd.

    sd = np.zeros((nframes, nparticles))

    # Loop through each particle.

    for particle in range(nparticles):

        # Loop through each frame.

        initpos = traj_unwrap[0, particle]
        for frame in range(1, nframes):
            
            # Calculate the squared displacement.

            pos = traj_unwrap[frame, particle]
            delr = pos - initpos
            sd[frame, particle] = np.dot(delr, delr)

    # Return the mean-squared displacement.

    return np.sum(sd, axis=1) / sd.shape[1]

@jit(nopython=True)
def calc_rdf(pos, box_config, r_arr):
    """
    Calculate the radial distribution function for a frame of a simulation.
    
    Args:

        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.

        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz)'.

        r_arr: A 1D numpy array of radius values to use in calculating the rdf.

    Returns:

        The radial distribution function for a frame of a simulation stored as
        a 1D numpy array with dimension radius (matches r_arr).
    """

    # Get the number of particles and box dimensions.

    nparticles = pos.shape[0]
    lx, ly, lz = box_config[0:3]
    
    # Preallocate rdf and calculate dr.

    nbins = r_arr.size
    rdf = np.zeros(nbins)
    dr = r_arr[1] - r_arr[0]

    # Loop through each pair of particles.

    for i in range(nparticles - 1):
        for j in range(i + 1, nparticles):

            # Get the rij vector and apply the minimum image convention.

            rij = pos[j] - pos[i]
            rij[0] += int(-2 * rij[0] / lx) * lx
            rij[1] += int(-2 * rij[1] / ly) * ly
            rij[2] += int(-2 * rij[2] / lz) * lz

            # Calculate the distance between the pair.

            distance = np.linalg.norm(rij)

            # Store the pair in rdf.
            
            idx = floor(distance / dr)
            if idx >= nbins:
                continue
            rdf[idx] += 2

    # Normalize rdf.

    box_vol = lx * lx * lx
    rho0 = nparticles / box_vol
    x = r_arr + dr
    ideal = 4 / 3 * np.pi * rho0 * (x * x * x - r_arr * r_arr * r_arr)
    rdf /= ideal
    rdf /= nparticles

    # Return rdf.

    return rdf


@jit(nopython=True)
def calc_S(q_arr, r_arr, rdf, rho0):
    """
    Calculate the structure factor from the radial distribution function. See
    page 73 of Allen and Tildesley (2nd edition) for more.
    
    Args:

        q_arr: A 1D numpy array of scattering length values to use in
               calculating the structure factor.

        r_arr: A 1D numpy array of radius values used in calculating the rdf.

        rdf: The radial distribution function for a frame of a simulation
             stored as a 1D numpy array with dimension radius (matches r_arr).

        rho0: The bulk density for the particles used in calculating the rdf
              stored as a float.

    Returns:

        The structure factor stored as a 1D numpy array with dimension 
        scattering length (matches q_arr).
    """

    # Preallocate memory for the structure factor.
    
    S = np.zeros(q_arr.size)

    # Perform the Fourier transform of the radial distribution function.

    for i, q in enumerate(q_arr):
        integrand = r_arr * (rdf - 1) * np.sin(q * r_arr)
        
        # Evaluate the integrand with the composite Simpson's 3/8 rule.

        a = r_arr[0]
        b = r_arr[-1]
        n = r_arr.size - 1
        h = (b - a) / n
        integral = 0
        for j in range(1, int(n / 3) + 1):
            integral += integrand[3 * j - 3] + 3 * integrand[3 * j - 2] +\
                        3 * integrand[3 * j - 1] + integrand[3 * j]
        integral *= 3 / 8 * h
        S[i] = 1 + 4 * pi * rho0 / q * integral

    # Return the structure factor.

    return S



@jit(nopython=True)
def calc_rgt(pos, mass):
    """
    Calculate the radius of gyration tensor for a molecule. The radius of 
    gyration tensor is a 3 by 3 matrix that describes the shape of a molecule.
    Each element is effectively a squared radius of gyration calculation in 1
    dimension or a convolution of 2 dimensions. When diagonlized, the radius of
    gyration tensor returns the squared radius of gyration through a sum of the
    diagonal elements.
    
    Args:

        pos: The position of each particle stored as a 2D numpy array with
             dimensions 'particle ID (ascending order) by particle position 
             (x, y, z)'.

        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:

        The radius of gyration tensor of a molecule stored as a 2D numpy array
        with dimensions 1st axis (x, y, z) by 2nd axis (x, y, z).
    """

    # Preallocate the rg tensor.

    rgt = np.zeros((3, 3))
    
    # Find the number of particles.

    nparticles = pos.shape[0]

    # Find the total mass of the molecule.

    m_total = np.sum(mass)

    # Calculate the center of mass for the molecule in x, y, and z.

    x_com = np.sum(pos[:, 0] * mass) / m_total
    y_com = np.sum(pos[:, 1] * mass) / m_total
    z_com = np.sum(pos[:, 2] * mass) / m_total

    # Calculate each element of the Rg tensor.

    xx = np.sum(mass * (pos[:, 0] - x_com) ** 2) / m_total
    xy = np.sum(mass * (pos[:, 0] - x_com) * (pos[:, 1] - y_com)) / m_total
    xz = np.sum(mass * (pos[:, 0] - x_com) * (pos[:, 2] - z_com)) / m_total
    yy = np.sum(mass * (pos[:, 1] - y_com) ** 2) / m_total
    yz = np.sum(mass * (pos[:, 1] - y_com) * (pos[:, 2] - z_com)) / m_total
    zz = np.sum(mass * (pos[:, 2] - z_com) ** 2) / m_total
    rgt = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    # Return the Rg tensor.
    
    return rgt

@jit(nopython=True)
def mol_rg_from_frame(pos, molid, mass):
    """
    Calculate the radius of gyration of each molecule for a single frame
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

        The Rg of each molecule stored as a 1D numpy array with dimensions
        'molecule ID (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    molrg = np.zeros(mols.shape[0])

    # Loop through each molecule and calculate its radius of gryation.

    for mol in mols:
        indices = np.where(molid == mol)
        mol_idx = np.where(mols == mol)
        mol_mass = mass[indices]
        mol_pos = pos[indices]
        molrg[mol_idx] = calc_rg(mol_pos, mol_mass)
    
    # Return the molecules' radii of gyration.

    return molrg

@jit(nopython=True)
def mol_rg_from_traj(traj, molid, mass):
    """
    Calculate the radius of gyration for each molecule for every frame.
    
    Args:

        traj: The trajectory of each particle stored as a 3D numpy array with
              dimensions 'frame (ascending order) by particle ID (ascending 
              order) by particle position (x, y, z)'.

        molid: The molecule ID of each particle stored as a 1D numpy array with
               dimension 'particle ID (ascending order)'.

        mass: The mass of each particle stored as a 1D numpy array with
              dimension 'particle ID (ascending order)'.

    Returns:

        The radius of gyration of each molecule for every frame stored as a 
        2D numpy array with dimensions 'frame (ascending order) by molecule ID 
        (ascending order)'.
    """

    # Get simulation parameters from the arguments and preallocate arrays.
    
    nframes = traj.shape[0]
    nmols = np.unique(molid).shape[0]
    traj_mol_rg = np.zeros((nframes, nmols))

    # Loop through each frame and get the radius of gyration of each molecule.

    for frame in range(nframes):
        mol_rg  = mol_rg_from_frame(traj[frame], molid, mass)
        traj_mol_rg[frame] = mol_rg

    # Return the molecules' radii of gyration over the trajectory.

    return traj_mol_rg

