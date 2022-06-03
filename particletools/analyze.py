#!/usr/bin/env python3

'''
Functions to analyze a particle based trajectory.
'''

# TODO Go through and change mol_data to being [mol_num][dimension] for
# consistency.
# TODO Go through and change [x][y][z] notation to [x, y, z].

import numpy as np
from math import sqrt
from numba import jit

@jit(nopython=True)
def img_flags_from_traj(p_wrap_traj, box_config):
    """
    Calculate the image flags of a wrapped particle-based trajectory. This 
    assumes that the dump frequency is sufficiently high such that beads never 
    travel more than half the length of a box dimension, otherwise image flags 
    may be incorrect. Molecules that are spread across a periodic boundary will
    have incorrect image flags, potentially introducing errors in other 
    calculations. Assumes that the box dimensions are constant in time.
    
    Args:
        p_wrap_traj: The wrapped trajectory of each particle stored as a 3D 
                     numpy array with dimensions 'frame by particle number by 
                     particle position (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz).
    Returns:
        traj_img_flags: The image flags of the particles across the trajectory
                        stored as a 3D numpy array with dimensions 'frame by 
                        particle number by particle image flag (ix, iy, iz).
    """

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = p_wrap_traj.shape[0]
    nparticles = p_wrap_traj.shape[1]
    lx, ly, lz = box_config[0:3]
    traj_img_flags = np.zeros(p_wrap_traj.shape, dtype=np.int32)

    # Loop through each frame but the first, since changes in position between
    # frames are needed to calculate image flags.

    for frame in range(1, nframes):

        # Loop through each particle.

        for p_num in range(nparticles):

            # Get the bead's change in position from the last frame

            del_x = p_wrap_traj[frame, p_num, 0] -\
                    p_wrap_traj[frame - 1, p_num, 0]
            del_y = p_wrap_traj[frame, p_num, 1] -\
                    p_wrap_traj[frame - 1, p_num, 1]
            del_z = p_wrap_traj[frame, p_num, 2] -\
                    p_wrap_traj[frame - 1, p_num, 2]

            # Store any periodic boundary crossings.

            if del_x > lx / 2:
                traj_img_flags[frame:, p_num, 0] -= 1
            if del_y > ly / 2:
                traj_img_flags[frame:, p_num, 1] -= 1
            if del_z > lz / 2:
                traj_img_flags[frame:, p_num, 2] -= 1
            if del_x < lx / -2:
                traj_img_flags[frame:, p_num, 0] += 1
            if del_y < ly / -2:
                traj_img_flags[frame:, p_num, 1] += 1
            if del_z < lz / -2:
                traj_img_flags[frame:, p_num, 2] += 1

    # Return the image flags.

    return traj_img_flags


@jit(nopython=True)
def unwrap_traj(p_wrap_traj, box_config, traj_img_flags):
    """
    Calculate the unwrapped trajectory of a particle-based trajectory using the
    trajectory's image flags and the simluation box configuration. Assumes that
    the box dimensions are constant in time.
    
    Args:
        p_wrap_traj: The wrapped trajectory of each particle stored as a 3D 
                     numpy array with dimensions 'frame by particle number by 
                     particle position (x, y, z)'.
        box_config: The simulation box configuration stored as a 1D numpy array
                    of length 6 with the first three elements being box length
                    (lx, ly, lz) and the last three being tilt factors 
                    (xy, xz, yz).
        traj_img_flags: The image flags of the particles across the trajectory
                        stored as a 3D numpy array with dimensions 'frame by 
                        particle number by particle image flag (ix, iy, iz).
    Returns:
        p_unwrap_traj: The unwrapped trajectory of each particle stored as a 3D 
                       numpy array with dimensions 'frame by particle number by 
                       particle position (x, y, z)'.
    """

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

    # Get simulation parameters from the arguments and preallocate arrays.

    nframes = p_wrap_traj.shape[0]
    nparticles = p_wrap_traj.shape[1]
    lx, ly, lz = box_config[0:3]
    p_unwrap_traj = np.zeros(p_wrap_traj.shape)

    # Loop through each frame. 

    for frame in range(nframes):

        # Loop through each particle.

        for p_num in range(nparticles):

            # Unwrap the particle's position based on the image flags and box
            # configuration.

            p_unwrap_traj[frame, p_num, 0] = p_wrap_traj[frame, p_num, 0] +\
                                             traj_img_flags[frame, p_num, 0] *\
                                             lx
            p_unwrap_traj[frame, p_num, 1] = p_wrap_traj[frame, p_num, 1] +\
                                             traj_img_flags[frame, p_num, 1] *\
                                             ly
            p_unwrap_traj[frame, p_num, 2] = p_wrap_traj[frame, p_num, 2] +\
                                             traj_img_flags[frame, p_num, 2] *\
                                             lz

    # Return the unwrapped trajectory.

    return p_unwrap_traj


@jit(nopython=True)
def mol_com_from_frame(p_pos, p_molid, p_mass):
    """
    Calculate the center of mass for each molecule for a single frame of a 
    trajectory.
    
    Args:
        p_pos: The position of each particle stored as a 2D numpy array with
               dimensions 'particle number by particle position (x, y, z)'.
        p_molid: The molecule ID of each particle stored as a 2D numpy array 
                 with dimensions 'particle number by molecule ID'.
        p_mass: The mass of each particle stored as a 2D numpy array with 
                dimensions 'particle number by particle mass'.

    Returns:
        frame_mol_com: The center of mass of each molecule stored as a 2D numpy
                       array with dimensions 'molecule ID by molecule center of
                       mass'.
    """

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
                mass = mdata[bead_num]

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
def calc_rg(xyz, mol_com, beads_per_mol, box_dims, img_flags, mdata):
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
        mdata: 1D numpy array of particle masses where the index is particle_ID.
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
def profile_density(xyz, bead_sel, beads_per_mol, nbins, nsections, ccut,
                    centering, box_dims, mdata):

    # TODO Build docstring.
    # TODO Implement clustering.
    # TODO Have it use get_mol_com for slab centering.

    # Filter the trajectory based on the selected beads.
    
    xyz_sel = xyz[:, bead_sel, :]

    # Find the longest dimension of the box, known as the slab dimension.

    slab_length = max(box_dims)
    slab_cross_section = 1
    for i in range(0, len(box_dims)):
        if box_dims[i] == slab_length:
            slab_dim = i
        else:
            slab_cross_section *= box_dims[i]
    slab_begin = slab_length / -2
    slab_end = slab_length / 2

    # Create histogram bins and dimension array.

    bin_width = slab_length / nbins
    bin_volume = bin_width * slab_cross_section
    dimension = np.arange(slab_begin, slab_end, bin_width)

    # Calculate the first frame to start, the frames per section, and prepare
    # arrays for data storage.

    nframes = xyz.shape[0]
    nbeads = xyz.shape[1]
    frames_per_sec = nframes // nsections
    frame_start = nframes - nsections * frames_per_sec
    sec_num = 0
    beads_per_bin = np.zeros((nsections, nbins))
    slab_molecules_per_frame = np.zeros(nsections * frames_per_sec)
    offset = np.zeros(3)

    # Get the center of mass and mass of each molecule if slab centering is 
    # used and preallocate memory for arrays.
    
    if centering == 'SLAB':
        nmolec = beads_per_mol.shape[0]
        mol_com = np.zeros((nframes, nmolec, 3))
        largest_mol = np.amax(beads_per_mol)
        mol_data = np.zeros((4, largest_mol))
        for frame in range(nframes):
            mol_start = 0
            for mol_num in range(nmolec):
                mol_end = beads_per_mol[mol_num] + mol_start 
                for bead_num in range(mol_start, mol_end):
                    mol_data[0][bead_num - mol_start] = xyz[frame][bead_num][0]
                    mol_data[1][bead_num - mol_start] = xyz[frame][bead_num][1]
                    mol_data[2][bead_num - mol_start] = xyz[frame][bead_num][2]
                    mol_data[3][bead_num - mol_start] = mdata[bead_num]
                mass_weighted_x = np.sum(mol_data[0] * mol_data[3])
                mass_weighted_y = np.sum(mol_data[1] * mol_data[3])
                mass_weighted_z = np.sum(mol_data[2] * mol_data[3])
                total_mass = np.sum(mol_data[3])
                mol_com[frame][mol_num][0] = mass_weighted_x / total_mass
                mol_com[frame][mol_num][1] = mass_weighted_y / total_mass
                mol_com[frame][mol_num][2] = mass_weighted_z / total_mass
                mol_start = mol_end
                mol_data.fill(0)
        mol_mass = np.zeros(nmolec)
        beads_in_cluster = np.zeros(nmolec)
        mol_start = 0
        for mol_num in range(nmolec):
            mol_end = beads_per_mol[mol_num] + mol_start
            mol_mass[mol_num] = np.sum(mdata[mol_start : mol_end]) 
            mol_start = mol_end

    # Loop through each frame and bin particles. The center of mass determined
    # by the method string is used as a baseline in the binning process.
    
    for frame in range(frame_start, nframes):

        if centering == 'SYSTEM':
            xdata = xyz[frame][:, 0]
            ydata = xyz[frame][:, 1]
            zdata = xyz[frame][:, 2]
            mass_weighted_x = np.sum(xdata * mdata)
            mass_weighted_y = np.sum(ydata * mdata)
            mass_weighted_z = np.sum(zdata * mdata)
            total_mass = np.sum(mdata)
            offset[0] = mass_weighted_x / total_mass
            offset[1] = mass_weighted_y / total_mass
            offset[2] = mass_weighted_z / total_mass
        
        # The slab method for calculating the positional offset for the system.
        # Works by finding the largest cluster in the frame, and setting that
        # cluster's center of mass to be the positional offset. Clusters are 
        # determined by looking at the distance between the center of mass of 
        # molecules.

        if centering == 'SLAB':

            # Find clusters based on the center of mass of molecules.

            mol_x = mol_com[frame][:, 0]
            mol_y = mol_com[frame][:, 1]
            mol_z = mol_com[frame][:, 2]

            # Create a contact of the molecules, which details whether a
            # molecule is within the cluster cutoff (ccut) of other molecules.

            contact_map = np.zeros((nmolec, nmolec))
            for i in range(nmolec):                           
                i_mol_pos = np.array([mol_x[i], mol_y[i], mol_z[i]])                      
                for j in range(i, nmolec):                                                
                    if i == j:
                        contact_map[i][j] += 1
                        continue
                    j_mol_pos = np.array([mol_x[j], mol_y[j], mol_z[j]])                  
                    dr = i_mol_pos - j_mol_pos
                    distance = sqrt(np.dot(dr, dr))
                    if distance <= ccut:                        
                        contact_map[i][j] += 1
                        contact_map[j][i] += 1

            # Compress the contact map by combining rows with shared contacts.
            # This causes each row to represent a unique cluster.
            
            for row in range(contact_map.shape[0]):
                new_contacts = True
                while new_contacts:
                    new_contacts = False
                    for col in range(contact_map.shape[1]):
                        if row == col:  # Skip self-contacts.
                            continue
                        if (contact_map[row][col] != 0 and 
                            np.any(contact_map[col])):  # Non-zero row contact.
                            new_contacts = True
                            contact_map[row] += contact_map[col]
                            contact_map[col].fill(0)

            # From the compressed contact map, find the largest cluster and
            # set the positional offset to the cluster's center of mass.

            beads_in_cluster.fill(0)
            cluster_sizes = np.count_nonzero(contact_map, axis=1)
            largest_cluster_idx = np.argmax(cluster_sizes)
            mass_weighted_x = 0
            mass_weighted_y = 0
            mass_weighted_z = 0
            total_mass = 0
            for col in range(contact_map.shape[1]):
                if contact_map[largest_cluster_idx][col] != 0:
                    mass_weighted_x += mol_x[col] * mol_mass[col]
                    mass_weighted_y += mol_y[col] * mol_mass[col]
                    mass_weighted_z += mol_z[col] * mol_mass[col]
                    total_mass += mol_mass[col]
            offset[0] = mass_weighted_x / total_mass
            offset[1] = mass_weighted_y / total_mass
            offset[2] = mass_weighted_z / total_mass

        # For each particle of interest, apply the center of mass offset and 
        # bin it. Periodic boundary conditions are applied when applicable.

        for bead_num in range(xyz_sel.shape[1]):
                bead_pos = xyz_sel[frame][bead_num][slab_dim] -\
                           offset[slab_dim]
                if bead_pos >= slab_end:
                    bead_pos -= slab_length
                if bead_pos < slab_begin:
                    bead_pos += slab_length
                bin_idx = int(((bead_pos / slab_length + 1 / 2) * nbins))
                if bin_idx >= nbins or bin_idx < 0:
                    print('Error: invalid bin index created.')
                beads_per_bin[sec_num][bin_idx] += 1

        # If a section is reached, increment the section number.

        if (frame - frame_start + 1) % frames_per_sec == 0:
            sec_num += 1
    
    # Convert binned beads to concentration values and create density profiles.

    conc = beads_per_bin / frames_per_sec / bin_volume
    density_profiles = np.zeros((nsections, nbins, 2))
    for i in range(nsections):
        density_profiles[i] = np.column_stack((dimension, conc[i]))

    return density_profiles


# @jit(nopython=True)
# def bead_heatmap(xyz, bead_list, fixed_dim, rcut, nbins, box_dims, img_flags):
#     """
#     Create a 2D heatmap showing the relative density of nearby beads. The
#     resolution of the heatmap cover the rij vector (which connects bead 1 to
#     bead 2) to a maximum of rcut, having free rotation in every dimension but
#     the fixed dimension given by fixed_dim.
#     
#     Args:
#         xyz: the coordinates of a trajectory stored as a 3D numpy array, where
#              the dimensions are (in order) simulation frame, atom number, and
#              xyz coordinates.
#         bead_list: 1D numpy array with the index of every bead of interest.
#         fixed_dim: Dimension to not allow rij to rotate in. Either x, y, or z.
#                    In the rij vector, it sets that dimension's component to 0.
#         rcut: maximum length for the rij vector.
#         nbins: number of bins for a dimension of the rij vector.
#         box_dims: the xyz values of the simulation box stored as a 1D numpy
#                   array, where index 0 is the x-dimension, index 1 is the
#                   y-dimension, and index 2 is the z-dimension.
#         img_flags: the trajectory's image flags stored as a 3D numpy array ... 
#                    (finish and test documentation later).
# 
#     Returns:
#         fig_heatmap, ax_heatmap: matplotlib figure and axes of the heatmap
#     """
#     
#     # TODO Make an rij matrix that is rij_matrix[ri_value][rj_value].
#     # TODO Go through each rij in the simulation and bin into a bin matrix (of
#     #      shape equal to rij_matrix). Find closest point.
#     # TODO Generate a heatmap from the bin matrix.
