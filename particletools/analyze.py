#!/usr/bin/env python3

'''
Functions to analyze a particle based trajectory.
'''

# TODO Go through and change [x][y][z] notation to [x, y, z].
# TODO Go through and change bead -> particle.

import numpy as np
from math import sqrt
from numba import jit

@jit(nopython=True)
def img_flags_from_traj(traj_wrap, box_config):
    """
    Calculate the image flags of a wrapped particle-based trajectory. This 
    assumes that the dump frequency is sufficiently high such that beads never 
    travel more than half the length of a box dimension, otherwise image flags 
    may be incorrect. Molecules that are spread across a periodic boundary will
    have incorrect image flags, potentially introducing errors in other 
    calculations. Assumes that the box dimensions are constant in time.
    
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

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

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

    # TODO Expand to triclinic boxes.
    # TODO Expand to non-constant box dimensions, though not sure how image
    # flags would be packaged in that case.

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
        mol_idx = np.where(mols == mol)[0][0]
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

    # TODO Add selection array to allow filtering.

    # Get simulation parameters from the arguments and preallocate arrays.

    mols = np.unique(molid)
    rg = np.zeros(mols.shape[0])

    # Loop through each molecule, find the corresponding particle indices, and
    # then calculate the radius of gyration of the molecule.

    for mol in mols:
        indices = np.where(molid == mol)[0]
        mol_idx = np.where(mols == mol)[0][0]
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

    # TODO Add selection array to allow filtering.

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


# @jit(nopython=True)
def density_from_frame(pos, molid, mass, box_config, selection, bin_dim, nbins, 
                       centering='NONE', ccut=350):

    # def profile_density(xyz, bead_sel, beads_per_mol, nbins, nsections, ccut,
    #                     centering, box_dims, mdata):

    # TODO Build docstring.
    # TODO Expand to triclinic boxes.
    # TODO Possibly separate centering from density_from_frame, but then this
    # might cause density_from_traj to still have centering if it is to handle
    # the averaging of densities. For example, the slab centering applied to
    # the final average compared to each individual frame will not be equal.

    # Calculate the cross section along the bin dimension.

    cross_section = 1
    for dim in range(3):
        if dim == bin_dim:
            continue
        cross_section *= box_config[dim]

    # Create histogram bins along the bin dimension.

    bin_range = box_config[bin_dim]
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
                    contact_map[i][j] += 1  # Contact of a mol with itself.
                    continue
                mol_j_pos = mol_com[j]
                dr = mol_i_pos - mol_j_pos
                dist = sqrt(np.dot(dr, dr))
                if dist <= ccut:                        
                    contact_map[i][j] += 1
                    contact_map[j][i] += 1

        # Compress the contact map by combining rows with shared contacts. This
        # causes each row to represent a unique cluster.
        
        for row in range(contact_map.shape[0]):
            new_contacts = True
            while new_contacts:
                new_contacts = False
                for col in range(contact_map.shape[1]):
                    if row == col:  # Skip contacts of a mol with itself.
                        continue
                    if (contact_map[row][col] != 0 and 
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
            pos_i = pos_sel[i][bin_dim] - offset[bin_dim]
            if pos_i >= bin_hi:
                pos_i -= bin_range
            if pos_i < bin_lo:
                pos_i += bin_range
            bin_idx = int(((pos_i / bin_range + 1 / 2) * nbins))
            if bin_idx < 0 or bin_idx >= nbins:
                print('Error: invalid bin index created.')
            bin_val[bin_idx] += 1

    # Convert the binned particles to a density profile.

    bin_conc = bin_val / bin_vol
    density_profile = np.column_stack((bin_pos, bin_conc))

    # Return the density profile across the desired dimension of the box.

    return density_profile


# # @jit(nopython=True)
# def density_from_frame(pos, molid, mass, box_config, selection, bin_dim, nbins,
#                        nsections=1, centering='NONE', ccut=350):
# 
#     # def profile_density(xyz, bead_sel, beads_per_mol, nbins, nsections, ccut,
#     #                     centering, box_dims, mdata):
# 
#     # TODO Build docstring.
#     # TODO Have it use get_mol_com for centering.
#     # TODO Expand to triclinic boxes.
# 
#     # Filter the particle positions based on the selection array. 
#     
#     pos_sel = pos[selection, :]
# 
#     # Find the longest dimension of the box and calculate the cross sectional
#     # area along that dimension.
# 
#     lx, ly, lz = box_config[0:3]
#     longest_len = max((lx, ly, lz))
#     cross_section = 1
#     for i in range(3):
#         if box_config[i] == longest_len:
#             longest_dim = i
#         else:
#             cross_section *= box_config[i]
# 
#     # Create histogram bins along the longest dimension.
# 
#     bin_lo = longest_len / -2
#     bin_hi = longest_len / 2
#     bin_width = longest_len / nbins
#     bin_vol = bin_width * cross_section
#     bin_poss = np.arange(bin_lo, bin_hi, bin_width)
# 
#     # Calculate the first frame to start, the frames per section, and prepare
#     # arrays for data storage.
# 
#     nframes = xyz.shape[0]
#     nbeads = xyz.shape[1]
#     frames_per_sec = nframes // nsections
#     frame_start = nframes - nsections * frames_per_sec
#     sec_num = 0
#     beads_per_bin = np.zeros((nsections, nbins))
#     slab_molecules_per_frame = np.zeros(nsections * frames_per_sec)
#     offset = np.zeros(3)
# 
#     # Get the center of mass and mass of each molecule if slab centering is 
#     # used and preallocate memory for arrays.
#     
#     if centering == 'SLAB':
#         nmol = beads_per_mol.shape[0]
#         mol_com = np.zeros((nframes, nmol, 3))
#         largest_mol = np.amax(beads_per_mol)
#         mol_data = np.zeros((4, largest_mol))
#         for frame in range(nframes):
#             mol_start = 0
#             for mol_num in range(nmol):
#                 mol_end = beads_per_mol[mol_num] + mol_start 
#                 for bead_num in range(mol_start, mol_end):
#                     mol_data[0][bead_num - mol_start] = xyz[frame][bead_num][0]
#                     mol_data[1][bead_num - mol_start] = xyz[frame][bead_num][1]
#                     mol_data[2][bead_num - mol_start] = xyz[frame][bead_num][2]
#                     mol_data[3][bead_num - mol_start] = mdata[bead_num]
#                 mass_weighted_x = np.sum(mol_data[0] * mol_data[3])
#                 mass_weighted_y = np.sum(mol_data[1] * mol_data[3])
#                 mass_weighted_z = np.sum(mol_data[2] * mol_data[3])
#                 total_mass = np.sum(mol_data[3])
#                 mol_com[frame][mol_num][0] = mass_weighted_x / total_mass
#                 mol_com[frame][mol_num][1] = mass_weighted_y / total_mass
#                 mol_com[frame][mol_num][2] = mass_weighted_z / total_mass
#                 mol_start = mol_end
#                 mol_data.fill(0)
#         mol_mass = np.zeros(nmol)
#         beads_in_cluster = np.zeros(nmol)
#         mol_start = 0
#         for mol_num in range(nmol):
#             mol_end = beads_per_mol[mol_num] + mol_start
#             mol_mass[mol_num] = np.sum(mdata[mol_start : mol_end]) 
#             mol_start = mol_end
# 
#     # Loop through each frame and bin particles. The center of mass determined
#     # by the method string is used as a baseline in the binning process.
#     
#     for frame in range(frame_start, nframes):
# 
#         if centering == 'SYSTEM':
#             xdata = xyz[frame][:, 0]
#             ydata = xyz[frame][:, 1]
#             zdata = xyz[frame][:, 2]
#             mass_weighted_x = np.sum(xdata * mdata)
#             mass_weighted_y = np.sum(ydata * mdata)
#             mass_weighted_z = np.sum(zdata * mdata)
#             total_mass = np.sum(mdata)
#             offset[0] = mass_weighted_x / total_mass
#             offset[1] = mass_weighted_y / total_mass
#             offset[2] = mass_weighted_z / total_mass
#         
#         # The slab method for calculating the positional offset for the system.
#         # Works by finding the largest cluster in the frame, and setting that
#         # cluster's center of mass to be the positional offset. Clusters are 
#         # determined by looking at the distance between the center of mass of 
#         # molecules.
# 
#         if centering == 'SLAB':
# 
#             # Find clusters based on the center of mass of molecules.
# 
#             mol_x = mol_com[frame][:, 0]
#             mol_y = mol_com[frame][:, 1]
#             mol_z = mol_com[frame][:, 2]
# 
#             # Create a contact of the molecules, which details whether a
#             # molecule is within the cluster cutoff (ccut) of other molecules.
# 
#             contact_map = np.zeros((nmol, nmol))
#             for i in range(nmol):                           
#                 mol_i_pos = np.array([mol_x[i], mol_y[i], mol_z[i]])                      
#                 for j in range(i, nmol):                                                
#                     if i == j:
#                         contact_map[i][j] += 1
#                         continue
#                     mol_j_pos = np.array([mol_x[j], mol_y[j], mol_z[j]])                  
#                     dr = mol_i_pos - mol_j_pos
#                     distance = sqrt(np.dot(dr, dr))
#                     if distance <= ccut:                        
#                         contact_map[i][j] += 1
#                         contact_map[j][i] += 1
# 
#             # Compress the contact map by combining rows with shared contacts.
#             # This causes each row to represent a unique cluster.
#             
#             for row in range(contact_map.shape[0]):
#                 new_contacts = True
#                 while new_contacts:
#                     new_contacts = False
#                     for col in range(contact_map.shape[1]):
#                         if row == col:  # Skip self-contacts.
#                             continue
#                         if (contact_map[row][col] != 0 and 
#                             np.any(contact_map[col])):  # Non-zero row contact.
#                             new_contacts = True
#                             contact_map[row] += contact_map[col]
#                             contact_map[col].fill(0)
# 
#             # From the compressed contact map, find the largest cluster and
#             # set the positional offset to the cluster's center of mass.
# 
#             beads_in_cluster.fill(0)
#             cluster_sizes = np.count_nonzero(contact_map, axis=1)
#             largest_cluster_idx = np.argmax(cluster_sizes)
#             mass_weighted_x = 0
#             mass_weighted_y = 0
#             mass_weighted_z = 0
#             total_mass = 0
#             for col in range(contact_map.shape[1]):
#                 if contact_map[largest_cluster_idx][col] != 0:
#                     mass_weighted_x += mol_x[col] * mol_mass[col]
#                     mass_weighted_y += mol_y[col] * mol_mass[col]
#                     mass_weighted_z += mol_z[col] * mol_mass[col]
#                     total_mass += mol_mass[col]
#             offset[0] = mass_weighted_x / total_mass
#             offset[1] = mass_weighted_y / total_mass
#             offset[2] = mass_weighted_z / total_mass
# 
#         # For each particle of interest, apply the center of mass offset and 
#         # bin it. Periodic boundary conditions are applied when applicable.
# 
#         for bead_num in range(xyz_sel.shape[1]):
#                 bead_pos = xyz_sel[frame][bead_num][slab_dim] -\
#                            offset[slab_dim]
#                 if bead_pos >= slab_end:
#                     bead_pos -= slab_length
#                 if bead_pos < slab_begin:
#                     bead_pos += slab_length
#                 bin_idx = int(((bead_pos / slab_length + 1 / 2) * nbins))
#                 if bin_idx >= nbins or bin_idx < 0:
#                     print('Error: invalid bin index created.')
#                 beads_per_bin[sec_num][bin_idx] += 1
# 
#         # If a section is reached, increment the section ID.
# 
#         if (frame - frame_start + 1) % frames_per_sec == 0:
#             sec_num += 1
#     
#     # Convert binned beads to concentration values and create density profiles.
# 
#     conc = beads_per_bin / frames_per_sec / bin_vol
#     density_profiles = np.zeros((nsections, nbins, 2))
#     for i in range(nsections):
#         density_profiles[i] = np.column_stack((dimension, conc[i]))
# 
#     return density_profiles


# # @jit(nopython=True)
# # def bead_heatmap(xyz, bead_list, fixed_dim, rcut, nbins, box_dims, img_flags):
# #     """
# #     Create a 2D heatmap showing the relative density of nearby beads. The
# #     resolution of the heatmap cover the rij vector (which connects bead 1 to
# #     bead 2) to a maximum of rcut, having free rotation in every dimension but
# #     the fixed dimension given by fixed_dim.
# #     
# #     Args:
# #         xyz: the coordinates of a trajectory stored as a 3D numpy array, where
# #              the dimensions are (in order) simulation frame, atom ID, and
# #              xyz coordinates.
# #         bead_list: 1D numpy array with the index of every bead of interest.
# #         fixed_dim: Dimension to not allow rij to rotate in. Either x, y, or z.
# #                    In the rij vector, it sets that dimension's component to 0.
# #         rcut: maximum length for the rij vector.
# #         nbins: ID of bins for a dimension of the rij vector.
# #         box_dims: the xyz values of the simulation box stored as a 1D numpy
# #                   array, where index 0 is the x-dimension, index 1 is the
# #                   y-dimension, and index 2 is the z-dimension.
# #         img_flags: the trajectory's image flags stored as a 3D numpy array ... 
# #                    (finish and test documentation later).
# # 
# #     Returns:
# #         fig_heatmap, ax_heatmap: matplotlib figure and axes of the heatmap
# #     """
# #     
# #     # TODO Make an rij matrix that is rij_matrix[ri_value][rj_value].
# #     # TODO Go through each rij in the simulation and bin into a bin matrix (of
# #     #      shape equal to rij_matrix). Find closest point.
# #     # TODO Generate a heatmap from the bin matrix.
