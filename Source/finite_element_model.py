# import numpy as np
import torch
# import torch.nn as nn
import yaml
import os
import enumerate_dof as ed

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)


# Load data nodes = np.loadtxt('nodes.txt', delimiter=',',dtype=np.float64)  # Each row contains X, Y, Z coordinates
# connectivity = np.loadtxt('connectivity.txt', delimiter=',', dtype=int) displacement = np.loadtxt(
# 'displacement_0.500000.txt', delimiter=',', dtype=np.float64)  # Adjust to zero-based indexing X = torch.tensor(
# nodes, dtype=torch.float64) connectivity = torch.tensor(connectivity, dtype=torch.float32)


# Shape function vectors for 3 DOF (Nx, Ny, Nz) at a given Gauss point

def shape_functions(Eps1, Eps2, Eps3):
    # Scalar shape functions for 8-node hexahedral element
    n1 = 1 / 8 * (1 + Eps1) * (1 + Eps2) * (1 + Eps3)
    n2 = 1 / 8 * (1 - Eps1) * (1 + Eps2) * (1 + Eps3)
    n3 = 1 / 8 * (1 + Eps1) * (1 + Eps2) * (1 - Eps3)
    n4 = 1 / 8 * (1 - Eps1) * (1 + Eps2) * (1 - Eps3)
    n5 = 1 / 8 * (1 + Eps1) * (1 - Eps2) * (1 + Eps3)
    n6 = 1 / 8 * (1 - Eps1) * (1 - Eps2) * (1 + Eps3)
    n7 = 1 / 8 * (1 + Eps1) * (1 - Eps2) * (1 - Eps3)
    n8 = 1 / 8 * (1 - Eps1) * (1 - Eps2) * (1 - Eps3)

    # Shape function vectors (Nx, Ny, Nz) for each of the 8 nodes
    return torch.tensor([
        [n1, 0, 0], [0, n1, 0], [0, 0, n1],
        [n2, 0, 0], [0, n2, 0], [0, 0, n2],
        [n3, 0, 0], [0, n3, 0], [0, 0, n3],
        [n4, 0, 0], [0, n4, 0], [0, 0, n4],
        [n5, 0, 0], [0, n5, 0], [0, 0, n5],
        [n6, 0, 0], [0, n6, 0], [0, 0, n6],
        [n7, 0, 0], [0, n7, 0], [0, 0, n7],
        [n8, 0, 0], [0, n8, 0], [0, 0, n8]
    ]).T


# Derivatives of shape functions with respect to (eps1, eps2, eps3)

def shape_function_derivatives(Eps1, Eps2, Eps3):
    # Compute derivatives with respect to eps1 (xi)
    dn_dxi = torch.tensor([
        1 / 8 * (1 + Eps2) * (1 + Eps3),  # n1
        -1 / 8 * (1 + Eps2) * (1 + Eps3),  # n2
        1 / 8 * (1 + Eps2) * (1 - Eps3),  # n3
        -1 / 8 * (1 + Eps2) * (1 - Eps3),  # n4
        1 / 8 * (1 - Eps2) * (1 + Eps3),  # n5
        -1 / 8 * (1 - Eps2) * (1 + Eps3),  # n6
        1 / 8 * (1 - Eps2) * (1 - Eps3),  # n7
        -1 / 8 * (1 - Eps2) * (1 - Eps3)  # n8
    ])

    # Compute derivatives with respect to eps2 (eta)
    dn_deta = torch.tensor([
        1 / 8 * (1 + Eps1) * (1 + Eps3),  # n1
        1 / 8 * (1 - Eps1) * (1 + Eps3),  # n2
        1 / 8 * (1 + Eps1) * (1 - Eps3),  # n3
        1 / 8 * (1 - Eps1) * (1 - Eps3),  # n4
        -1 / 8 * (1 + Eps1) * (1 + Eps3),  # n5
        -1 / 8 * (1 - Eps1) * (1 + Eps3),  # n6
        -1 / 8 * (1 + Eps1) * (1 - Eps3),  # n7
        -1 / 8 * (1 - Eps1) * (1 - Eps3)  # n8
    ])

    # Compute derivatives with respect to eps3 (zeta)
    dn_dzeta = torch.tensor([
        1 / 8 * (1 + Eps1) * (1 + Eps2),  # n1
        1 / 8 * (1 - Eps1) * (1 + Eps2),  # n2
        -1 / 8 * (1 + Eps1) * (1 + Eps2),  # n3
        -1 / 8 * (1 - Eps1) * (1 + Eps2),  # n4
        1 / 8 * (1 + Eps1) * (1 - Eps2),  # n5
        1 / 8 * (1 - Eps1) * (1 - Eps2),  # n6
        -1 / 8 * (1 + Eps1) * (1 - Eps2),  # n7
        -1 / 8 * (1 - Eps1) * (1 - Eps2)  # n8
    ])

    return torch.stack([dn_dxi, dn_deta, dn_dzeta], dim=0)



def X_position(Eps1, Eps2, Eps3, element_nodes):
    X = torch.zeros((1, ed.dof_per_node))
    N_shape_functions = shape_functions(Eps1, Eps2, Eps3)
    element_nodes_flattened = element_nodes.flatten().reshape(1, total_dof_per_element)
    # a = torch.tensor([[0,0.5,0.1]])
    for x in range(0, total_dof_per_element):
        X = X + N_shape_functions[:, x] * element_nodes_flattened[:, x]
        #  X = X + N_shape_functions[:, i] * a[:,i]
    return X



def Compute_energyNeoHooke(C, detF):
    # Get material parameters
    kappa = config["Material Properties"]["Bulk Modulus"]
    mu = config["Material Properties"]["Shear Modulus"]

    Energy = ((kappa / 2) * ((1 / 2) * (detF ** 2 - 1) - torch.log(detF)) + (1 / 2) * mu * (
            detF ** (-2 / 3) * torch.trace(C) - 3))
    return Energy



def get_element_coordinates(nodes, connectivity, el):
    return nodes[connectivity[el]]



def get_element_displacements(displacement, connectivity, el):
    return displacement[connectivity[el]]



def quadrature_points():
    a = torch.tensor(0.5773502692, dtype=torch.float64)
    N1 = torch.tensor([a, a, a], dtype=torch.float64)
    N2 = torch.tensor([-a, a, a], dtype=torch.float64)
    N3 = torch.tensor([a, a, -a], dtype=torch.float64)
    N4 = torch.tensor([-a, a, -a], dtype=torch.float64)
    N5 = torch.tensor([a, -a, a], dtype=torch.float64)
    N6 = torch.tensor([-a, -a, a], dtype=torch.float64)
    N7 = torch.tensor([a, -a, -a], dtype=torch.float64)
    N8 = torch.tensor([-a, -a, -a], dtype=torch.float64)
    Quadrature_point = torch.stack([N1, N2, N3, N4, N5, N6, N7, N8])
    Weights = torch.tensor([1.0], dtype=torch.float64)
    return Quadrature_point, Weights


def fourth_order_unit_tensor_function():
    # Create indices for the 4D tensor
    indices = torch.arange(3)
    # Use broadcasting to set the diagonal entries to 1
    fourth_order_identity_tensor = torch.zeros((3, 3, 3, 3))
    fourth_order_identity_tensor[indices[:, None], indices, indices[:, None], indices] = 1
    return fourth_order_identity_tensor


def Compute_secondPiolaStress(C, detF):
    # Get material parameters
    kappa = config["Material Properties"]["Bulk Modulus"]
    mu = config["Material Properties"]["Shear Modulus"]

    # Compute invers of right Cauchy-Green
    Cinv = torch.linalg.inv(C)
    
    # Compute volumetric stress
    stress_vol = (detF * kappa / 2 * ((detF ** 2 - 1) / detF)) * Cinv

    # Compute isochoric stress
    s_bar = mu * torch.eye(ed.dof_per_node, dtype = torch.float64)
    P = fourth_order_unit_tensor_function() - (1 / 3) * torch.einsum('ij,kl->ijkl', Cinv, C)
    a = detF ** (-2 / 3) * P
    stress_iso = torch.einsum('ijkl,kl->ij', a, s_bar)

    # Compute total stress
    stress = stress_vol + stress_iso

    return stress



def Calculate_internalForceVector(stress, defGrad, dN_dX, weights, detJ):
    # Compute the symmetric part of [F.T * dN_dX] for all dofs in a batch
    tmp1 = torch.einsum('ij,kjl->kil', defGrad.T, dN_dX)
    tmp1 = 0.5 * (tmp1 + tmp1.transpose(1, 2))

    # Double contraction with stress for all dofs
    tmp2 = torch.einsum('kij,ij->k', tmp1, stress)

    # Multiply with detJ and weights and store in cellRHS_
    cellRHS_ = tmp2 * detJ * weights 

    return cellRHS_



def Extract_freeDofs(force_vector):
    force_vector_free = torch.zeros((ed.dof_counter, 1))

    for dof in range(ed.dof_counter):
        if ed.is_dof_constrained[dof] == False:
            force_vector_free[dof] = force_vector[dof]

    return force_vector_free


def Calculate_totalLoss(total_energy, force):
    alpha = 1  #forward problem
    L_R = 8 * total_energy + alpha * torch.linalg.norm(force) ** 2  #multiplied with 8 bcs of symmetry
    return L_R


def Precalculate_shapeFunctionDetails(connectivity, nodes):
    # Get quadrature points and weights
    quadrature_point, weights = quadrature_points()

    nElements = connectivity.shape[0]
    nQuadraturePoints = 8
    nDofsPer = ed.dof_per_node

    # Initialize empty tensor
    # - Shape function gradients w.r.t X
    shape_grad = torch.zeros([nElements, nQuadraturePoints, total_dof_per_element, ed.dof_per_node, ed.dof_per_node], dtype = torch.float64)
    # - Determinant of the mapping
    detJ = torch.zeros([nElements, nQuadraturePoints], dtype = torch.float64)

    # Loop through elements
    for el in range(connectivity.shape[0]):
        # Extract element nodes
        element_nodes = get_element_coordinates(nodes, connectivity, el).flatten()

        # Loop through quadrature points
        for qp in range(8):
            eps1 = quadrature_point[qp, 0]
            eps2 = quadrature_point[qp, 1]
            eps3 = quadrature_point[qp, 2]

            # Compute shape function derivatives w.r.t ref. coords
            dN_deta = shape_function_derivatives(eps1, eps2, eps3)
            
            # Construct vector valued shape function derivatives
            dN_eta_vec = torch.zeros([total_dof_per_element, ed.dof_per_node, ed.dof_per_node], dtype = torch.float64)
            for node in range(ed.num_nodes_per_element):
                for dof in range(ed.dof_per_node):
                    tmp = torch.zeros([ed.dof_per_node, ed.dof_per_node], dtype = torch.float64)
                    tmp[dof,:] = dN_deta[:, node]
                    dN_eta_vec[node * ed.dof_per_node + dof,:] = tmp
            
            # Compute Jacobian matrix and its inverse
            J = torch.zeros([ed.dof_per_node, ed.dof_per_node], dtype = torch.float64)
            for dof in range(total_dof_per_element):
                J += dN_eta_vec[dof] * element_nodes[dof]
                
            J_inv = torch.linalg.inv(J)

            # Compute and store shape function gradients w.r.t material coords.
            for dof in range(total_dof_per_element):
                shape_grad[el, qp, dof] = dN_eta_vec[dof] @ J_inv
            
            # Store mapping
            detJ[el, qp] = torch.linalg.det(J)

    return shape_grad, detJ



def main(connectivity, nodes, displacement_all, dofMatrix, preCalc):
    # Get quadrature points and weights
    quadrature_point, weights = quadrature_points()
    
    # Total loss across all load steps
    loss = 0.0

    # Reshape to have displacement block for each load step
    nLoadSteps = config["General"]["Number Load Steps"]
    displacement_all = displacement_all.reshape(nLoadSteps, -1, 3)

    for time in range(nLoadSteps):
        # Extract displacement for the load step
        displacement = displacement_all[time, :, :]

        # delete
        base_dir = config["Directory"]["File_directory"]
        input_folder = os.path.join(base_dir, "Input")
        time_values = [f"{i * 0.05:.6f}" for i in range(1, 11)]
        file_name = f"{input_folder}\displacement_{time_values[time]}.txt"  # Construct the file name
        # print(file_name)
        displacement = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
        displacement = torch.from_numpy(displacement)
        displacement = displacement[:, 3:6]
        # end delete

        # Initialize empty cell and global right hand side vector
        rightHandSide = torch.zeros(ed.dof_counter)
        cellRHS = torch.zeros(24, dtype = torch.double)

        # Energy for this load step
        energy_function = 0.0

        # Alias for pre-calculated quantities related to mapping
        shape_grad_ = preCalc.shape_grad
        detJ_ = preCalc.detJ

        for el in range(connectivity.shape[0]):

            # Extract element displacements
            element_displacements = get_element_displacements(displacement, connectivity, el).flatten()
            element_displacements = element_displacements.to(dtype = torch.float64)

            # Extract local DoFs on this element and set cellRHS to zero
            element_dofs = dofMatrix[el, :]
            cellRHS = 0.0

            # Loop through quadrature points
            for qp in range(0, 8):
                eps1 = quadrature_point[qp, 0]
                eps2 = quadrature_point[qp, 1]
                eps3 = quadrature_point[qp, 2]

                # Extract shape function gradients and mapping
                shape_function_gradients = shape_grad_[el, qp, :, :] # (24, 3, 3)
                detJ = detJ_[el, qp] # (1,)
                
                # Compute dformation gradient and its determinant
                F = (torch.eye(ed.dof_per_node, dtype = torch.float64) +
                        torch.einsum('ijk, i -> jk', shape_function_gradients, element_displacements))
                detF = torch.linalg.det(F)

                # Compute right Cauchy Green tensor
                C = F.T @ F
                
                # Compute energy, Piola stress and local cellRHS contribution at this qp
                E = Compute_energyNeoHooke(C, detF)
                S = Compute_secondPiolaStress(C, detF)
                cellRHS += Calculate_internalForceVector(S, F, shape_function_gradients, weights, detJ)

                # Add contribution to global energy
                energy_function += weights * detJ * E
            
            # Distribute constraints from cellRHS to rightHandSide
            for dof_ctr in range(0, 24):
                rightHandSide[element_dofs[dof_ctr]] += cellRHS[dof_ctr]

        # Set the entries of rightHandSide where we have constrained dofs to zero
        rightHandSide_condensed = Extract_freeDofs(rightHandSide) 

        # Calculate overall loss and accumulate into total loss variable
        loss += Calculate_totalLoss(energy_function, rightHandSide_condensed)
    #print(loss)
    return loss


total_dof_per_element = ed.num_nodes_per_element * ed.dof_per_node