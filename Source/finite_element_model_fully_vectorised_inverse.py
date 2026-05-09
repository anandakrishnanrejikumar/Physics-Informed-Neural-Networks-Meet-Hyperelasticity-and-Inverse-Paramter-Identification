import os
import torch
import yaml
import enumerate_dof as ed
from enumerate_dof import constraintInfo

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

nLoadSteps = config["General"]["Number Load Steps"]

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


def Compute_energyNeoHooke_batch_NN(C, detF, kappa, mu):

    # Energy calculation for each element in the batch
    if config["Energy"]["Method"] == 'Neo Hooke':
        Energy_NN = ((kappa / 2) * ((1 / 2) * (detF ** 2 - 1) - torch.log(detF)) +
                 (1 / 2) * mu * (detF ** (-2 / 3) * torch.einsum('beqii->beq', C) - 3))
    else:
        Energy_NN = 0
    return Energy_NN, kappa, mu

def get_element_coordinates(nodes, connectivity, el):
    return nodes[connectivity[el]]


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



def Extract_freeDofs_batch(force_vectors):
    # Create a mask for unconstrained DOFs
    # unconstrained_mask = torch.from_numpy(
    #     ~ed.is_dof_constrained)  # Negate the constrained DOFs to get unconstrained mask
    # # Use the mask to selectively copy the force vectors where DOFs are unconstrained
    # force_vector_free_batch = force_vectors * unconstrained_mask  # Element-wise multiplication

    # Activate when used with automatic differentiation to calculate force
    identity_matrix = torch.ones(ed.total_nodes, ed.dof_per_node)
    temp = identity_matrix - constraintInfo
    constraintInfo_flatten = temp.flatten()
    force_vector_free_batch = force_vectors * constraintInfo_flatten
    return force_vector_free_batch


def Extract_nodes_force_for_loss(connectivity, nodes):
    # Extracting nodes where y =0.5
    Required_nodes = []
    for i in range(0, ed.total_nodes):
        if nodes[i, 1] == 0.5:
            temp = i
            Required_nodes.append(temp)

    # Required_nodes = torch.tensor(Required_nodes)
    # element_index = torch.zeros(ed.num_elements, ed.num_nodes_per_element * ed.dof_per_node)
    # # setting value to 1 where ever its required for extracting required dof from dof
    # for j in range(Required_nodes.size(0)):
    #     for el in range(ed.num_elements):
    #         k = 0
    #         for q in range(ed.num_nodes_per_element):
    #             if connectivity[el, q] == Required_nodes[j]:
    #                 element_index[el, k + 1] = 1
    #                 k = k + 3
    #             else:
    #                 k = k + 3
    # dof = torch.from_numpy(dof)
    # extraction of required dof
    # dof = dof * element_index
    # dof_extraction_for_loss_calculation = dof.flatten()
    #
    # # removing unwanted dofs
    # dof_extraction_for_loss_calculation = dof_extraction_for_loss_calculation[dof_extraction_for_loss_calculation != 0]
    #
    # # removing repeatitions
    # dof_extraction_for_loss_calculation = torch.unique(dof_extraction_for_loss_calculation)
    # dof_extraction_for_loss_calculation=dof_extraction_for_loss_calculation.to(torch.int)
    return torch.tensor(Required_nodes)


def Calculate_totalLoss_batch(E_NN, force_NN, force, force_NN_condensed, U_NN, U, dof):
    alpha = config["Hyperparameters"]["Alpha"]
    N_u = ed.total_nodes * 10
    N_t = config["General"]["Number Load Steps"]
    dof = dof.to(torch.int64).view(45,1)
    # Extraction of required forces for calculating L_F
    f_NN = force_NN[:, dof,1]
    f = force
    U_NN = U_NN[:,:,:].view(nLoadSteps,-1,3)
    U = U[:,:,:].view(nLoadSteps,-1,3)
    # Summing of forces along the columns for respective load steps
    f_NN = torch.sum(f_NN, dim=1).view(10)

    # Compute the norm of the forces for each element in the batch
    force_norms_NN_condensed = torch.linalg.norm(force_NN_condensed, dim=1)**2  # Shape: (N, )
    # Loss from displacement data,exp vs NN

    L_U = 1*(1 / N_u) * (
            torch.sum(torch.linalg.norm(U - U_NN,dim=2)**2))
    L_F =  1*(1 / N_t) * (
            torch.sum(torch.linalg.norm(f - f_NN)**2))

    # Physics Loss
    L_R = (E_NN) + alpha* torch.sum(force_norms_NN_condensed, dim=0)  # Shape: (N, )

    # Total Loss
    T_L = config["Hyperparameters"]["Gamma"] * L_U +  + config["Hyperparameters"]["Delta"] * L_F + config["Hyperparameters"]["Beta"]*L_R
    return T_L, L_U,L_F,L_R


def Precalculate_shapeFunctionDetails(connectivity, nodes):
    # Get quadrature points and weights
    quadrature_point, weights = quadrature_points()

    nElements = connectivity.shape[0]
    nQuadraturePoints = 8
    nDofsPer = ed.dof_per_node

    # Initialize empty tensor
    # - Shape function gradients w.r.t X
    shape_grad = torch.zeros([nElements, nQuadraturePoints, total_dof_per_element, ed.dof_per_node, ed.dof_per_node],
                             dtype=torch.float64)
    # - Determinant of the mapping
    detJ = torch.zeros([nElements, nQuadraturePoints], dtype=torch.float64)

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
            dN_eta_vec = torch.zeros([total_dof_per_element, ed.dof_per_node, ed.dof_per_node], dtype=torch.float64)
            for node in range(ed.num_nodes_per_element):
                for dof in range(ed.dof_per_node):
                    tmp = torch.zeros([ed.dof_per_node, ed.dof_per_node], dtype=torch.float64)
                    tmp[dof, :] = dN_deta[:, node]
                    dN_eta_vec[node * ed.dof_per_node + dof, :] = tmp

            # Compute Jacobian matrix and its inverse
            J = torch.zeros([ed.dof_per_node, ed.dof_per_node], dtype=torch.float64)
            for dof in range(total_dof_per_element):
                J += dN_eta_vec[dof] * element_nodes[dof]

            J_inv = torch.linalg.inv(J)

            # Compute and store shape function gradients w.r.t material coords.
            for dof in range(total_dof_per_element):
                shape_grad[el, qp, dof] = dN_eta_vec[dof] @ J_inv

            # Store mapping
            detJ[el, qp] = torch.linalg.det(J)

    return shape_grad, detJ


def main(nEl,connectivity, displacement_all, preCalc, required_dof_for_loss_calculation, Y, nu,force_all_load_steps,Loss_displacement):
    # Get quadrature points and weights
    quadrature_point, weights = quadrature_points()

    # Reshape to have displacement block for each load step

    displacement = displacement_all.reshape(nLoadSteps, -1, 3)

    # Delete
    # time_values = [f"{i * 0.05:.6f}" for i in range(1, nLoadSteps + 1)]
    # displacement_prescribed = torch.zeros((10, ed.total_nodes, 3))
    # print(time_values)
    # for i in range(nLoadSteps):
    #     base_dir = config["Directory"]["File_directory"]
    #     input_folder = os.path.join(base_dir, "Input","Element512")
    #     file_name = os.path.join(input_folder, f"displacement_{time_values[i]}.txt")  # Construct the file name
    #     tmp = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
    #     tmp = torch.from_numpy(tmp)
    #     displacement_prescribed[i] = tmp[:, 3:6]
    # Loss_displacement = displacement_prescribed.view(-1, ed.dof_per_node)


    # Vectorized computation for all elements

    displacements_all_time_steps = displacement[:, connectivity[:], :].view(nLoadSteps, nEl, 24, 1).squeeze(-1)
    displacements_all_time_steps = displacements_all_time_steps.to(dtype=torch.float64)
    shape_function_gradients = preCalc.shape_grad  # Shape (64, 8, 24, 3, 3)

    # displacements_all_time_steps = displacement_prescribed #Test

    detJ2 = preCalc.detJ  # Shape (64, 8)

    # Compute deformation gradient across all load steps, elements, and qps
    F_NN = torch.eye(ed.dof_per_node, dtype=torch.double).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(nLoadSteps, nEl,
                                                                                                        8,
                                                                                                        1,
                                                                                                        1)  # (10, 8,3, 3)

    F_NN += torch.einsum('eqijk, bei -> beqjk', shape_function_gradients,
                         displacements_all_time_steps)


    # All determinants in one go
    detF_NN = torch.linalg.det(F_NN)  # (10, 64, 8)


    # C = torch.einsum('bqij, bqjk -> bqik', F.transpose(2, 3), F) # Shape (10, 8, 3, 3)
    F_NN_reshaped = F_NN.view(-1, 3, 3)

    C_reshaped_NN = torch.bmm(F_NN_reshaped.transpose(1, 2), F_NN_reshaped)

    C_NN = C_reshaped_NN.view(nLoadSteps, nEl, 8, 3, 3)  # Shape (10, 64, 8, 3, 3)

    # Compute energy and stress across all load steps and qps
    E_NN, Bulk_mod, shear_mod = Compute_energyNeoHooke_batch_NN(C_NN, detF_NN, Y, nu)  # Shape (10, 64, 8)


    # S_NN = Compute_secondPiolaStress_batch_NN(C_NN, detF_NN, Bulk_mod, shear_mod)  # Shape (10, 64, 8, 3, 3)

    # Assemble right hand side vectors
    # cellRHS_NN = Calculate_internalForceVector_batch(nEl,S_NN, F_NN, shape_function_gradients, weights,
    #                                                  detJ2)  # Shape (10, 64, 24)
    # Compute energy (Shape (10,))
    energy_function_NN = (weights * detJ2.view(1, nEl, 8) * E_NN).sum(dim=2).sum(dim=1).sum(dim=0)

    # Distribute cellRHS into global vector
    # See https://discuss.pytorch.org/t/how-to-add-at-specific-but-possibly-repeating-indices/85865
    # rightHandSide_NN = torch.zeros(nLoadSteps, ed.dof_counter, dtype=torch.double)
    # dofMatrix_flatten = torch.from_numpy(dofMatrix).flatten()
    # for i in range(nLoadSteps):
    #     rightHandSide_NN[i].put_(dofMatrix_flatten, cellRHS_NN[i].flatten(), accumulate=True)

    # Set rightHandSide at constrained dofs to zero across all loadsteps
    # rightHandSide_condensed_NN = Extract_freeDofs_batch(rightHandSide_NN)
    rightHandSide_NN = torch.autograd.grad(energy_function_NN, displacement, create_graph=True)[0]
    rightHandSide_condensed_NN = Extract_freeDofs_batch(rightHandSide_NN.view(nLoadSteps, -1))

    # Calculate overall loss
    loss, L_U,L_F,L_R = Calculate_totalLoss_batch(energy_function_NN, rightHandSide_NN, force_all_load_steps,
                                                    rightHandSide_condensed_NN,
                                                    displacement, Loss_displacement,
                                                    required_dof_for_loss_calculation)
    # print(loss.sum())
    return loss, Bulk_mod, shear_mod,L_U,L_F,L_R


total_dof_per_element = ed.num_nodes_per_element * ed.dof_per_node
