import numpy as np
import matplotlib.pyplot as plt
import pydrake.symbolic as sym
from pydrake.all import (
    Parser, AddMultibodyPlantSceneGraph, SpatialInertia_, RotationalInertia_, DiagramBuilder,
    FindResourceOrThrow, ToLatex
)
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.multibody.tree import UnitInertia_, MultibodyForces_, UnitInertia, SpatialInertia, MultibodyForces
from tqdm import tqdm


def create_symbolic_plant(plant, dim=2, three_dee=False, with_object=False):
    """
    :param plant:
    :param with_object: If specified, also returns the object parameters
    :return:
    """
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()

    # State variables
    q = sym.MakeVectorVariable(dim, "q")
    v = sym.MakeVectorVariable(dim, "v")
    vd = sym.MakeVectorVariable(dim, "\dot{v}")
    tau = sym.MakeVectorVariable(dim, "u")
    state_variables = [q, v, vd, tau]

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    # Parameters
    parameters = []
    for i in range(dim):
        if three_dee:
            m = sym.Variable(f"m_{i + 1}")
            cx = sym.Variable(f"c_{{x_{i + 1}}}")
            cy = sym.Variable(f"c_{{y_{i + 1}}}")
            cz = sym.Variable(f"c_{{z_{i + 1}}}")
            Gxx = sym.Variable(f"G_{{xx_{i + 1}}}")
            Gyy = sym.Variable(f"G_{{yy_{i + 1}}}")
            Gzz = sym.Variable(f"G_{{zz_{i + 1}}}")
            Gxy = sym.Variable(f"G_{{xy_{i + 1}}}")
            Gxz = sym.Variable(f"G_{{xz_{i + 1}}}")
            Gyz = sym.Variable(f"G_{{yz_{i + 1}}}")
            parameters += [m, cx, cy, cz, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz]
        else:
            m = sym.Variable(f"m_{i + 1}")
            cx = sym.Variable(f"c_{{x_{i + 1}}}")
            cz = sym.Variable(f"c_{{z_{i + 1}}}")
            Gyy = sym.Variable(f"G_{{yy_{i + 1}}}")
            parameters += [m, cx, cz, Gyy]

        if dim > 2:
            link = sym_plant.GetBodyByName(f"panda_link{i + 1}")
        else:
            link = sym_plant.GetBodyByName(f"link{i + 1}")
        link.SetMass(sym_context, m)
        if three_dee:
            link.SetCenterOfMassInBodyFrame(sym_context, [cx, cy, cz])
            new_unit = UnitInertia_[sym.Expression](Gxx, Gyy, Gzz, Gxy, Gxz, Gyz)
            inertia = SpatialInertia_[sym.Expression](m, [cx, cy, cz], new_unit, skip_validity_check=False)
        else:
            current_spa = link.CalcSpatialInertiaInBodyFrame(sym_context)
            new_cm = current_spa.get_com()
            new_cm[0] = cx
            new_cm[2] = cz
            new_unit = current_spa.get_unit_inertia().get_moments()
            new_unit[1] = Gyy
            inertia = SpatialInertia_[sym.Expression](
                m, new_cm, UnitInertia_[sym.Expression](new_unit[0], new_unit[1], new_unit[2]),
                skip_validity_check=False)

        link.SetSpatialInertiaInBodyFrame(sym_context, inertia)

    if with_object:

        if three_dee:
            mp = sym.Variable("m_p")
            cxp = sym.Variable("c_{x_p}")
            cyp = sym.Variable("c_{y_p}")
            czp = sym.Variable("c_{z_p}")
            Gxxp = sym.Variable("G_{xx_p}")
            Gyyp = sym.Variable("G_{yy_p}")
            Gzzp = sym.Variable("G_{zz_p}")
            Gxyp = sym.Variable("G_{xy_p}")
            Gxzp = sym.Variable("G_{xz_p}")
            Gyzp = sym.Variable("G_{yz_p}")
            obj_parameters = [mp, cxp, cyp, czp, Gxxp, Gyyp, Gzzp, Gxyp, Gxzp, Gyzp]
        else:
            mp = sym.Variable("m_p")
            cxp = sym.Variable("c_{x_p}")
            czp = sym.Variable("c_{z_p}")
            Gyyp = sym.Variable("G_{yy_p}")
            obj_parameters = [mp, cxp, czp, Gyyp]

        if dim > 2:
            obj = sym_plant.GetBodyByName('base_link_meat')
        else:
            obj = sym_plant.GetBodyByName('payload')

        # Just get the ground truth values
        print('object mass: ', obj.get_mass(sym_context))
        print('object mass*cm: ', obj.get_mass(sym_context) * obj.CalcCenterOfMassInBodyFrame(sym_context))
        test_I = obj.CalcSpatialInertiaInBodyFrame(sym_context).CalcRotationalInertia()
        print('object inertia: ', test_I.CopyToFullMatrix3())

        obj.SetMass(sym_context, mp)
        if three_dee:
            obj.SetCenterOfMassInBodyFrame(sym_context, [cxp, cyp, czp])
            new_unit = UnitInertia_[sym.Expression](Gxxp, Gyyp, Gzzp, Gxyp, Gxzp, Gyzp)
            obj_inertia = SpatialInertia_[sym.Expression](mp, [cxp, cyp, czp], new_unit, skip_validity_check=False)
        else:
            current_spa = obj.CalcSpatialInertiaInBodyFrame(sym_context)
            new_cm = current_spa.get_com()
            new_cm[0] = cxp
            new_cm[2] = czp
            obj.SetCenterOfMassInBodyFrame(sym_context, new_cm)
            new_unit = current_spa.get_unit_inertia().get_moments()
            new_unit[1] = Gyyp
            obj_inertia = SpatialInertia_[sym.Expression](
                mp, new_cm, UnitInertia_[sym.Expression](new_unit[0], new_unit[1], new_unit[2]),
                skip_validity_check=False)
        obj.SetSpatialInertiaInBodyFrame(sym_context, obj_inertia)

    if with_object:
        return sym_plant, sym_context, (parameters, obj_parameters), state_variables

    return sym_plant, sym_context, parameters, state_variables


def create_plant_with_symbolic_object(estimated_plant, dim=2, three_dee=False, with_object=False):
    print('running create symbolic plant')  # Add this print statement unlocks the error message in this fn?
    sym_plant = estimated_plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()

    # State variables
    q = sym.MakeVectorVariable(dim, "q")
    v = sym.MakeVectorVariable(dim, "v")
    vd = sym.MakeVectorVariable(dim, "\dot{v}")
    tau = sym.MakeVectorVariable(dim, "u")
    state_variables = [q, v, vd, tau]

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    # Parameters
    parameters = []

    prefix = 'panda_link' if dim > 2 else 'link'
    terminal_link = sym_plant.GetBodyByName(f'{prefix}{dim}')

    # Get existing estimates
    m_N = terminal_link.get_mass(sym_context)
    cx_N, cy_N, cz_N = terminal_link.CalcCenterOfMassInBodyFrame(sym_context)
    spa_N = terminal_link.CalcSpatialInertiaInBodyFrame(sym_context)
    Ixx_N = spa_N.CopyToFullMatrix6()[:3, :3][0, 0]
    Iyy_N = spa_N.CopyToFullMatrix6()[:3, :3][1, 1]
    Izz_N = spa_N.CopyToFullMatrix6()[:3, :3][2, 2]
    Ixy_N = spa_N.CopyToFullMatrix6()[:3, :3][0, 1]
    Ixz_N = spa_N.CopyToFullMatrix6()[:3, :3][0, 2]
    Iyz_N = spa_N.CopyToFullMatrix6()[:3, :3][1, 2]

    if three_dee:
        mp = sym.Variable("m_p")
        cxp = sym.Variable("c_{x_p}")
        cyp = sym.Variable("c_{y_p}")
        czp = sym.Variable("c_{z_p}")
        Gxxp = sym.Variable("G_{xx_p}")
        Gyyp = sym.Variable("G_{yy_p}")
        Gzzp = sym.Variable("G_{zz_p}")
        Gxyp = sym.Variable("G_{xy_p}")
        Gxzp = sym.Variable("G_{xz_p}")
        Gyzp = sym.Variable("G_{yz_p}")
        obj_parameters = [mp, cxp, cyp, czp, Gxxp, Gyyp, Gzzp, Gxyp, Gxzp, Gyzp]

        new_m = m_N + mp
        new_c = [(mp * cxp + m_N * cx_N) / new_m,
                 (mp * cyp + m_N * cy_N) / new_m,
                 (mp * czp + m_N * cz_N) / new_m]
        G_new = UnitInertia_[sym.Expression](Gxxp, Gyyp, Gzzp, Gxyp, Gxzp, Gyzp)
    else:
        mp = sym.Variable("m_p")
        cxp = sym.Variable("c_{x_p}")
        czp = sym.Variable("c_{z_p}")
        Gyyp = sym.Variable("G_{yy_p}")
        obj_parameters = [mp, cxp, czp, Gyyp]

        new_m = m_N + mp
        new_c = [(mp * cxp + m_N * cx_N) / new_m,
                 cy_N,
                 (mp * czp + m_N * cz_N) / new_m]
        new_G = UnitInertia_[sym.Expression](Ixx_N / m_N, Gyyp, Izz_N / m_N)
        print('stuck completing symbolic plant')

    terminal_link.SetMass(sym_context, m_N + mp),

    new_spa = SpatialInertia_[sym.Expression](
        new_m, new_c, new_G,
        skip_validity_check=False)
    terminal_link.SetCenterOfMassInBodyFrame(sym_context, new_c)
    terminal_link.SetSpatialInertiaInBodyFrame(sym_context, new_spa)

    print('finished creating symbolic plant')

    return sym_plant, sym_context, obj_parameters, state_variables


def calculate_decomposition_with_object(sym_plant, sym_context, parameters, obj_parameters, state_variables):
    # THIS ALL WORKS JUST FINE YIPPEE
    q, v, vd, tau = state_variables

    forces = MultibodyForces_[sym.Expression](sym_plant)
    sym_plant.CalcForceElementsContribution(sym_context, forces)
    print('got force elements')
    sym_torques = sym_plant.CalcInverseDynamics(sym_context, vd, forces)
    print('calculated inverse dynamics')
    W, alpha, w0 = sym.DecomposeLumpedParameters(sym_torques, parameters)
    print('decomposed lumped params')

    W_p, alpha_p, w0_p = sym.DecomposeLumpedParameters(w0, obj_parameters)
    print(alpha_p)

    return W_p, alpha_p, w0_p


def calculate_decomposition_just_object(sym_plant, sym_context, obj_parameters, state_values):
    new_context = sym_context.Clone()
    sym_plant.SetPositions(new_context, state_values['q'])
    sym_plant.SetVelocities(new_context, state_values['qdot'])

    forces = MultibodyForces_[sym.Expression](sym_plant)
    sym_plant.CalcForceElementsContribution(new_context, forces)
    sym_torques = sym_plant.CalcInverseDynamics(new_context, state_values['qddot'], forces)

    W_p, alpha_p, w0_p = sym.DecomposeLumpedParameters(sym_torques, obj_parameters)
    if not alpha_p[0].EqualTo(sym.Expression(obj_parameters[0])):
        print("NOT MASS!", alpha_p[0])  # This never happens

    return W_p, alpha_p, w0_p


def calculate_decomposition_known_states(sym_plant, sym_context, parameters, obj_parameters, state_vars, state_values):
    """
    At a given time T, set the state of the plant to be a known value and calculate the lumped parameter decomposition
    :param sym_plant:
    :param sym_context:
    :param parameters:
    :param obj_parameters:
    :param state_values:
    :return:
    """
    new_context = sym_context.Clone()
    sym_plant.SetPositions(new_context, state_values['q'])
    sym_plant.SetVelocities(new_context, state_values['qdot'])

    print(state_values)


    forces = MultibodyForces_[sym.Expression](sym_plant)
    sym_plant.CalcForceElementsContribution(new_context, forces)
    sym_torques = sym_plant.CalcInverseDynamics(new_context, state_values['qddot'], forces)

    W, alpha, w0 = sym.DecomposeLumpedParameters(sym_torques, parameters)
    if obj_parameters:
        W_p, alpha_p, w0_p = sym.DecomposeLumpedParameters(w0, obj_parameters)
        if not alpha_p[0].EqualTo(sym.Expression(obj_parameters[0])):
            print("NOT MASS!", alpha_p[0])  # This never happens

        return W_p, alpha_p, w0_p
    return W, alpha, w0

def calculate_lumped_parameters(sym_plant, sym_context, parameters, state_variables):
    # derivatives = sym_context.Clone().get_mutable_continuous_state()
    q, v, vd, tau = state_variables
    forces = MultibodyForces_[sym.Expression](sym_plant)
    sym_plant.CalcForceElementsContribution(sym_context, forces)
    sym_torques = sym_plant.CalcInverseDynamics(sym_context, vd.T, forces)
    # derivatives.SetFromVector(np.hstack((0*v, vd)))
    # print(type(sym_plant), type(sym_context), type(derivatives))
    # residual = sym_plant.CalcImplicitTimeDerivativesResidual(
    #     sym_context, derivatives)

    # print("Residual: ", residual)
    W, alpha, w0 = sym.DecomposeLumpedParameters(sym_torques, parameters)
    print("Lumped parameters: ", alpha)

    return W, alpha, w0


def remove_nonidentifiable(W, alpha, w0):
    """
    Remove the parameters that are nonidentifiable
    """
    W_new = np.copy(W)
    w0_new = np.copy(w0)
    alpha_new = np.copy(alpha)
    print(w0_new.shape)
    Q, R = np.linalg.qr(W)
    R_diag = np.diagonal(R)
    tol = 1e-9  # rows * eps * np.max(R_diag)

    to_delete = [i for i, r in enumerate(R_diag) if abs(r) < tol]
    print(R_diag)
    print(alpha[to_delete])

    W_new[:, to_delete] = 0
    # W_new = np.delete(W_new, to_delete, axis=1)
    # w0_new = np.delete(w0_new, to_delete, axis=0)
    # alpha_new = np.delete(alpha_new, to_delete, axis=0)

    return W_new, alpha_new, w0_new

def prune_w_SVD(W, alpha):
    """
    I have literally no basis for this to work but shruggies
    Nope this is complete garbage, go read a linear algebra book dumbass
    :param W:
    :param alpha:
    :return:
    """
    S_diag = np.linalg.svd(W, compute_uv=False)
    print(S_diag)
    tol = 1e-9
    to_delete = [i for i, s in enumerate(S_diag) if abs(s) < tol]

    U, S_diag, Vh = np.linalg.svd(W, full_matrices=False)

    print(Vh)
    # S = np.zeros(W.shape[0], W.shape)
    # S[:W.shape[1], :W.shape[1]] = np.diag(S_diag)
    S = np.diag(S_diag)
    U_new = np.delete(U, to_delete, axis=1)
    U_new = np.delete(U_new, to_delete, axis=0)
    Vh_new = np.delete(Vh, to_delete, axis=0)
    Vh_new = np.delete(Vh_new, to_delete, axis=1)
    S_new = np.delete(S, to_delete, axis=0)
    S_new = np.delete(S_new, to_delete, axis=1)
    alpha_new = np.delete(alpha, to_delete, axis=0)

    print('U', U_new.shape)
    print('S', S_new.shape)
    print('V', Vh_new.shape)
    inner = np.dot(S_new, Vh_new)
    W_new = np.dot(U_new, inner)

    print(np.linalg.svd(W_new, compute_uv=False))

    return W_new, alpha_new


def calc_data_matrix(sym_plant, sym_context, data, parameters, state_variables, dim=2):
    t = data['ts']
    q_log = data['q_meas']
    v_log = data['qdot_meas']
    vd_log = data['qddot_est']
    tau_log = data['tau_meas']

    W_sym, alpha_sym, w0_sym = calculate_lumped_parameters(sym_plant, sym_context, parameters, state_variables)
    print('data matrix size: ', W_sym.shape)

    q_sym, v_sym, vd_sym, tau_sym = state_variables

    M = t.shape[0] - 1
    MM = dim * M  # This might just be 2xM, check if extra zeros
    N = alpha_sym.size  # number of parameters
    Wdata = np.zeros((MM, N))
    w0data = np.zeros((MM, 1))
    tau_data = np.zeros((MM, 1))
    offset = 0
    for i in tqdm(range(M)):
        env = {}
        for q in range(dim):
            env[q_sym[q]] = q_log[i, q]
            env[v_sym[q]] = v_log[i, q]
            env[vd_sym[q]] = vd_log[i, q]
            env[tau_sym[q]] = tau_log[i, q]

        Wdata[offset:offset + dim, :] = sym.Evaluate(W_sym, env)
        w0data[offset:offset + dim] = sym.Evaluate(w0_sym, env)
        tau_data[offset:offset + dim] = sym.Evaluate(tau_sym, env)

        offset += dim

    print(Wdata)
    print('Condition of Wdata: ', np.linalg.cond(Wdata))
    # Wdata, alpha_sym, w0data = remove_nonidentifiable(Wdata, alpha_sym, w0data)
    # Wdata = np.delete(Wdata, 4, axis=1)
    # alpha_sym = np.delete(alpha_sym, 4, axis=0)
    # Doing this leads to worse results for the other values
    # Wdata, alpha_sym = prune_w_SVD(Wdata, alpha_sym)
    print("singular values: ", np.linalg.svd(Wdata, full_matrices=False)[1])

    alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]
    print('New condition of Wdata: ', np.linalg.cond(Wdata))

    # 2D case
    print(parameters)
    if dim == 2:
        gt_parameters = {
            parameters[0]: 0.6,
            parameters[4]: 0.8,
            parameters[1]: 0.,
            parameters[5]: 0.,
            parameters[2]: -0.5,
            parameters[6]: -0.75,
            parameters[3]: (0.05 + 0.6 * (-0.5) ** 2) / 0.6,
            parameters[7]: (0.15 + 0.8 * (-0.75) ** 2) / 0.8,  # adjust inertia to be not around center of mass
        }
    else:
        # 1D case
        gt_parameters = {
            parameters[0]: 0.6,
            parameters[1]: 0.,
            parameters[2]: -0.5,
            parameters[3]: (0.05 + 0.6 * (-0.5) ** 2) / 0.6,
        }
    alpha_gt = sym.Evaluate(alpha_sym, gt_parameters)

    return alpha_fit, alpha_gt, alpha_sym


def calc_data_matrix_7dof(sym_plant,
                          sym_context,
                          data,
                          robot_params,
                          state_variables,
                          dim=7):

    t = data['ts'] # Expected to be the same as with_data
    q_log = data['q_meas']
    v_log = data['qdot_meas']
    vd_log = data['qddot_est']
    tau = data['tau_meas']

    q_sym, v_sym, vd_sym, tau_sym = state_variables

    M = t.shape[0] - 1
    MM = dim * M
    N = 59 # len(robot_params)  # number of lumped parameters (happens to coincide with num paramters)
    Wdata = np.zeros((MM, N))
    tau_data = np.zeros((MM, 1))
    offset = 0

    mass_values = []
    for i in tqdm(range(M)):
        state_values = {
            'q': q_log[i],
            'qdot': v_log[i],
            'qddot': vd_log[i],
            'tau': tau[i],
        }

        env_tau = {}
        for j in range(dim):
            env_tau[tau_sym[j]] = tau[i, j]
        W, alpha, w0 = calculate_decomposition_known_states(sym_plant, sym_context, robot_params, None, state_variables, state_values)
        try:
            Wdata[offset:offset + dim, :] = sym.Evaluate(W, {})
        except ValueError as e:
            print(e)
            print(alpha)
            continue  # just skipping works okay, its fine if Wdata has zeros at the end
        tau_data[offset:offset + dim] = sym.Evaluate(tau_sym, env_tau)
        offset += dim
        alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]
        mass_values.append(alpha_fit[0])

    print('Condition of Wdata: ', np.linalg.cond(Wdata))  # this is okay, according to QR decomp
    # _, _, _ = remove_nonidentifiable(Wdata, alpha, tau_data)  # this is okay
    alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]

    plt.plot(mass_values)
    plt.show()
    return alpha_fit, alpha

# def evaluate_torques(W_sym)
#
#     expected_torques = []
#     est_torques = []
#     for t in range



def calc_data_matrix_with_object(sym_plant,
                                 sym_context,
                                 without_data,
                                 with_data,
                                 robot_params,
                                 object_params,
                                 state_variables,
                                 dim=2):

    W_sym, alpha_sym, w0_sym = calculate_decomposition_with_object(
        sym_plant, sym_context, robot_params, object_params, state_variables)
    # w0 should be zero

    t = without_data['ts'] # Expected to be the same as with_data
    q_log = without_data['q_meas']
    v_log = without_data['qdot_meas']
    vd_log = without_data['qddot_est']
    # tau_residual = with_data['tau_meas'] - without_data['tau_meas'] # this gives the same result
    tau_with = with_data['tau_meas']
    tau_without = without_data['tau_meas']

    q_sym, v_sym, vd_sym, tau_sym = state_variables

    M = t.shape[0] - 1
    MM = dim * M
    N = alpha_sym.size  # number of parameters
    Wdata = np.zeros((MM, N))
    tau_data = np.zeros((MM, 1))
    offset = 0
    for i in tqdm(range(M)):
        env = {}
        for j in range(dim):
            env[q_sym[j]] = q_log[i, j]
            env[v_sym[j]] = v_log[i, j]
            env[vd_sym[j]] = vd_log[i, j]
            env[tau_sym[j]] = tau_with[i, j] - tau_without[i, j]

        Wdata[offset:offset + dim, :] = sym.Evaluate(W_sym, env)
        tau_data[offset:offset + dim] = sym.Evaluate(tau_sym, env)
        offset += dim

    print('Condition of Wdata: ', np.linalg.cond(Wdata))
    # Wdata, alpha_sym, w0data = remove_nonidentifiable(Wdata, alpha_sym, w0data)
    # print("w0: ", w0data)
    # print(tau_residual.shape)
    alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]  # works :)

    return alpha_fit, alpha_sym


def calc_data_matrix_w_obj_7dof(sym_plant,
                                sym_context,
                                without_data,
                                with_data,
                                robot_params,
                                object_params,
                                state_variables,
                                dim=7,
                                real_data=False,
                                W=None,
                                sdp=False):
    # if without_data['ts'].shape[0] > with_data['ts'].shape[0]:
    #     offset = without_data['ts'].shape[0] - with_data['ts'].shape[0]
    #     t = with_data['ts']
    #     q_log = with_data['q_meas']
    #     v_log = with_data['qdot_meas']
    #     vd_log = with_data['qddot_est']
    #     tau_residual = with_data['tau_meas'] - without_data['tau_meas'][:-offset]
    # else:
    #     offset = with_data['ts'].shape[0] - without_data['ts'].shape[0]
    #     t = without_data['ts']
    #     q_log = without_data['q_meas']
    #     v_log = without_data['qdot_meas']
    #     vd_log = without_data['qddot_est']
    #     tau_residual = with_data['tau_meas'][:-offset] - without_data['tau_meas']

    t = with_data['ts']
    q_log = with_data['q_meas']
    v_log = with_data['qdot_meas']

    if real_data:
        vd_log = with_data['qddot_filtered']
        tau_residual = with_data['tau_filtered'] - without_data['tau_filtered']
    else:
        vd_log = with_data['qddot_est']
        tau_residual = with_data['tau_meas'] - without_data['tau_meas']
    q_sym, v_sym, vd_sym, tau_sym = state_variables

    M = t.shape[0] - 1
    MM = dim * M
    N = len(object_params)  # number of lumped parameters (happens to coincide with num paramters)
    Wdata = np.zeros((MM, N))
    tau_data = np.zeros((MM, 1))
    offset = 0

    mass_values = []
    for i in tqdm(range(M)):
        state_values = {
            'q': q_log[i],
            'qdot': v_log[i],
            'qddot': vd_log[i],
            'tau_diff': tau_residual[i],
        }

        env_tau = {}
        for j in range(dim):
            env_tau[tau_sym[j]] = tau_residual[i, j]
        W, alpha_p, w0 = calculate_decomposition_known_states(sym_plant, sym_context, robot_params, object_params, state_variables, state_values)
        try:
            Wdata[offset:offset + dim, :] = sym.Evaluate(W, {})
        except ValueError as e:
            print(alpha_p)
            continue  # just skipping works okay, its fine if Wdata has zeros at the end
        tau_data[offset:offset + dim] = sym.Evaluate(tau_sym, env_tau)
        offset += dim

        if sdp:
            mp = MathematicalProgram()

            alpha_mp = mp.NewContinuousVariables(10, "$\\alpha$")
            I_mp = mp.NewSymmetricContinuousVariables(3)
            mp.AddConstraint(I_mp[0, 0] == alpha_mp[4])
            mp.AddConstraint(I_mp[1, 1] == alpha_mp[5])
            mp.AddConstraint(I_mp[2, 2] == alpha_mp[6])
            mp.AddConstraint(I_mp[0, 1] == alpha_mp[7])
            mp.AddConstraint(I_mp[0, 2] == alpha_mp[8])
            mp.AddConstraint(I_mp[1, 2] == alpha_mp[9])

            posdef = (np.trace(I_mp)) / 2 * np.ones((3,3))
            mp.AddPositiveSemidefiniteConstraint(posdef)

            mp.AddLinearConstraint(alpha_mp[0] >= 0.)

            # mp.AddCost(np.linalg.norm(Wdata.dot(alpha_mp) - tau_data))
            Q = 2 * Wdata.T @ Wdata
            b = -2 * tau_data.T @ Wdata
            c = tau_data.T @ tau_data
            print(Q.shape, b.shape, c.shape)
            mp.AddQuadraticCost(Q, b, c, alpha_mp, is_convex=True)

            result = Solve(mp)
            alpha_fit = result.get_solution_result()
        else:
            alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]

        mass_values.append(alpha_fit[0])


    # np.save('W_data_5-5_sim_pick-and-place.npy', Wdata)
    print('Condition of Wdata: ', np.linalg.cond(Wdata))  # this is okay, according to QR decomp
    cond = np.linalg.cond(Wdata)
    #_, _, _ = remove_nonidentifiable(Wdata, alpha_p, tau_data)  # this is okay
    alpha_fit = np.linalg.lstsq(Wdata, tau_data, rcond=None)[0]
    # np.save('alphafit_5-5_sim_pick-and-place.npy', alpha_fit)
    np.save('mass_values_sinusoidal_5-8.npy', mass_values)

    plt.plot(mass_values)
    plt.show()
    return alpha_fit, alpha_p, cond

def calc_data_matrix_just_obj(sym_plant,
                              sym_context,
                              data,
                              object_params,
                              state_variables,
                              dim=7):

    t = data['ts']
    q_log = data['q_meas']
    v_log = data['qdot_meas']
    vd_log = data['qddot_est']
    tau = data['tau_meas']

    q_sym, v_sym, vd_sym, tau_sym = state_variables

    M = t.shape[0] - 1
    MM = dim * M
    N = len(object_params)  # number of lumped parameters (happens to coincide with num paramters)
    Wdata = np.zeros((MM, N))
    w0_data = np.zeros((MM, 1))
    offset = 0

    mass_values = []
    for i in tqdm(range(M)):
        state_values = {
            'q': q_log[i],
            'qdot': v_log[i],
            'qddot': vd_log[i],
            'tau': tau[i],
        }

        env_tau = {}
        for j in range(dim):
            env_tau[tau_sym[j]] = tau
        W, alpha_p, w0 = calculate_decomposition_just_object(sym_plant, sym_context, object_params, state_values)
        try:
            Wdata[offset:offset + dim, :] = sym.Evaluate(W, {})
        except ValueError as e:
            print('alpha: ', alpha_p)  # currently giving an extra wack decomposition
            continue  # just skipping works okay, its fine if Wdata has zeros at the end
        w0_data[offset:offset + dim] = sym.Evaluate(tau_sym, env_tau) - w0
        offset += dim
        alpha_fit = np.linalg.lstsq(Wdata, w0_data, rcond=None)[0]
        mass_values.append(alpha_fit[0])

    print('Condition of Wdata: ', np.linalg.cond(Wdata))  # this is okay, according to QR decomp
    # _, _, _ = remove_nonidentifiable(Wdata, alpha_p, w0_data)  # this is okay
    alpha_fit = np.linalg.lstsq(Wdata, w0_data, rcond=None)[0]

    plt.plot(mass_values)
    plt.show()
    return alpha_fit, alpha_p


def evaluate_new_plant_with_object(with_plant, alpha_fit, with_data, dim=2, three_dee=False):
    """
    Careful, mutates with_plant
    :param with_plant:
    :param alpha_fit:
    :param with_data:
    :return:
    """
    context = with_plant.CreateDefaultContext()
    if dim > 2:
        obj = with_plant.GetBodyByName('base_link_meat')
    else:
        obj = with_plant.GetBodyByName('payload')

    if three_dee:
        # Find out why these are out of order
        # Probably just the order they are instantiated in
        # Or just make it a dictionary instead
        m = alpha_fit[0]
        cx = alpha_fit[1] / m
        cy = alpha_fit[2] / m
        cz = alpha_fit[3] / m
        Gxx = alpha_fit[4] / m
        Gyy = alpha_fit[5] / m
        Gzz = alpha_fit[6] / m
        Gxy = alpha_fit[7] / m
        Gxz = alpha_fit[8] / m
        Gyz = alpha_fit[9] / m

        new_cm = [cx, cy, cz]
        G = UnitInertia(Gxx, Gyy, Gzz, Gxy, Gxz, Gyz)
    else:
        m = alpha_fit[0]
        cx = alpha_fit[1] / m
        cz = alpha_fit[2] / m
        Gyy = alpha_fit[3] / m

        current_spa_obj = obj.CalcSpatialInertiaInBodyFrame(context)
        new_cm = current_spa_obj.get_com()
        new_cm[0] = cx
        new_cm[2] = cz
        new_unit = current_spa_obj.get_unit_inertia().get_moments()
        new_unit[1] = Gyy
        G = UnitInertia(new_unit[0], new_unit[1], new_unit[2])

    spa = SpatialInertia(
        m, new_cm, G, skip_validity_check=False,  # should be false
    )
    obj.SetCenterOfMassInBodyFrame(context, new_cm)
    obj.SetSpatialInertiaInBodyFrame(context, spa)
    obj.SetMass(context, m)

    T = with_data['T']
    q_log = with_data['q_meas']
    v_log = with_data['qdot_meas']
    v_dot = with_data['qddot_est']
    tau_log = with_data['tau_meas']

    tau_est = np.zeros((T, dim), dtype="object")

    for t in range(T):
        with_plant.SetPositions(context, q_log[t, :])
        with_plant.SetVelocities(context, v_log[t, :])
        with_plant.get_actuation_input_port().FixValue(context, tau_log[t, :])
        forces = MultibodyForces(with_plant)
        with_plant.CalcForceElementsContribution(context, forces)
        tau_est[t, :] = with_plant.CalcInverseDynamics(context, v_dot[t, :], forces)

    print(tau_est)
    return tau_est


def plot_model_errors(with_plant, alpha_fit, with_data, dim=2, three_dee=False):
    tau_model = evaluate_new_plant_with_object(with_plant, alpha_fit, with_data, dim, three_dee)

    ts = np.linspace(0, with_data['ts'][-1], len(with_data['ts']))  # make sure this is still correct

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plots = plt.plot(ts, with_data['tau_meas'])
    plt.legend(plots, ['joint ' + str(i+1) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Nm")
    plt.title("Measured torques")

    plt.subplot(2, 2, 2)
    plots = plt.plot(ts, tau_model)
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Nm")
    plt.title("Estimated torques")

    plt.subplot(2, 2, 3)
    plots = plt.plot(ts, with_data['tau_meas'] - tau_model)
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Nm")
    plt.title("New residual torques")

    plt.show()