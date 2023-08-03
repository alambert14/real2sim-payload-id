import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MultibodyPlant, MultibodyPlant_
from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.multibody.tree import UnitInertia, UnitInertia_, SpatialInertia, SpatialInertia_, MultibodyForces_, MultibodyForces
from build_system import create_planar_manipulation_station, create_panda_manipulation_station


def create_autodiff_plant(plant, dim = 2, three_dee = False):
    """
    Returns an autodiff plant and variables

    Very redundant, could use global variables to make faster
    """

    if three_dee:
        N = dim * 10
    else:
        N = dim * 4

    plant_ad = plant.ToScalarType[AutoDiffXd]()
    context_ad = plant_ad.CreateDefaultContext()

    prefix = "link"
    if dim > 2:
        prefix = "panda_link"
    for i in range(dim):
        link = plant_ad.GetBodyByName(f"{prefix}{i + 1}")
        m_val = link.get_mass(context_ad).value()

        # ExtractValue takes the current value from the plant
        cx_val, cy_val, cz_val = ExtractValue(link.CalcCenterOfMassInBodyFrame(context_ad))
        spa = link.CalcSpatialInertiaInBodyFrame(context_ad)
        Ixx_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 0]
        Iyy_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 1]
        Izz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
        Ixy_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 1]
        Ixz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 2]
        Iyz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 2]

        if three_dee:
            m_vec = np.zeros(N)
            print(N)
            m_vec[(i * 10)] = 1  # Create autodiff unit vector
            m_ad = AutoDiffXd(m_val, m_vec)  # create autodiff variable

            cx_vec = np.zeros(N)
            cx_vec[(i * 10) + 1] = 1
            cx_ad_times_mass = AutoDiffXd(cx_val * m_val, cx_vec)  # lumped parameter, m*cm
            cx_ad = cx_ad_times_mass / m_ad
            cy_vec = np.zeros(N)
            cy_vec[(i * 10) + 2] = 1
            cy_ad_times_mass = AutoDiffXd(cy_val * m_val, cy_vec)  # lumped parameter, m*cm
            cy_ad = cy_ad_times_mass / m_ad
            cz_vec = np.zeros(N)
            cz_vec[(i * 10) + 3] = 1
            cz_ad_times_mass = AutoDiffXd(cz_val * m_val, cz_vec)
            cz_ad = cz_ad_times_mass / m_ad
            c_ad = [cx_ad, cy_ad, cz_ad]

            Ixx_vec = np.zeros(N)
            Ixx_vec[(i * 10) + 4] = 1
            Ixx_ad = AutoDiffXd(Ixx_val, Ixx_vec)
            Gxx_ad = Ixx_ad / m_ad
            Iyy_vec = np.zeros(N)
            Iyy_vec[(i * 10) + 5] = 1
            Iyy_ad = AutoDiffXd(Iyy_val, Iyy_vec)
            Gyy_ad = Iyy_ad / m_ad
            Izz_vec = np.zeros(N)
            Izz_vec[(i * 10) + 6] = 1
            Izz_ad = AutoDiffXd(Izz_val, Izz_vec)
            Gzz_ad = Izz_ad / m_ad
            Ixy_vec = np.zeros(N)
            Ixy_vec[(i * 10) + 7] = 1
            Ixy_ad = AutoDiffXd(Ixy_val, Ixy_vec)
            Gxy_ad = Ixy_ad / m_ad
            Ixz_vec = np.zeros(N)
            Ixz_vec[(i * 10) + 8] = 1
            Ixz_ad = AutoDiffXd(Ixz_val, Ixz_vec)
            Gxz_ad = Ixz_ad / m_ad
            Iyz_vec = np.zeros(N)
            Iyz_vec[(i * 10) + 9] = 1
            Iyz_ad = AutoDiffXd(Iyz_val, Iyz_vec)
            Gyz_ad = Iyz_ad / m_ad
            G_ad = UnitInertia_[AutoDiffXd](Gxx_ad, Gyy_ad, Gzz_ad, Gxy_ad, Gxz_ad, Gyz_ad)
        else:
            m_vec = np.zeros(N)
            m_vec[(i * 4)] = 1  # Create autodiff unit vector
            m_ad = AutoDiffXd(m_val, m_vec)  # create autodiff variable

            cx_vec = np.zeros(N)
            cx_vec[(i * 4) + 1] = 1
            cx_ad_times_mass = AutoDiffXd(cx_val * m_val, cx_vec)  # lumped parameter, m*cm
            cx_ad = cx_ad_times_mass / m_ad
            cz_vec = np.zeros(N)
            cz_vec[(i * 4) + 2] = 1
            cz_ad_times_mass = AutoDiffXd(cz_val * m_val, cz_vec)
            cz_ad = cz_ad_times_mass / m_ad
            c_ad = [cx_ad, cy_val[0], cz_ad]

            Iyy_vec = np.zeros(N)
            Iyy_vec[(i * 4) + 3] = 1
            Iyy_ad = AutoDiffXd(Iyy_val, Iyy_vec)
            Gyy_ad = Iyy_ad / m_ad
            G_ad = UnitInertia_[AutoDiffXd](Ixx_val / m_val, Gyy_ad, Izz_val / m_val)

        # We need to keep all of the other parameters consistent, otherwise spatial inertia fails
        spa_ad = SpatialInertia_[AutoDiffXd](
            m_ad, c_ad, G_ad, skip_validity_check=True,
        )
        link.SetSpatialInertiaInBodyFrame(context_ad, spa_ad)
        link.SetMass(context_ad, m_ad)
        link.SetCenterOfMassInBodyFrame(context_ad, c_ad)

    return plant_ad, context_ad


def create_plant_with_autodiff_object(plant, dim = 2, three_dee = False):
    """
    Returns an autodiff plant and variables

    Very redundant, could use global variables to make faster
    """

    if three_dee:
        N = 10
    else:
        N = 4

    plant_ad = plant.ToScalarType[AutoDiffXd]()
    context_ad = plant_ad.CreateDefaultContext()

    prefix = "link"
    if dim > 2:
        prefix = "panda_link"
    terminal_link = plant_ad.GetBodyByName(f"{prefix}{dim}")

    # Get existing values for the terminal link
    m_val = terminal_link.get_mass(context_ad).value()
    cx_val, cy_val, cz_val = ExtractValue(terminal_link.CalcCenterOfMassInBodyFrame(context_ad))
    spa = terminal_link.CalcSpatialInertiaInBodyFrame(context_ad)
    Ixx_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 0]
    Iyy_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 1]
    Izz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
    Ixy_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 1]
    Ixz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 2]
    Iyz_val = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 2]

    if three_dee:
        # Give a random initial position for the payload
        m_vec = np.zeros(N)
        m_vec[0] = 1  # Create autodiff unit vector
        m_ad = AutoDiffXd(0, m_vec)  # Initialize to 0
        terminal_link.SetMass(context_ad, m_val + m_ad)

        cx_vec = np.zeros(N)
        cx_vec[1] = 1
        cx_ad_times_mass = AutoDiffXd(0., cx_vec)  # Initialize to 0
        cx_ad = cx_ad_times_mass / m_ad
        cy_vec = np.zeros(N)
        cy_vec[2] = 1
        cy_ad_times_mass = AutoDiffXd(0., cy_vec)  # Initialize to 0
        cy_ad = cy_ad_times_mass / m_ad
        cz_vec = np.zeros(N)
        cz_vec[3] = 1
        cz_ad_times_mass = AutoDiffXd(0., cz_vec)  # Initialize to 0
        cz_ad = cz_ad_times_mass / m_ad
        new_c_ad = [((cx_val * m_val) + cx_ad_times_mass) / (m_val + m_ad),
                    ((cy_val * m_val) + cy_ad_times_mass) / (m_val + m_ad),
                    ((cz_val * m_val) + cz_ad_times_mass) / (m_val + m_ad)]

        Ixx_vec = np.zeros(N)
        Ixx_vec[4] = 1
        Ixx_ad = AutoDiffXd(0, Ixx_vec)  # initialize to zero (point mass)

        Iyy_vec = np.zeros(N)
        Iyy_vec[5] = 1
        Iyy_ad = AutoDiffXd(0, Iyy_vec)  # initialize to zero (point mass)

        Izz_vec = np.zeros(N)
        Izz_vec[6] = 1
        Izz_ad = AutoDiffXd(0, Izz_vec)  # initialize to zero (point mass)

        Ixy_vec = np.zeros(N)
        Ixy_vec[7] = 1
        Ixy_ad = AutoDiffXd(0, Ixy_vec)  # initialize to zero (point mass)

        Ixz_vec = np.zeros(N)
        Ixz_vec[8] = 1
        Ixz_ad = AutoDiffXd(0, Ixz_vec)  # initialize to zero (point mass)

        Iyz_vec = np.zeros(N)
        Iyz_vec[9] = 1
        Iyz_ad = AutoDiffXd(0, Iyz_vec)  # initialize to zero (point mass)

        new_G_ad = UnitInertia_[AutoDiffXd](
            (Ixx_val + Ixx_ad) / (m_val + m_ad),
            (Iyy_val + Iyy_ad) / (m_val + m_ad),
            (Izz_val + Izz_ad) / (m_val + m_ad),
            (Ixy_val + Ixy_ad) / (m_val + m_ad),
            (Ixz_val + Ixz_ad) / (m_val + m_ad),
            (Iyz_val + Iyz_ad) / (m_val + m_ad),
        )
    else:
        m_vec = np.zeros(N)
        m_vec[0] = 1  # Create autodiff unit vector
        m_ad = AutoDiffXd(0, m_vec)  # Initialize to 0
        terminal_link.SetMass(context_ad, m_val + m_ad)

        cx_vec = np.zeros(N)
        cx_vec[1] = 1
        cx_ad_times_mass = AutoDiffXd(0., cx_vec)  # Initialize to 0
        cx_ad = cx_ad_times_mass / m_ad
        cz_vec = np.zeros(N)
        cz_vec[2] = 1
        cz_ad_times_mass = AutoDiffXd(0., cz_vec)  # Initialize to 0
        cz_ad = cz_ad_times_mass / m_ad
        new_c_ad = [((cx_val * m_val) + cx_ad_times_mass) / (m_val + m_ad),
                    cy_val,
                    ((cz_val * m_val) + cz_ad_times_mass) / (m_val + m_ad)]

        Iyy_vec = np.zeros(N)
        Iyy_vec[3] = 1
        Iyy_ad = AutoDiffXd(0, Iyy_vec)  # initialize to zero (point mass)
        new_G_ad = UnitInertia_[AutoDiffXd](
            Ixx_val / m_val, (Iyy_val + Iyy_ad) / (m_val + m_ad), Izz_val / m_val)

    spa_ad = SpatialInertia_[AutoDiffXd](
        m_val + m_ad, new_c_ad, new_G_ad, skip_validity_check=False,
    )
    terminal_link.SetSpatialInertiaInBodyFrame(context_ad, spa_ad)
    terminal_link.SetMass(context_ad, m_val + m_ad)
    terminal_link.SetCenterOfMassInBodyFrame(context_ad, new_c_ad)

    return plant_ad, context_ad


def evaluate_model_on_data(plant, context, data, dim=2):
    """
    compute modeled errors of a plant against the measure torque

    Uses stacked trajectories and loops through all of them to accumulate the offset
    """
    T = data['T']
    q_log = data['q_meas']
    v_log = data['qdot_meas']
    v_dot = data['qddot_est']
    tau_log = data['tau_meas']

    tau_ad = np.zeros((T, dim), dtype="object")
    if isinstance(plant, MultibodyPlant_[AutoDiffXd]):
        forces = MultibodyForces_[AutoDiffXd](plant)
    else:
        forces = MultibodyForces(plant)
    for t in range(T):
        plant.SetPositions(context, q_log[t, :])
        plant.SetVelocities(context, v_log[t, :])
        plant.get_actuation_input_port().FixValue(context, tau_log[t, :])

        plant.CalcForceElementsContribution(context, forces)
        tau_ad[t, :] = plant.CalcInverseDynamics(context, v_dot[t, :], forces)

        # Confirm that I get the same back out with forward dynamics
        # plant.get_actuation_input_port().FixValue(context, tau_ad[t, :])
        # if plant.time_step() == 0:
        #     v_dot_again = plant.EvalTimeDerivatives(context).CopyToVector()
        #     np.testing.assert_almost_equal(ExtractValue(v_dot[t, :]), ExtractValue(v_dot_again[-dim:]))

    return tau_ad


def compute_errors(plant_ad, context_ad, data, dim=2):
    """
    compute modeled errors against the measure torque

    Uses stacked trajectories and loops through all of them to accumulate the offset
    """
    tau_ad = evaluate_model_on_data(plant_ad, context_ad, data, dim=dim)

    e_vector = tau_ad - data['tau_meas']
    E = np.trace(np.dot(e_vector.T, e_vector))  # squared error
    print("the baseline RMSE error value is ", (E.value() / e_vector.shape[0]) ** 0.5)  # root squared error
    # print("the baseline mean squared error is ", E.value() / e_vector.shape[0])

    return e_vector


def plot_model_errors(model_plant, context_plant, data, dim=2, plot_index=0):
    tau_model = evaluate_model_on_data(model_plant, context_plant, data, dim=dim)

    ts = np.linspace(0, 11*10, len(data['ts']))  # make sure this is still correct

    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plots = plt.plot(ts, data['tau_meas'])
    plt.legend(plots, ['joint ' + str(i) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Nm")
    plt.title("Measured torques")

    plt.subplot(1, 2, 2)
    plots = plt.plot(ts, ExtractValue(tau_model))
    plt.legend(plots, ['joint ' + str(i) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Nm")
    plt.title("Estimated torques")

    plt.show()


def least_square_regression(e_vector, T, dim=2, three_dee=False):
    """
    use the jacobian (error with respect to arm variables) from the autodiff gradient, solve least square and get delta_x
    """
    # if three_dee:
    #     N = dim * 10
    # else:
    #     N = dim * 4
    e_r = e_vector.reshape(T * dim)  # reshape with column first
    print('er[0]:  ', e_r[0])
    S = np.vstack([y.derivatives().reshape(1, -1) for y in e_r])  # is this correct?

    print(f'Condition number of S: {np.linalg.cond(S)}')
    e_0 = ExtractValue(e_r)
    sol = np.linalg.lstsq(-S, ExtractValue(e_r), rcond=None)
    delta_0 = sol[0]
    new_error = e_0 + S @ delta_0
    E = np.trace(np.dot(new_error.T, new_error))
    print("the new OLS RMSE error value is ", (E / new_error.shape[0]) ** 0.5)
    return delta_0, new_error


def ridge_regression(e_vector, T, plot=False, base_factor=6):
    """
    use the jacobian (error with respect to arm variables) from the autodiff gradient, solve damped least square and get delta_x
    only adding regularizations to inertia of 7 links (less on the last link)
    """
    num_params = 8
    I = list(range(num_params))

    e_r = e_vector.T.reshape(-1)  # reshape with column first
    S = np.vstack([y.derivatives().reshape(1, num_params) for y in e_r])
    # lots of zeros in S, the friction parameters and the first link, so parameters 10:-21
    print("Condition of S: ", np.linalg.cond(S))  # Currently infinite. Bad

    e_0 = ExtractValue(e_r)
    W = np.eye(num_params)

    # we want to regularize the last link, link7, less than others.
    base_ratio = 5
    for i in I:
        W[i, i] *= base_ratio ** (base_factor - 2 * (i > 60))

    for i in range(70, full_arm_dim):
        W[i, i] *= base_ratio
    # print(np.diag(W))

    sol = np.linalg.lstsq(np.dot(S.T, S) + 1 * W, -np.dot(S.T, e_0), rcond=None)

    print("Condition after Regularization: ", np.linalg.cond(np.dot(S.T, S) + W))

    delta_ridge = sol[0]
    new_error = ExtractValue(e_r) + S @ delta_ridge
    print("damped least square error is ", (np.dot(new_error.T, new_error) ** 0.5)[0, 0])

    if plot:
        delta_0, _ = least_square_regression(e_vector, T)
        plt.figure(figsize=(15, 24))
        plt.plot(delta_0[I], "o", label="LS")
        plt.plot(delta_ridge[I], "o", label="ridge")
        plt.title("Comparing regularized vs non-regularized solution")
        plt.legend()

    print("new error: ", np.linalg.norm(new_error))
    print("delta ridge: ", np.linalg.norm(delta_ridge))

    return delta_ridge, new_error


def apply_model(delta, dim=2, three_dee=False):
    """
    apply the calibrated delta_x, and get a new plant
    """
    # Test with zero delta
    # delta = np.zeros((8, 1))

    # apply the identified model delt to the robot
    if dim > 2:
        plant, _, _, _, _, _ = create_panda_manipulation_station(None)
    else:
        plant, _, _, _, _, _ = create_planar_manipulation_station(None, dim=dim)
    context = plant.CreateDefaultContext()

    original_params = []

    if three_dee:
        N = dim * 10
    else:
        N = dim * 4

    prefix = "link"
    if dim > 2:
        prefix = "panda_link"
    for i in range(dim):
        link = plant.GetBodyByName(f"{prefix}{i + 1}")
        m = link.get_mass(context)
        cx, cy, cz = link.CalcCenterOfMassInBodyFrame(context)
        spa = link.CalcSpatialInertiaInBodyFrame(context)
        Ixx = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 0]
        Iyy = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 1]
        Izz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
        Ixy = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
        Ixz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
        Iyz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]


        if three_dee:
            original_params.append(m)
            m_new = m + delta[(i * 10), 0]
            original_params.append(cx * m)
            cx_new = ((cx * m) + delta[(i * 10) + 1, 0]) / m_new
            original_params.append(cy * m)
            cy_new = ((cy * m) + delta[(i * 10) + 2, 0]) / m_new
            original_params.append(cz * m)
            cz_new = ((cz * m) + delta[(i * 10) + 3, 0]) / m_new
            original_params.append(Ixx)
            Ixx_new = Ixx + delta[(i * 10) + 4, 0]
            original_params.append(Iyy)
            Iyy_new = Iyy + delta[(i * 10) + 5, 0]
            original_params.append(Izz)
            Izz_new = Izz + delta[(i * 10) + 6, 0]
            original_params.append(Ixy)
            Ixy_new = Ixy + delta[(i * 10) + 7, 0]
            original_params.append(Ixz)
            Iyz_new = Iyz + delta[(i * 10) + 8, 0]
            original_params.append(Izz)
            Iyz_new = Iyz + delta[(i * 10) + 9, 0]

            c_new = [cx_new, cy_new, cz_new]
            G_new = UnitInertia(Ixx_new / m_new, Iyy_new / m_new, Izz_new / m_new)
        else:
            original_params.append(m)
            m_new = m + delta[(i * 4), 0]
            original_params.append(cx * m)
            cx_new = ((cx * m) + delta[(i * 4) + 1, 0]) / m_new
            original_params.append(cz * m)
            cz_new = ((cz * m) + delta[(i * 4) + 2, 0]) / m_new
            original_params.append(Iyy)
            Iyy_new = Iyy + delta[(i * 4) + 3, 0]

            c_new = [cx_new, cy, cz_new]
            G_new = UnitInertia(Ixx / m_new, Iyy_new / m_new, Izz / m_new)

        spa_new = SpatialInertia(
            m_new, c_new, G_new, skip_validity_check=True, # Change this back
        )
        link.SetSpatialInertiaInBodyFrame(context, spa_new)
        link.SetMass(context, m_new)
        link.SetCenterOfMassInBodyFrame(context, c_new)

    # Print the resulting parameters after the SpatialInertia
    # Shows that they are consistent
    # print("Link1 mass (calculated from delta, in spa): ", m1_new, link1.get_mass(context))
    # print("Link1 CoM (calculated from delta, in spa): ",
    #       [cx1_new, cy1, cz1_new],
    #       link1.CalcCenterOfMassInBodyFrame(context))
    # print("Link1 Inertia (calculated from delta, in spa): ",
    #       Iyy1_new,
    #       link1.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3, :3])

    original_params = np.array(original_params)
    print(original_params)
    print('Original Parameters and estimated parameters: ', original_params, original_params + delta[:, 0])

    return plant, context


def apply_model_object(delta, dim=2, three_dee=False):
    """
    apply the calibrated delta_x, and get a new plant
    modifies old plant...
    """
    # Test with zero delta
    # delta = np.zeros((8, 1))

    # apply the identified model delt to the robot
    plant, _, _, _, _, _ = create_planar_manipulation_station(None, dim=dim)
    context = plant.CreateDefaultContext()

    original_params = []

    if three_dee:
        N = 10
    else:
        N = 4

    prefix = "link"
    if dim > 2:
        prefix = "panda_link"

    terminal_link = plant.GetBodyByName(f"{prefix}{dim}")
    m = terminal_link.get_mass(context)
    cx, cy, cz = terminal_link.CalcCenterOfMassInBodyFrame(context)
    spa = terminal_link.CalcSpatialInertiaInBodyFrame(context)
    Ixx = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[0, 0]
    Iyy = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[1, 1]
    Izz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
    Ixy = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
    Ixz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]
    Iyz = ExtractValue(spa.CopyToFullMatrix6()[:3, :3])[2, 2]

    if three_dee:
        m_obj = delta[0, 0]
        m_new = m + m_obj

        cx_times_mass_obj = delta[1, 0]
        cx_new = ((cx * m) + cx_times_mass_obj) / m_new
        cy_times_mass_obj = delta[2, 0]
        cy_new = ((cy * m) + cy_times_mass_obj) / m_new
        cz_times_mass_obj = delta[3, 0]
        cz_new = ((cz * m) + cz_times_mass_obj) / m_new

        Ixx_obj = delta[4, 0]
        Ixx_new = Ixx + Ixx_obj
        Iyy_obj = delta[5, 0]
        Iyy_new = Iyy + Iyy_obj
        Izz_obj = delta[6, 0]
        Izz_new = Izz + Izz_obj
        Ixy_obj = delta[7, 0]
        Ixy_new = Ixy + Ixy_obj
        Ixz_obj = delta[8, 0]
        Ixz_new = Ixz + Ixz_obj
        Iyz_obj = delta[9, 0]
        Iyz_new = Iyz + Iyz_obj

        c_new = [cx_new, cy_new, cz_new]
        G_new = UnitInertia(Ixx_new / m_new, Iyy_new / m_new, Izz_new / m_new)
    else:
        m_obj = delta[0, 0]
        m_new = m + m_obj

        cx_times_mass_obj = delta[1, 0]
        cx_new = ((cx * m) + cx_times_mass_obj) / m_new
        cz_times_mass_obj = delta[2, 0]
        cz_new = ((cz * m) + cz_times_mass_obj) / m_new

        Iyy_obj = delta[3, 0]
        Iyy_new = Iyy + Iyy_obj

        c_new = [cx_new, cy, cz_new]
        G_new = UnitInertia(Ixx / m, Iyy_new / m_new, Izz / m)

    spa_new = SpatialInertia(
        m_new, c_new, G_new, skip_validity_check=True, # Change this back
    )
    terminal_link.SetSpatialInertiaInBodyFrame(context, spa_new)
    terminal_link.SetMass(context, m_new)
    terminal_link.SetCenterOfMassInBodyFrame(context, c_new)

    # Print the resulting parameters after the SpatialInertia
    # Shows that they are consistent
    # print("Link1 mass (calculated from delta, in spa): ", m1_new, link1.get_mass(context))
    # print("Link1 CoM (calculated from delta, in spa): ",
    #       [cx1_new, cy1, cz1_new],
    #       link1.CalcCenterOfMassInBodyFrame(context))
    # print("Link1 Inertia (calculated from delta, in spa): ",
    #       Iyy1_new,
    #       link1.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3, :3])

    return plant, context
