import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pydot
from IPython.display import SVG, display, Math
from pydrake.all import (AddMultibodyPlantSceneGraph,
                         DiagramBuilder,
                         Parser, PiecewisePolynomial, RigidTransform,
                         Simulator, StartMeshcat, MeshcatVisualizer)

# from manipulation.meshcat_cpp_utils import StartMeshcat
# from manipulation.scenarios import MakeManipulationStation
# from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.tree import SpatialInertia, UnitInertia

import symbolic_sysid_utils as sym_utils
import autodiff_sysid_utils as ad_utils
from build_system import create_planar_manipulation_station, create_panda_manipulation_station, add_point_to_plant, \
    PickAndPlaceTrajectorySource
from physical_feasibility_plots import create_mesh_grid
from sigproc import load_and_parse_file, plot_warping, align_sequences, load_multiple_residuals_files, match_sequences

import sys
sys.path.append('src/calibration/system_id_utils')
from src.calibration.system_id_utils import load_multiple_files


def generate_sinusoidal_traj(base_idx, dim=2, base_position=None):
    ts = np.arange(0, 10, 0.01)
    # frequency of base_idx * 0.5
    joints = []
    for i in range(dim):
        if dim == 7:
            amp = 0.3
        else:
            amp = 1
        if base_position:
            base_joint = base_position[i]
        else:
            base_joint = np.pi
        joints.append(base_joint + amp * np.sin((base_idx + i) * 0.5 * ts))
        # joints.append(base_joint + amp * np.sin(np.random.uniform(1, 5) * ts))
    joints = np.array(joints).reshape((dim, -1))
    return PiecewisePolynomial.CubicShapePreserving(
        ts, joints, zero_end_point_derivatives=True)


def display_logs(logs, simulator: Simulator, total_time) -> None:
    """

    :param logs: (Dict[str, Logger]) available keys - 'state', 'torque'
    :param simulator:
    :param total_time:
    :return: None
    """
    state_log = logs['state'].FindLog(simulator.get_context())
    torque_log = logs['torque'].FindLog(simulator.get_context())

    ts = np.linspace(0, total_time, state_log.data().shape[1])

    plt.figure(figsize=(15, 20))
    plt.subplot(2, 2, 1)
    plots = plt.plot(ts, state_log.data()[:2, :].T)
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(2)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("radians")
    plt.title("Joint positions")

    plt.subplot(2, 2, 2)
    plots = plt.plot(ts, state_log.data()[2:, :].T)
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(2)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("radians/s")
    plt.title("Joint velocities")

    plt.subplot(2, 2, 3)
    plots = plt.plot(ts, torque_log.data()[:, :].T)
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(2)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Joint torques")

    plt.show()


def plot_trajectory_difference(without_data, with_data, total_time, dim=2) -> None:
    """

    :param logs: (Dict[str, Logger]) available keys - 'state', 'torque'
    :param simulator:
    :param total_time:
    :return: None
    """
    q_wout = without_data['q_meas']
    q_with = with_data['q_meas']

    print(q_wout)
    print(q_with)

    qdot_wout = without_data['qdot_meas']
    qdot_with = with_data['qdot_meas']

    qddot_wout = without_data['qddot_est']
    if 'qddot_filtered' in without_data:
        qddot_filtered = without_data['qddot_filtered']

    # tau_label = 'tau_filtered'
    tau_label = 'tau_meas'
    torque_diff = with_data[tau_label] - without_data[tau_label]
    # if qdot_wout.shape != qdot_with.shape:
    #     # offset = q_wout.shape[0] - q_with.shape[0]
    #     offset = q_with.shape[0] - q_wout.shape[0]
    #     nans = np.empty((offset, 7))
    #     nans[:] = np.nan
    #     # q_with = np.concatenate((q_with, nans))
    #     q_wout = np.concatenate((q_wout, nans))
    #     #qdot_with = np.concatenate((qdot_with, nans))
    #     qdot_wout = np.concatenate((qdot_wout, nans))
    #     # torque_with = np.concatenate((with_data['tau_meas'], nans))
    #     torque_wout = np.concatenate((without_data['tau_meas'], nans))
    #     torque_with = with_data['tau_meas']
    #     torque_diff = torque_with - torque_wout # ithout_data['tau_meas']# [:-offset]
    # else:
    #     torque_diff = with_data['tau_meas'] - without_data['tau_meas']

    ts = np.linspace(0, total_time, q_with.shape[0])

    # if q_wout.shape != q_with.shape:
    #     raise ValueError("Trajectories are not the same length")

    plt.style.use('src/calibration/2dof/test.mplstyle')

    plt.figure(figsize=(30, 15))
    plt.subplot(2, 3, 1)

    muted_rainbow1 = ["#806BFF", "#5dace1", "#6ddad4", "#4bb652", "#e9c348", "#e99d48", "#e96248"]
    muted_rainbow2 = ["#e53935", "#ffb74d", "#ffee58", "#388e5a", "#42a5f5", "#8756d5", "#ea9bd7"]
    C = muted_rainbow1
    for i in range(dim):
        plt.plot(ts, q_wout[:, i], color=C[i], linewidth=2, label="joint " + str(i + 1), alpha=0.5)
        plt.plot(ts, q_with[:, i], '--', linewidth=2, color=C[i])
    plt.legend(loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("radians")
    plt.title("Joint positions")

    plt.subplot(2, 3, 2)
    for i in range(dim):
        plt.plot(ts, qdot_wout[:, i], color=C[i], label="joint " + str(i + 1), alpha=0.5)
        plt.plot(ts, qdot_with[:, i], '--', color=C[i])
    plt.legend(loc=1)
    # plt.legend(plots, ['joint ' + str(i + 1) for i in range(2)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("radians/s")
    plt.title("Joint velocities")

    plt.subplot(2, 3, 3)
    for i in range(dim):
        plt.plot(ts, qddot_wout[:, i], color=C[i], label="joint " + str(i + 1), alpha=0.5)

    if 'qddot_filtered' in without_data:
        for i in range(dim):
            plt.plot(ts, qddot_filtered[:, i], color=C[i])
    plt.legend(loc=1)
    # plt.legend(plots, ['joint ' + str(i + 1) for i in range(2)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("radians/$s^2$")
    plt.title("Joint Accelerations")

    plt.subplot(2, 3, 4)
    plots = plt.plot(ts, torque_diff[:, :])
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Residual Joint torques")

    plt.subplot(2, 3, 5)
    plots = plt.plot(ts, q_with[:, :] - q_wout[:, :])
    plt.legend(plots, ['joint ' + str(i + 1) for i in range(dim)], loc=1)
    plt.xlabel("time (s)")
    plt.ylabel("rad")
    plt.title("Difference in joint positions")

    plt.show()


def generate_data(q_traj, total_time, timestep, with_object=False, dim=2, std=0):
    if dim == 7:
        plant, diagram, viz, meshcat, state_logger, torque_logger = \
            create_panda_manipulation_station(q_traj, timestep, with_object, object_name='009_gelatin_box')
    else:
        plant, diagram, viz, meshcat, state_logger, torque_logger = \
            create_planar_manipulation_station(q_traj, timestep, dim, with_object)
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())

    # starting_q = [np.pi if i == 0 else 0 for i in range(dim)]
    # if dim == 7:
    #     starting_q = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785]
    plant.SetPositions(plant_context,
                       plant.GetModelInstanceByName("arm"),
                       np.array(q_traj.value(0)))
    viz.StartRecording()
    simulator.AdvanceTo(total_time)

    viz.StopRecording()
    viz.PublishRecording()

    state_log = state_logger.FindLog(simulator.get_context())
    torque_log = torque_logger.FindLog(simulator.get_context())

    data = gather_data(state_log, torque_log, dim, std=std)
    return data, plant


def gather_data(state_log, torque_log, dim=2, std=0):
    t = state_log.sample_times()
    T = len(t)
    qdot = state_log.data()[dim:, :]
    qddot = np.zeros((dim, T))
    for i in range(dim):
        qddot[i, :] = np.gradient(qdot[i, :], t)
    q_meas = state_log.data()[:dim, :] # + np.random.normal(0, std, (dim, T))
    qdot = qdot # + np.random.normal(0, std, (dim, T))

    data = {
        'q_meas': q_meas.T,
        'qdot_meas': qdot.T,
        'qddot_est': qddot.T, #  + np.random.normal(0, std, qddot.T.shape),
        'tau_meas': torque_log.data().T, #  + np.random.normal(0, std, (T, dim)),
        't': t.reshape((t.shape[0], 1)),
    }
    return data

def run_multiple_trials(q_traj_fn, trials, total_time, timestep, dim=2, with_object=False):
    datas = []
    for i in range(trials):
        if dim == 2:
            base_position = [np.pi, 0]
        if dim == 1:
            base_position = [np.pi]
        if dim == 7:
            base_position = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785]
        q_traj = q_traj_fn(i + 1, dim, base_position=base_position)

        data, plant = generate_data(q_traj, total_time, timestep, with_object, dim)
        datas.append(data)

    all_data = stack_data(datas)
    return all_data, plant


def stack_data(data_list):
    """
    I think this can be looped over dictionary keys
    :param data_list:
    :return:
    """
    q_meas = np.vstack([data['q_meas'] for data in data_list])  # T x N
    qdot_meas = np.vstack([data['qdot_meas'] for data in data_list])
    qddot_est = np.vstack([data['qddot_est'] for data in data_list])
    tau_meas = np.vstack([data['tau_meas'] for data in data_list])
    if 'tau_filtered' in data_list[0]:
        tau_filtered = np.vstack([data['tau_filtered'] for data in data_list])
    if 'qddot_filtered' in data_list[0]:
        qddot_filtered = np.vstack([data['qddot_filtered'] for data in data_list])
    if 'q_meas_unaligned' in data_list[0]:
        q_meas_unaligned = np.vstack([data['q_meas_unaligned'] for data in data_list])
    ts = np.vstack([data['t'] for data in data_list])
    T = np.sum([len(data['t']) for data in data_list])

    stacked_dict = {
        'q_meas': q_meas,
        'qdot_meas': qdot_meas,
        'qddot_est': qddot_est,
        'tau_meas': tau_meas,
        'ts': ts,
        'T': T,
    }
    if 'tau_filtered' in data_list[0]:
        stacked_dict['tau_filtered'] = tau_filtered
    if 'qddot_filtered' in data_list[0]:
        stacked_dict['qddot_filtered'] = qddot_filtered
    if 'q_meas_unaligned' in data_list[0]:
        stacked_dict['q_meas_unaligned'] = q_meas_unaligned

    return stacked_dict


def residual_torque_data(trials=10, dim=2, pick_and_place=False, std=0.0):
    without_datas = []
    with_datas = []
    total_time = 10.0
    timestep = 0
    for i in range(trials):
        if pick_and_place:
            # Do all of the things necessary to make an IK trajectory, i.e. create dummy plant
            builder = DiagramBuilder()
            # might need to use the other code base in order to get the plant
            # nah just upload from the urdf
            dummy_plant, _ = AddMultibodyPlantSceneGraph(builder, 0)
            parser = Parser(dummy_plant)
            dummy_arm = parser.AddModelFromFile("src/calibration/models/panda_arm_hand_no_collision.urdf",
                                                  "dummy_arm")
            dummy_plant.WeldFrames(dummy_plant.world_frame(), dummy_plant.GetFrameByName("panda_link0"))
            dummy_plant.Finalize()

            ik_failure = True
            while ik_failure:
                init_pose = [np.random.uniform(0.3, 0.6), np.random.uniform(-0.3, 0.3), np.random.uniform(0.4, 0.7)]
                end_pose = [np.random.uniform(-0.3, -0.6), np.random.uniform(-0.3, 0.3), np.random.uniform(0.4, 0.7)]
                X_L7_start = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), init_pose)
                X_L7_end = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 3.14)), end_pose)
                try:
                    q_traj = PickAndPlaceTrajectorySource(dummy_plant, X_L7_start, X_L7_end).calc_q_traj()
                    ik_failure = False
                except:
                    print('IK Failure, trying again')
        else:
            base_position = None
            if dim == 7:
                base_position = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785]
            elif dim == 2:
                base_position = [np.pi, 0]
            elif dim == 1:
                base_position = [np.pi]
            q_traj = generate_sinusoidal_traj(i + 1, dim=dim, base_position=base_position)

        print('generating data without object')
        without_data, without_plant = generate_data(q_traj, total_time, timestep, False, dim=dim, std=std)
        without_datas.append(without_data)

        with_data, with_plant = generate_data(q_traj, total_time, timestep, True, dim=dim, std=std)
        with_datas.append(with_data)

    without_object_data = stack_data(without_datas)
    with_object_data = stack_data(with_datas)
    np.save('src/calibration/plotting_data/pick_and_place_5-8.npy', [without_data])
    # np.save('src/calibration/plotting_data/noise_with_5-8.npy', [with_data])
    return without_object_data, with_object_data, without_plant, with_plant


def convert_real_data(data):
    # Put data in our format
    qdot = data['f_vel']
    t = data['f_t']
    T = t.shape[1]
    print(t)
    # qddot = np.zeros((T, 7))
    # print(qdot.shape)
    # for i in range(7):
    #     qddot[:, i] = np.gradient(qdot[:, i], t[:, 0])

    new_dict = {
        'q_meas': data['f_meas'].T,
        'qdot_meas': data['f_vel'].T,
        'qddot_est': data['v_dot'].T,
        'tau_meas': data['f_tau'].T,
        'tau_filtered': data['tau_filtered'].T,
        'qddot_filtered': data['v_dot_filtered'].T,
        'q_meas_unaligned': data['f_meas_unaligned'],
        'ts': t.T,
        'T': T,  # should use T from data...
    }
    return new_dict

def load_robot_data(without_files, with_files=None, calc_qdot=False):
    if with_files:
        D = load_multiple_residuals_files(without_files, with_files, calc_qdot=calc_qdot)
        without_data = D['without']
        with_data = D['with']
        # without_data = load_multiple_files(without_files)
        # with_data = load_multiple_files(with_files)
    else:
        without_data = load_multiple_files(without_files)
    # without_data = load_multiple_files(without_files, filter_acceleration=True)

    # if with_files:
    #     with_data = load_multiple_files(with_files, filter_acceleration=True)
    # else:
    #     with_data = None

    builder1 = DiagramBuilder()
    # might need to use the other code base in order to get the plant
    # nah just upload from the urdf
    without_plant, _ = AddMultibodyPlantSceneGraph(builder1, 0)
    parser = Parser(without_plant)
    without_arm = parser.AddModelFromFile("src/calibration/models/panda_arm_hand_no_collision.urdf", "without_arm")
    without_plant.WeldFrames(without_plant.world_frame(), without_plant.GetFrameByName("panda_link0"))
    without_plant.Finalize()

    builder2 = DiagramBuilder()
    with_plant, scene_graph = AddMultibodyPlantSceneGraph(builder2, 0)
    parser = Parser(with_plant)
    with_arm = parser.AddModelFromFile("src/calibration/models/panda_arm_hand_no_collision.urdf", "with_arm")
    with_plant.WeldFrames(with_plant.world_frame(), with_plant.GetFrameByName("panda_link0"))

    # Weld the object just like in original simulation
    # -0.7853915
    tf = RigidTransform(RollPitchYaw(0, 0, -0.7853915), [0.0, 0.0, 0.1034])  # Still unsure what this transform is
    obj = parser.AddModelFromFile("src/calibration/models/010_potted_meat_can.sdf", 'gelatin')
    # obj = parser.AddModelFromFile("src/calibration/models/dumbbell_2lb.sdf", 'object')
    l7_body_frame = with_plant.GetBodyByName("panda_link8").body_frame()
    obj_body_frame = with_plant.GetBodyByName('base_link_meat').body_frame()
    with_plant.WeldFrames(l7_body_frame, obj_body_frame, tf)
    with_plant.Finalize()

    meshcat = StartMeshcat()
    viz = MeshcatVisualizer.AddToBuilder(builder2, scene_graph, meshcat)
    diagram = builder2.Build()

    simulator = Simulator(diagram)
    plant_context = with_plant.GetMyContextFromRoot(simulator.get_mutable_context())

    starting_q = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785]
    with_plant.SetPositions(plant_context,
                            with_arm,
                            np.array(starting_q))
    # viz.StartRecording()
    # simulator.AdvanceTo(5)
    #
    # viz.StopRecording()
    # viz.PublishRecording()



    without_data = convert_real_data(without_data)
    # np.save('src/calibration/plotting_data/joint_data_wout_obj_unaligned.npy', [without_data])
    if with_data:
        with_data = convert_real_data(with_data)
        # np.save('src/calibration/plotting_data/joint_data_with_obj_unaligned.npy', [with_data])

    return without_data, with_data, without_plant, with_plant


if __name__ == "__main__":
    symbolic = False
    autodiff = False
    residual = True
    sensitivity_state = False
    show_logs = False
    point_mass = False
    real_data = False
    pick_and_place = False
    with_object = True
    num_traj = 10
    dim = 2


    # # TEST SIGNAL PROCESSING
    # without_object_data = load_and_parse_file('src/calibration/real_data/4-27_without_weight_dynamic_amp0.3_traj_3.npy')['f_meas'][:, 0]
    # with_object_data = load_and_parse_file('src/calibration/real_data/5-1_with_weight_dynamic_amp0.3_traj_3.npy')['f_meas'][:, 0]
    #
    # index_map = match_sequences(without_object_data, with_object_data)
    # print(len(without_object_data) < len(with_object_data))
    # align_sequences(without_object_data, with_object_data, index_map, plot=True)
    # assert False

    if residual:
        three_dee = dim > 2
        if real_data:
            without_object_data, with_object_data, without_plant, with_plant = load_robot_data(
                [f'src/calibration/real_data/4-27_without_weight_dynamic_amp0.3_traj_{i+1}.npy' for i in range(0, 1)], # 10, 2)],
                [f'src/calibration/real_data/5-1_with_weight_dynamic_amp0.3_traj_{i+1}.npy' for i in range(0, 1)], # 10, 2)],
                calc_qdot=True)
                # [f'src/calibration/sim_data/4-07_without_obj_amp0.3_traj_1_2e-3.npy'], #  for i in range(1, 6)],
                # [f'src/calibration/sim_data/4-07_with_obj_amp0.3_traj_1_2e-3.npy'])  # for i in range(1, 6)])
                # ['src/calibration/sim_data/2-28_without_obj_amp0.3_traj_1.npy'],
                # ['src/calibration/sim_data/2-28_with_obj_amp0.3_traj_1.npy'])
                # ['src/calibration/sim_data/4-11_without_obj_amp0.4_traj_1.npy'],
                # ['src/calibration/sim_data/4-11_with_obj_amp0.4_traj_1.npy'])
            dim = 7
        else:
            if three_dee:
                dim = 7
            else:
                dim = 2
            without_object_data, with_object_data, without_plant, with_plant = residual_torque_data(num_traj, dim=dim, pick_and_place=pick_and_place)
        # assert False
        # Trajectories match up :) for 2 and 7 dof simulated cases
        # Also match pretty closely for the real data! Big win, but also a small object
        plot_trajectory_difference(without_object_data, with_object_data, 11, dim=dim)

        sym_plant, sym_context, parameters, state_variables = sym_utils.create_symbolic_plant(
            with_plant, dim=dim, with_object=True, three_dee=three_dee)
        robot_params, object_params = parameters

        load_data = False
        if real_data or dim == 7:
            if load_data:
                alpha_fit = np.load('alphafit_5-4_train.npy')
                print(alpha_fit)
            else:
                alpha_fit, alpha_p = sym_utils.calc_data_matrix_w_obj_7dof(sym_plant, sym_context,
                                                                  without_object_data, with_object_data,
                                                                  robot_params, object_params,
                                                                  state_variables, real_data=real_data, sdp=False)
                print(alpha_fit)
        else:
            # Values change drastically with more data, but they seem to get closer to the goal?
            # Hard to tell because the estimates still are not good
            alpha_fit, alpha_p = sym_utils.calc_data_matrix_with_object(
                sym_plant, sym_context, without_object_data, with_object_data, robot_params, object_params, state_variables,
                dim=dim)

        if real_data:
            _, with_object_data, _, with_plant = load_robot_data(
                [f'src/calibration/real_data/4-27_without_weight_dynamic_amp0.3_traj_{i + 1}.npy' for i in range(1, 10, 2)],
                [f'src/calibration/real_data/5-1_with_weight_dynamic_amp0.3_traj_{i + 1}.npy' for i in range(1, 10, 2)])
        sym_utils.plot_model_errors(with_plant, alpha_fit, with_object_data, dim=dim, three_dee=three_dee)
        print(alpha_fit)
    elif symbolic:
        three_dee = dim > 2

        if real_data:
            without_data, with_data, without_plant, with_plant = load_robot_data(
                [f'src/calibration/real_data/4-27_without_weight_dynamic_amp0.3_traj_{i + 1}.npy' for i in range(num_traj)])
        else:
            without_data, with_data, without_plant, with_plant = residual_torque_data(trials=num_traj, dim=dim)
        sym_plant, sym_context, parameters, state_variables = sym_utils.create_symbolic_plant(without_plant, dim=dim, three_dee=three_dee)
        if dim > 2:
            alpha_fit, alpha_sym = sym_utils.calc_data_matrix_7dof(sym_plant, sym_context, without_data, parameters, state_variables)
            alpha_gt = np.zeros(59)
        else:
            alpha_fit, alpha_gt, alpha_sym = sym_utils.calc_data_matrix(
                sym_plant, sym_context, without_data, parameters, state_variables, dim=dim)

        for fit, gt, sym in zip(alpha_fit, alpha_gt, alpha_sym):
            print(f'Estimated {sym.to_string()}: {fit} \t GT: {gt}')

        if dim > 2:
            new_plant, new_context = sym_utils.update_plant(without_plant, alpha_fit)  # doesn't exist yet

        if with_object:
            print('with object!')
            sym_plant_obj, sym_context_obj, parameters_obj, state_variables_obj = \
                sym_utils.create_plant_with_symbolic_object(without_plant, dim=dim, three_dee=three_dee)
            alpha_p_fit, alpha_p_sym = sym_utils.calc_data_matrix_just_obj(
                sym_plant_obj,
                sym_context_obj,
                with_data,
                parameters_obj,
                state_variables_obj,
                dim=dim)

            print(alpha_p_fit)


    elif autodiff:
        three_dee = dim > 2

        if real_data:
            without_data, with_data, without_plant, with_plant = load_robot_data(
                [f'src/calibration/real_data/4-27_without_weight_dynamic_amp0.3_traj_{i + 1}.npy' for i in
                 range(num_traj)])
        else:
            without_data, with_data, without_plant, with_plant = residual_torque_data(trials=num_traj, dim=dim)
        plant_ad, context_ad = ad_utils.create_autodiff_plant(without_plant, dim=dim, three_dee=three_dee)
        e_vector = ad_utils.compute_errors(plant_ad, context_ad, without_data, dim=dim)
        delta, new_error = ad_utils.least_square_regression(e_vector, without_data['T'], dim=dim, three_dee=three_dee)
        print(np.linalg.norm(delta))
        new_plant, new_context = ad_utils.apply_model(delta, dim=dim, three_dee=three_dee)
        ad_utils.plot_model_errors(new_plant, new_context, without_data, dim=dim)

        if with_object:
            plant_ad_obj, context_ad_obj = ad_utils.create_plant_with_autodiff_object(without_plant, dim=dim, three_dee=three_dee)
            e_vector = ad_utils.compute_errors(plant_ad_obj, context_ad_obj, with_data)
            delta, new_error = ad_utils.least_square_regression(e_vector, with_data['T'], dim=dim,
                                                                three_dee=three_dee)
            print('object delta: ', delta)  # why is delta size 8, should be size 4
            print('object delta norm: ', np.linalg.norm(delta))
            new_plant_obj, new_context_obj = ad_utils.apply_model_object(delta)  # using an existing plant here causes bugs
            ad_utils.plot_model_errors(new_plant_obj, new_context_obj, with_data, dim=dim)

            # Transform the resulting object delta
            # only for 2D right now
            if not three_dee:
                con = without_plant.CreateDefaultContext()
                link_N = without_plant.GetBodyByName(f"link{dim}")
                old_m = link_N.get_mass(con)
                old_spa = link_N.CalcSpatialInertiaInBodyFrame(con)
                Ixx = old_spa.CopyToFullMatrix6()[:3, :3][0, 0]
                Izz = old_spa.CopyToFullMatrix6()[:3, :3][2, 2]
                new_c = [delta[1, 0], link_N.CalcCenterOfMassInBodyFrame(con)[1], delta[2, 0]]
                new_G = UnitInertia(Ixx / old_m, delta[3, 0] / delta[0, 0], Izz / old_m)
                N_spa = SpatialInertia(
                    delta[0, 0], new_c, new_G, skip_validity_check=True)

                # P_spa = N_spa.ReExpress(...)  # Already in the same frame yippee
                P_spa = N_spa.Shift([0, 0, -1.])

                print(P_spa.get_mass())
                print(P_spa.get_com())
                print(P_spa.CopyToFullMatrix6()[:3, :3])


    elif sensitivity_state:
        num_trials = 10

        alphas = []
        conditions = []
        for i in range(num_trials):
            print(f'Trial {i + 1}')
            without_object_data, with_object_data, without_plant, with_plant = residual_torque_data(1, dim=dim,
                                                                                                    pick_and_place=pick_and_place,
                                                                                                    std=0.0)
            sym_plant, sym_context, parameters, state_variables = sym_utils.create_symbolic_plant(
                with_plant, dim=7, with_object=True, three_dee=True)
            robot_params, object_params = parameters

            # plot_trajectory_difference(without_object_data, with_object_data, 11, dim=dim)

            alpha_fit, alpha_p, cond = sym_utils.calc_data_matrix_w_obj_7dof(sym_plant, sym_context,
                                                                       without_object_data, with_object_data,
                                                                       robot_params, object_params,
                                                                       state_variables, real_data=False)

            alphas.append(alpha_fit)
            conditions.append(cond)

            np.save('src/calibration/fitting_alphas_pick_and_place2.npy', alphas)
            # np.save('src/calibration/fitting_alphas_sinusoidal.npy', alphas)

            np.save('src/calibration/condition_numbers_pick-and-place2.npy', np.array(conditions))
            print(f"average condition number: {np.mean(np.array(conditions))}")

