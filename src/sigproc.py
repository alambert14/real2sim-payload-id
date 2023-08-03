import matplotlib.pyplot as plt
import numpy as np
import scipy
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.signal import butter, sosfiltfilt


def convert_data(data, crop_end_point=False):
    """
    Convert data from a list of dictionaries to a tuple of numpy arrays
    Each list entry in data is from a timestep
    """
    joint_position_commanded = np.array([0.0 for t in data])

    # t["joint_position_commanded"][:num_joints] for t in data])
    joint_position_measured = np.array([t["joint_position_measured"][:7] for t in data])
    joint_velocity_estimated = np.array([t["joint_velocity_estimated"][:7] for t in data])
    joint_torque_measured = np.array([t["joint_torque_measured"][:7] for t in data])
    external_torques = np.array([t["external_torque"][:7] for t in data])
    external_wrenches = np.array([t["cartesian_measured"][:7] for t in data])
    utime = np.array([t["utime"] for t in data])

    if crop_end_point:
        end_point_ratio = 0.1
        end_point_num = int(end_point_ratio * len(joint_position_measured))
        joint_position_commanded = joint_position_commanded[end_point_num:-end_point_num]
        joint_position_measured = joint_position_measured[end_point_num:-end_point_num]
        joint_velocity_estimated = joint_velocity_estimated[end_point_num:-end_point_num]
        joint_torque_measured = joint_torque_measured[end_point_num:-end_point_num]
        external_torques = external_torques[end_point_num:-end_point_num]
        external_wrenches = external_wrenches[end_point_num:-end_point_num]
        utime = utime[end_point_num:-end_point_num]

    return (
        joint_position_commanded,
        joint_position_measured,
        joint_velocity_estimated,
        joint_torque_measured,
        external_torques,
        external_wrenches,
        utime,
    )


def moving_avg_filter(data, window_size=0.1):
    # window_size between 0 (no filter) and 1 (all window average) ~0.1 recommended
    N = int(max(1, int(len(data) * window_size)))
    kernel = np.ones(N) / N
    data_convolved = np.convolve(data, kernel, mode="same")  # mode
    return data_convolved


def butter_lowpass_filter(data, cutoff, fs, order):
    sos = butter(order, cutoff / fs, output="sos")
    filtered = sosfiltfilt(sos, data, padlen=0)
    return filtered


def filter_vdot_and_tau(v_dot, tau, T, cutoff=10, fs=200, order=2):
    moving_ave_v_dot = np.zeros((T, 7))
    low_pass_v_dot = np.zeros((T, 7))
    low_pass_tau = np.zeros((T, 7))
    for i in range(7):
        moving_ave_v_dot[:, i] = moving_avg_filter(v_dot[:, i], 0.02)
        low_pass_v_dot[:, i] = butter_lowpass_filter(v_dot[:, i], cutoff=cutoff, fs=fs, order=order)
        low_pass_tau[:, i] = butter_lowpass_filter(tau[:, i], cutoff=cutoff, fs=fs, order=order)

    return low_pass_v_dot, low_pass_tau


def filter_data(signal, cutoff=10, fs=200, order=2):
    low_pass = np.zeros(signal.shape)
    for i in range(signal.shape[1]):
        low_pass[:, i] = butter_lowpass_filter(signal[:, i], cutoff=cutoff, fs=fs, order=order)

    return low_pass


def load_and_parse_file(file, show_plots=True, calc_qdot=False, fit_splines=False):
    """
    parse offline data and apply low-pass filter.
    """
    F = np.load(file, allow_pickle=True)
    f_comm, f_meas, f_vel, f_tau, tau_ext, wrench_ext, f_t = convert_data(F)
    T = len(f_t)

    f_meas = filter_data(f_meas)
    if calc_qdot:
        q_dot = np.zeros((T, 7))
        for i in range(7):
            q_dot[:, i] = np.gradient(f_meas[:, i], f_t)
    else:
        q_dot = f_vel
    v_dot = np.zeros((T, 7))
    for i in range(7):
        v_dot[:, i] = np.gradient(q_dot[:, i], f_t)

    v_dot_filtered, tau_filtered = filter_vdot_and_tau(v_dot, f_tau, T)
    return {
        "f_comm": f_comm,
        "f_meas": f_meas,
        "f_vel": q_dot,
        "f_tau": f_tau,
        "v_dot": v_dot,
        "v_dot_filtered": v_dot_filtered,
        "tau_filtered": tau_filtered,
        "tau_ext": tau_ext,
        "wrench_ext": wrench_ext,
        "f_t": f_t,
    }


def load_multiple_files(
    list_of_files,
    show_plots=False,
    show_external_torques=False,
):
    """
    parse multiple offline data and stack them
    """
    F = {}

    for file in list_of_files:
        print("parsing " + file)
        F[file] = load_and_parse_file(
            file,
            show_plots=show_plots,
        )

    dim = np.min([len(F[file]["f_meas"]) for file in list_of_files])
    f_comm = np.vstack([F[file]["f_comm"][:dim] for file in list_of_files])
    f_meas = np.vstack([F[file]["f_meas"][:dim] for file in list_of_files])
    f_vel = np.vstack([F[file]["f_vel"][:dim] for file in list_of_files])
    f_tau = np.vstack([F[file]["f_tau"][:dim] for file in list_of_files])
    v_dot = np.vstack([F[file]["v_dot"][:dim] for file in list_of_files])
    # low_pass_v_dot = np.vstack([F[file]["low_pass_v_dot"][:dim] for file in list_of_files])
    tau_ext = np.vstack([F[file]["tau_ext"][:dim] for file in list_of_files])
    wrench_ext = np.vstack([F[file]["wrench_ext"][:dim] for file in list_of_files])
    # low_pass_tau = np.vstack([F[file]["low_pass_tau"][:dim] for file in list_of_files])

    # concatenate all the times
    f_t = np.vstack([F[file]["f_t"][:dim].reshape((dim, 1)) for file in list_of_files])
    # print(f_t.shape, f_meas.shape)

    return {
        "f_comm": f_comm,
        "f_meas": f_meas,
        "f_vel": f_vel,
        "f_tau": f_tau,
        "v_dot": v_dot,
        "tau_ext": tau_ext,
        "wrench_ext": wrench_ext,
        "f_t": f_t,
        "T": dim,
        "N": len(list_of_files),
    }


# def load_and_parse_file(file, show_plots=True):
#     """
#     parse offline data and apply low-pass filter.
#     """
#     F = np.load(file, allow_pickle=True)
#     f_comm, f_meas, f_vel, f_tau, tau_ext, wrench_ext, f_t = convert_data(F)
#
#     T = len(f_t)
#     v_dot = np.zeros((T, 7))
#     for i in range(7):
#         v_dot[:, i] = np.gradient(f_vel[:, i], f_t)
#
#     return {
#         "f_comm": f_comm,
#         "f_meas": f_meas,
#         "f_vel": f_vel,
#         "f_tau": f_tau,
#         "v_dot": v_dot,
#         # "low_pass_v_dot": low_pass_v_dot,
#         # "low_pass_tau": low_pass_tau,
#         "tau_ext": tau_ext,
#         "wrench_ext": wrench_ext,
#         "f_t": f_t,
#     }


def load_multiple_residuals_files(
    without_files,
    with_files,
    show_plots=False,
    calc_qdot=False,
    fit_splines=False,
):
    """
    parse multiple offline data, align and stack them
    so ugly!! 🤮
    less ugly now :]
    but still ugly
    """

    D = {'without': {},
         'with': {}}
    to_return = {'without': {},
                 'with': {}}
    to_return['without']['N'] = len(without_files)
    to_return['with']['N'] = len(with_files)
    data_labels = ['f_meas', 'f_vel', 'f_tau', 'v_dot', 'v_dot_filtered', 'tau_filtered', 'tau_ext', 'f_t']
    for files in zip(without_files, with_files):
        print(f"parsing {files[0]} and {files[1]}")
        without_data = load_and_parse_file(
            files[0],
            show_plots=show_plots,
            calc_qdot=calc_qdot,
            fit_splines=fit_splines,
        )
        with_data = load_and_parse_file(
            files[1]
        )

        to_return['without']['f_meas_unaligned'] = without_data['f_meas']
        to_return['with']['f_meas_unaligned'] = with_data['f_meas']

        without_shorter = len(without_data['f_meas'][:, 0]) < len(with_data['f_meas'][:, 0])

        index_maps = []
        for i in range(7):
            shorter = without_data['f_meas'][:, i] if without_shorter else with_data['f_meas'][:, i]
            longer = with_data['f_meas'][:, i] if without_shorter else without_data['f_meas'][:, i]
            index_maps.append(match_sequences(shorter, longer))

        D['without'][files[0]] = {}
        D['with'][files[1]] = {}
        for label in data_labels:
            D['without'][files[0]][label] = []
            D['with'][files[1]][label] = []

            if label == 'f_t':
                if without_shorter:
                    ts = without_data[label].reshape((1, -1))
                else:
                    ts = with_data[label].reshape((1, -1))
                D['without'][files[0]]['f_t'] = np.array(ts) # make ts the same
                D['with'][files[1]]['f_t'] = np.array(ts)
                continue  # don't do the rest if time

            for i in range(7):
                shorter = without_data[label][:, i] if without_shorter else with_data[label][:, i]
                longer = with_data[label][:, i] if without_shorter else without_data[label][:, i]
                shorter, new_longer = align_sequences(shorter, longer, index_maps[i])

                if without_shorter:
                    D['without'][files[0]][label].append(shorter)
                    D['with'][files[1]][label].append(new_longer)
                else:
                    D['without'][files[0]][label].append(new_longer)
                    D['with'][files[1]][label].append(shorter)

            D['without'][files[0]][label] = np.array(D['without'][files[0]][label])
            D['with'][files[1]][label] = np.array(D['with'][files[1]][label])

    for label in data_labels:
        print(D['without'][without_files[0]][label].shape)
        to_return['without'][label] = np.hstack([D['without'][file][label] for file in without_files])
        to_return['with'][label] = np.hstack([D['with'][file][label] for file in with_files])

    return to_return


def align_sequences(shorter, longer, index_map, plot=False):
    new_longer = np.zeros(shorter.shape)
    for i in range(new_longer.shape[0]):
        # print(index_map[i])
        new_longer[i] = longer[index_map[i]]
        # THIS WORKS INCREDIBLY WOOHOOO!!!
    if plot:
        plt.plot(shorter)
        plt.plot(longer)
        plt.plot(new_longer)
        plt.show()

    return (shorter, new_longer)


def match_sequences(shorter, longer):
    """
    Align 2 sequences, where the longer sequence is made to fit into the shorter sequence
    :param with_data:
    :param without_data:
    :return:
    """
    path = dtw.warping_path(shorter, longer, window=50)
    # nonunique but meh, probably need to average instead of overwrite
    indices_map = {x[0]: x[1] for x in path}
    
    return indices_map
    

def plot_warping(with_data, without_data):
    """
    Used for testing DTW algorithm
    """
    q_with = with_data['f_meas'][:, 3]
    q_wout = without_data['f_meas'][:, 3]

    # d, paths = dtw.warping_paths(q_wout, q_with, window=25, psi=2)
    # best_path = dtw.best_path(paths)
    # dtwvis.plot_warpingpaths(q_wout, q_with, paths, best_path, filename='warp.png')
    # plt.plot()

    # need to sample more finely I think
    # but then we would need to interpolate
    path = dtw.warping_path(q_wout, q_with, window=50)
    # print(path)
    dtwvis.plot_warping(q_wout, q_with, path, filename='warp.png')

    # try multidimensional