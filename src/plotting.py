import sys
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import csv
import pandas as pd

from generate_data_2dof import residual_torque_data

DPI = 300
PREFIX = "/home/aalamber/Pictures/Thesis"


COLORS = {
    "purple": "#806BFF",
    "blue": "#5dace1",
    "cyan": "#6ddad4",
    "green": "#4bb652",
    "yellow": "#e9c348",
    "orange": "#e99d48",
    "red": "#e96248",
}


def joint_angles(filename, save_name):
    rcParams['axes.labelpad'] = 7

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]
    plt.figure(figsize=(6, 3))
    plt.title("Pick-and-Place Trajectory", fontsize=20)

    data = np.load(filename, allow_pickle=True)[0]
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['q_meas'].shape[0]
        ts = np.linspace(0, data['t'][-1], l)

        plt.plot(ts, data['q_meas'][:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')

        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Position (rad)', fontsize=18)
        plt.ylim(-3, 2.5)
        plt.xlim(0, data['t'][-1][0])
        plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(f'{PREFIX}/{save_name}', dpi=DPI, bbox_inches="tight")


def all_joint_data(filename, save_name):
    rcParams['axes.labelpad'] = 7

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]
    # plt.figure(figsize=(18, 3))

    fig, ax = plt.subplots(1, 3, figsize=(22, 5.5))
    fig.tight_layout(pad=4)
    fig.suptitle("Example Joint Trajectory", fontsize=24)
    ax[0].set_title('$\mathbf{q}[t]$')

    data = np.load(filename, allow_pickle=True)[0]
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['q_meas'].shape[0]
        ts = np.linspace(0, data['t'][-1], l)

        ax[0].plot(ts, data['q_meas'][:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')

        #plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
        ax[0].set(xlabel='Time (s)', ylabel='Position (rad)')
        # ax[0].xlabel('Time (s)', fontsize=18)
        # ax[0].ylabel('Position (rad)', fontsize=18)
        ax[0].set_ylim(-3, 2.5)
        ax[0].set_xlim(0, data['t'][-1][0])
        ax[0].set_yticks(np.linspace(-np.pi, np.pi, 5))
        ax[0].set_yticklabels(['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    # Velocities
    ax[1].set_title('$\mathbf{\dot{q}}[t]$')

    data = np.load(filename, allow_pickle=True)[0]
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['qdot_meas'].shape[0]
        ts = np.linspace(0, data['t'][-1], l)

        ax[1].plot(ts, data['qdot_meas'][:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')

        # plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
        ax[1].set(xlabel='Time (s)', ylabel='Angular Velocity (rad/$s$)')
        # ax[1].xlabel('Time (s)', fontsize=18)
        # ax[1].ylabel('Angular Velocity (rad/$s$)', fontsize=18)
        ax[1].set_ylim(-3, 2.5)
        ax[1].set_xlim(0, data['t'][-1][0])
        ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
        ax[1].set_yticklabels(['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    # Accelerations
    # plt.subplot(1, 3, 3)
    ax[2].set_title('$\mathbf{\ddot{q}}[t]$')

    data = np.load(filename, allow_pickle=True)[0]
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['qddot_est'].shape[0]
        ts = np.linspace(0, data['t'][-1], l)

        ax[2].plot(ts, data['qddot_est'][:, i], ls, color=c, linewidth=2) # , alpha=0.5)
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['qddot_est'].shape[0]
        ts = np.linspace(0, data['t'][-1], l)

        # ax[2].plot(ts, data['qddot_filtered'][:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')
        # ax[2].set_yticks(np.linspace(-np.pi, np.pi, 5))
        # ax[2].set_yticklabels(['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])
    ax[2].set(xlabel='Time (s)', ylabel='Angular Acceleration (rad/$s^2$)')
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
    # ax[2].ylabel(, fontsize=18)
    ax[2].set_ylim(-10, 10)
    ax[2].set_xlim(0, data['t'][-1][0])
    # ax[2].set_yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

    # ax[2].gca().spines['top'].set_visible(False)
    # ax[2].gca().spines['right'].set_visible(False)

    plt.savefig(f'{PREFIX}/{save_name}', dpi=DPI, bbox_inches="tight")


def traj_unalignment(with_filename, without_filename):
    rcParams['axes.labelpad'] = 7

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]
    plt.figure(figsize=(6, 3))
    plt.title("$\mathbf{q}[t]$ Without Alignment", fontsize=24)

    with_data = np.load(with_filename, allow_pickle=True)[0]
    without_data = np.load(without_filename, allow_pickle=True)[0]

    without_obj = without_data['q_meas_unaligned']
    with_obj = with_data['q_meas_unaligned']
    print(without_obj.shape, with_obj.shape)

    if without_obj.shape[0] > with_obj.shape[0]:
        offset = without_obj.shape[0] - with_obj.shape[0]
        nans = np.empty((offset, 7))
        nans[:] = np.nan
        with_obj = np.concatenate((with_obj, nans))
        l = without_data['q_meas_unaligned'].shape[0]
        ts = np.linspace(0, without_data['ts'][-1], l)
    else:
        offset = with_obj.shape[0] - without_obj.shape[0]
        nans = np.empty((offset, 7))
        nans[:] = np.nan
        without_obj = np.concatenate((without_obj, nans))
        l = with_data['q_meas_unaligned'].shape[0]
        ts = np.linspace(0, with_data['ts'][-1], l)
    for i in range(7):
        c = joint_colors[i]
        ls = '-'

        print(ts.shape)
        if i == 0:
            plt.plot(ts, without_obj[:, i], ls, color=c, linewidth=2, label=f'Unloaded', alpha=0.5)

            plot, = plt.plot(ts, with_obj[:, i], '--', color=c, linewidth=2, label=f'Loaded')
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')
        else:
            plt.plot(ts, without_obj[:, i], ls, color=c, linewidth=2, alpha=0.5)
            plot,= plt.plot(ts, with_obj[:, i], '--', color=c, linewidth=2)
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Position (rad)', fontsize=18)
        plt.ylim(-3, 2.5)
        plt.xlim(0, without_data['ts'][-1][0])
        plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
    plt.savefig(f'{PREFIX}/unalignment.png', dpi=DPI, bbox_inches="tight")


def traj_alignment(with_filename, without_filename):
    rcParams['axes.labelpad'] = 7

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]
    plt.figure(figsize=(6, 3))
    plt.title("$\mathbf{q}[t]$ After Alignment", fontsize=24)

    with_data = np.load(with_filename, allow_pickle=True)[0]
    without_data = np.load(without_filename, allow_pickle=True)[0]

    without_obj = without_data['q_meas']
    with_obj = with_data['q_meas']
    l = without_data['q_meas_unaligned'].shape[0]
    ts = np.linspace(0, without_data['ts'][-1], l)
    for i in range(7):
        c = joint_colors[i]
        ls = '-'

        print(ts.shape)
        if i == 0:
            plt.plot(ts, without_obj[:, i], ls, color=c, linewidth=2, label=f'Unloaded', alpha=0.5)

            plot, = plt.plot(ts, with_obj[:, i], '--', color=c, linewidth=2, label=f'Loaded')
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')
        else:
            plt.plot(ts, without_obj[:, i], ls, color=c, linewidth=2, alpha=0.5)
            plot,= plt.plot(ts, with_obj[:, i], '--', color=c, linewidth=2)
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Position (rad)', fontsize=18)
        plt.ylim(-3, 2.5)
        plt.xlim(0, without_data['ts'][-1][0])
        plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
    plt.savefig(f'{PREFIX}/alignment.png', dpi=DPI, bbox_inches="tight")


def filtered_data(filename):
    rcParams['axes.labelpad'] = 15

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]
    plt.figure(figsize=(6, 3))

    plt.title("Torque Filtering", fontsize=24)

    data = np.load(filename, allow_pickle=True)[0]
    for i in range(7):
        c = joint_colors[i]
        ls = '-'
        l = data['tau_filtered'].shape[0]
        ts = np.linspace(0, data['ts'][-1], l)

        plt.plot(ts, data['tau_filtered'][:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
        plt.plot(ts, data['tau_meas'][:, i], ls, color=c, linewidth=2, alpha=0.5)

        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Torque (Nm)', fontsize=18)
        # plt.ylim(-3, 2.5)
        plt.xlim(0, data['ts'][-1][0])
        # plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(f'{PREFIX}/filtered_torques.png', dpi=DPI, bbox_inches="tight")


def mass_estimates(filename):
    rcParams['axes.labelpad'] = 7

    plt.figure(figsize=(6, 3))
    plt.title("Mass Estimates from Noisy \nTorque Measurements", fontsize=18)

    data = np.load(filename, allow_pickle=True)
    c = COLORS['blue']
    ls = '-'
    l = data.shape[0]
    # ts = np.linspace(0, data['ts'][-1], l)

    print(data.shape)
    plt.plot(data, ls, color=c, linewidth=2)

    # plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('Mass (kg)', fontsize=18)
    plt.ylim(0.36, 0.38)
    plt.xlim(0, len(data))
    # plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(f'{PREFIX}/mass_estimates.png', dpi=DPI, bbox_inches="tight")


def error_bars_traj_types(filename1, filename2=None):
    # Get ground truth
    without_object_data, with_object_data, without_plant, with_plant = residual_torque_data(1, dim=7, std=0.0)

    obj = with_plant.GetBodyByName('base_link_meat')
    context = with_plant.CreateDefaultContext()
    spa = obj.CalcSpatialInertiaInBodyFrame(context)

    mass = obj.get_mass(context)
    com = obj.CalcCenterOfMassInBodyFrame(context)
    I = spa.CalcRotationalInertia().CopyToFullMatrix3()
    gt = np.array([
        mass,
        mass * com[0],
        mass * com[1],
        mass * com[2],
        I[0, 0],
        I[1, 1],
        I[2, 2],
        I[0, 1],
        I[0, 2],
        I[1, 2],
    ])

    print(gt)

    data = np.load(filename1).squeeze()
    if filename2:
        data2 = np.load(filename2).squeeze()

    errors = (data - gt) ** 2
    if filename2:
        errors2 = (data2 - gt) ** 2

    rcParams['xtick.minor.size'] = 0
    rcParams['axes.grid'] = False

    plt.figure(figsize=(12, 6))

    offset = 0.2 if filename2 else 0.
    width = 0.4 if filename2 else 0.75
    plt.bar(np.arange(errors.shape[1]) - offset, np.mean(errors, axis=0), yerr=np.std(errors, axis=0), capsize=5,
            width=width, color="#806BFF", label='Sinusoidal')

    if filename2:
        plt.bar(np.arange(errors.shape[1]) + offset, np.mean(errors2, axis=0), yerr=np.std(errors2, axis=0), capsize=5,
                width=width, color="#4bb652", label='Pick-and-Place')

    plt.legend()
    plt.title('Mean Squared Error and Standard Deviation of Estimates from \n Sinusoidal and Pick-and-Place Trajectory Data',
              fontsize=18)
    plt.yscale('log')
    plt.xticks(np.arange(errors.shape[1]),
               ['$m$', '$mc_x$', '$mc_y$', '$mc_z$', '$I_{xx}$', '$I_{yy}$', '$I_{zz}$', '$I_{xy}$', '$I_{xz}$',
                '$I_{yz}$'])
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.savefig(f'/home/aalamber/Pictures/Thesis/pick-and-place-errors.png', dpi=300, bbox_inches="tight")


def error_bars_noise(filename1, save, filename2=None):
    # Get ground truth
    without_object_data, with_object_data, without_plant, with_plant = residual_torque_data(1, dim=7, std=0.0)

    obj = with_plant.GetBodyByName('base_link_meat')
    context = with_plant.CreateDefaultContext()
    spa = obj.CalcSpatialInertiaInBodyFrame(context)

    mass = obj.get_mass(context)
    com = obj.CalcCenterOfMassInBodyFrame(context)
    I = spa.CalcRotationalInertia().CopyToFullMatrix3()
    gt = np.array([
        mass,
        mass * com[0],
        mass * com[1],
        mass * com[2],
        I[0, 0],
        I[1, 1],
        I[2, 2],
        I[0, 1],
        I[0, 2],
        I[1, 2],
    ])

    print(gt)

    data = np.load(filename1).squeeze()
    if filename2:
        data2 = np.load(filename2).squeeze()

    errors = (data - gt) ** 2
    if filename2:
        errors2 = (data2 - gt) ** 2

    rcParams['xtick.minor.size'] = 0
    rcParams['axes.grid'] = False

    plt.figure(figsize=(12, 6))

    offset = 0.2 if filename2 else 0.
    width = 0.4 if filename2 else 0.75
    plt.bar(np.arange(errors.shape[1]) - offset, np.mean(errors, axis=0), yerr=np.std(errors, axis=0), capsize=5,
            width=width, color=COLORS['orange'], label='Sinusoidal')

    if filename2:
        plt.bar(np.arange(errors.shape[1]) + offset, np.mean(errors2, axis=0), yerr=np.std(errors2, axis=0), capsize=5,
                width=width, color="#4bb652", label='Pick-and-Place')

    # plt.legend()
    # plt.title('Mean Squared Error and Standard Deviation of Estimates from \n Data with Gaussian Noise in State Measurements ($\sigma=0.01$) but \n'
    #           'Accelerations Calculated from Smooth State Measurements', fontsize=18)
    plt.title('Mean Squared Error and Standard Deviation of Estimates\nwith Added Gaussian Noise in Torque Observation, $\sigma=0.01$')
    plt.yscale('log')
    plt.xticks(np.arange(errors.shape[1]),
               ['$m$', '$mc_x$', '$mc_y$', '$mc_z$', '$I_{xx}$', '$I_{yy}$', '$I_{zz}$', '$I_{xy}$', '$I_{xz}$',
                '$I_{yz}$'])
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.savefig(f'/home/aalamber/Pictures/Thesis/{save}', dpi=300, bbox_inches="tight")


def model_predicted_torques(filename_meas, filename_mp):
    rcParams['axes.labelpad'] = 7

    joint_colors = [COLORS[color] for color in ["purple", "blue", "cyan", "green", "yellow", "orange", "red"]]

    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    fig.tight_layout(pad=4)
    # fig.suptitle("Example Joint Trajectory", fontsize=24)
    ax[0].set_title('Match Between Measured Torques and Model-Predicted Torques \n from Panda Estimate')

    # plt.figure(figsize=(6, 3))
    mp = np.load(filename_mp, allow_pickle=True)
    meas = np.load(filename_meas, allow_pickle=True)[:mp.shape[0]]

    print("L2 Loss:, ", np.linalg.norm(mp.flatten() - meas.flatten()))

    l2 = np.linalg.norm(mp.flatten() - meas.flatten())
    ax[1].set_title(f"Delta Between Measured Torques and Model-Predicted Torques \n L2 Loss = {np.round(l2, 3)}")
    print(meas.shape)
    print(mp.shape)
    for i in range(7):
        c = joint_colors[i]
        ls = '-'

        if i == 0:
            ax[0].plot(meas[:, i], ls, color=c, linewidth=2, label=f'Measured Torque', alpha=0.5)

            plot, = ax[0].plot(mp[:, i], '--', color=c, linewidth=2, label=f'Model-Predicted Torque')
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')

        else:
            ax[0].plot(meas[:, i], ls, color=c, linewidth=2, alpha=0.5)

            plot, = ax[0].plot(mp[:, i], '--', color=c, linewidth=2)
            plot.set_dashes([1, 2])
            plot.set_dash_capstyle('round')
        ax[1].plot(meas[:, i] - mp[:, i], ls, color=c, linewidth=2, label=f'Joint {i + 1}')
        ax[0].set_xlabel('Time steps', fontsize=18)
        ax[0].set_ylabel('Torque (Nm)', fontsize=18)
        ax[1].set_xlabel('Time steps', fontsize=18)
        ax[1].set_ylabel('Torque (Nm)', fontsize=18)
        # plt.ylim(-3, 2.5)
        # plt.xlim(0, without_data['ts'][-1][0])
        # plt.yticks(np.linspace(-np.pi, np.pi, 5), ['$-\pi$', '$-\\frac{1}{\pi}$', '0', '$\\frac{1}{\pi}$', '$\pi$'])

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    ax[0].legend()
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
    plt.savefig(f'{PREFIX}/mp_torques.png', dpi=DPI, bbox_inches="tight")

if __name__ == '__main__':
    plt.style.use('src/calibration/2dof/test.mplstyle')
    # joint_angles('../plotting_data/pick_and_place_5-8.npy', 'pick_and_place_5-8.png')
    # all_joint_data('../plotting_data/pick_and_place_5-8.npy', 'pick_and_place_5-8.png')
    # filtered_data('../plotting_data/joint_data_with_obj.npy')
    # traj_alignment('src/calibration/plotting_data/joint_data_with_obj_unaligned.npy', 'src/calibration/plotting_data/joint_data_wout_obj_unaligned.npy')
    # traj_unalignment('src/calibration/plotting_data/joint_data_with_obj_unaligned.npy', 'src/calibration/plotting_data/joint_data_wout_obj_unaligned.npy')
    # mass_estimates('/home/aalamber/realworld_panda_stack/mass_values_5-8.npy')
    # model_predicted_torques('src/calibration/measured_tau.npy', 'src/calibration/model_predicted_tau.npy')

    # error_bars_traj_types(filename1='src/calibration/fitting_alphas_sinusoidal.npy', filename2='src/calibration/fitting_alphas_pick_and_place.npy')

    # error_bars_noise(filename1='src/calibration/fitting_alphas_noisy_state.npy', save='noisy_states.png')
    # error_bars_noise(filename1='src/calibration/fitting_alphas_smooth_acceleration.npy', save='smooth_accelerations.png')
    error_bars_noise(filename1='src/calibration/fitting_alphas_noisy_tau.npy', save='noisy_tau_1.png')
    # error_bars_noise(filename1='src/calibration/fitting_alphas_noisy_tau2.npy', save='noisy_tau_2.png')


    # traj_unalignment('../plotting