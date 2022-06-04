import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt

import transformations
from thruster import Thruster
from spacecraft import Spacecraft
from dualEKF import DualEKF, EKFState, DualEKFState
from EKF import EKF
from IMM import IMM

from tqdm import tqdm

initial_orientation = np.array([0.0, 0.0, 0.0])
R0 = Rot.from_euler('xyz', initial_orientation)
q0 = R0.as_quat()
omega0 = np.zeros(3)

J = np.eye(3)

C = np.eye(6)
C[:3, :3] = np.zeros((3,3))


def u(t):
    profiles = np.array([np.cos(3*t)*np.ones(4), np.sin(2*t)*np.ones(4),
                         np.cos(2.5*(t + np.pi/4))*np.ones(4), np.cos(7*(t+np.pi/6))*np.ones(4)])
    thrusts = np.zeros(4)
    selected_profiles = [[1, 2, 3, 4], [1, 3], [2, 4], [1, 2, 3]]
    selected_profiles = [np.array(x, dtype=int)-1 for x in selected_profiles]
    for i, thrust in enumerate(thrusts):
        thrusts[i] = np.sum(profiles[selected_profiles[i]])
        if thrusts[i] < 0:
            thrusts[i] = -1
        else:
            thrusts[i] = 1
    thrusts = np.clip(thrusts, -1, 1)
    # profiles = np.array([np.cos(t), np.sin(t),
    #                      np.cos((t + np.pi/4)), np.cos((t+np.pi/4))])
    # thrusts = np.append(thrusts, thrusts)

    return thrusts


delta_t = 0.005
T = 30.0
ts = np.linspace(0, T, int(T/delta_t))
t_failure = 15.0
t_failure2 = 22.0


# def theta(t):
#     thetas = np.array([0.7, 1.35, 1.0, 1.0])
#     thetas = thetas - 0.1 * \
#         np.array([1.7*np.cos(t/12), 1.5, 0.2, -2.3*np.sin(t/5+np.pi/4)])*(t/T)
#     # thetas = np.append(thetas, thetas)
#     return thetas

def theta(t):
    # thetas = np.array([0.8, 1.35, 1.0, 0.9])
    thetas = np.array([1.0, 1.0, 1.0, 1.0])
    thetas = thetas - 0.1 * \
        np.array([0.6*np.cos(t/12), 0.75, 0.2, -1.0*np.sin(t/7+np.pi/4)])*(t/T)
    if t > t_failure:
        thetas[0] = 0.0
    if t > t_failure2:
        thetas[2] = 0.0
    return thetas

control_profile = np.empty((len(ts), len(u(0))))
theta_profile = np.empty((len(ts), len(theta(0))))
for i, t in enumerate(ts):
    control_profile[i] = u(t)
    theta_profile[i] = theta(t)

thruster_positions = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
thruster_orientations = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
thruster_thrusts = np.array([1.0, 1.0, 1.0, 1.0])

thrusters = [Thruster(pos, ori, thrust) for pos, ori, thrust in zip(
    thruster_positions, thruster_orientations, thruster_thrusts)]
spacecraft = Spacecraft(q0, np.zeros(3), J, thrusters, C)

q = np.empty((len(ts), 4))
omega = np.empty((len(ts), 3))

mu_s = np.empty((len(ts), 6))
sigma_s = np.empty((len(ts), 6, 6))
mu_p = np.empty((len(ts), len(thrusters)))
mu_pDEKF = np.empty((len(ts), len(thrusters)))
sigma_p = np.empty((len(ts), len(thrusters), len(thrusters)))
q_base = []

mu_q = np.zeros_like(q)
mu_omega = np.zeros_like(omega)

measurements = np.zeros((len(ts), 6))
q[0] = q0
omega[0] = np.zeros(3)

x0 = np.concatenate((q0[:3], omega0))
x_tilde0 = transformations.forward_x_transformation(x0, J)

Qs = np.zeros((6, 6))
Qs[3:, 3:] = np.eye(3)*0.025
Qp = np.eye(len(thrusters))*0.0001
R = np.eye(6)*0.0001

DEKF_initial_state = DualEKFState(EKFState(
    x_tilde0, 0.01*np.eye(6)), EKFState(np.ones(len(thrusters)), 0.01*np.eye(len(thrusters))))
DEKF = DualEKF(DEKF_initial_state, J, R, Qs, Qp, spacecraft.T)

mu_s[0] = DEKF.state.s.mu
sigma_s[0] = DEKF.state.s.sigma
mu_p[0] = DEKF.state.p.mu
mu_pDEKF[0] = DEKF.state.p.mu
sigma_p[0] = DEKF.state.p.sigma
q_base.append(DEKF.q_base)

modes = [DEKF]
mode_zeros = [[]]
for i, t in enumerate(thrusters):
    theta_i = np.ones(len(thrusters))
    theta_i[i] = 0.0
    EKF_initial_state = EKFState(x_tilde0, 0.01*np.eye(6))
    EKF_i = EKF(EKF_initial_state, J, R, Qs, spacecraft.T, theta_i)
    modes.append(EKF_i)
    mode_zeros.append([i])

for i, t in enumerate(thrusters):
    for j in range(i+1, len(thrusters)):
        theta_i = np.ones(len(thrusters))
        theta_i[i] = 0.0
        theta_i[j] = 0.0
        EKF_initial_state = EKFState(x_tilde0, 0.01*np.eye(6))
        EKF_i = EKF(EKF_initial_state, J, R, Qs, spacecraft.T, theta_i)
        modes.append(EKF_i)
        mode_zeros.append([i, j])

pi = 0.9996*np.eye(len(modes))
for i in range(len(modes)):
    for j in range(len(modes)):
        if not i == j:
            pi[j, i] = (1.0 - pi[0, 0])/(len(modes)-1)
mu0 = np.zeros(len(modes))
mu0[0] = 1.0

imm = IMM(modes, pi, mu0, mode_zeros)

mu = np.zeros((len(ts), len(modes)))
mu[0] = mu0

for i, t in enumerate(tqdm(ts[:-1])):
    measurements[i+1] = spacecraft.step(u(t), theta(t), Qs, R, delta_t)
    state = spacecraft.get_true_state()
    q[i+1] = state[0]
    omega[i+1] = state[1]

    imm.step(measurements[i+1], u(t), delta_t)
    mu_s[i+1] = DEKF.state.s.mu
    sigma_s[i+1] = DEKF.state.s.sigma
    mu_pDEKF[i+1] = DEKF.state.p.mu
    mu_p[i+1] = imm.theta_overall
    sigma_p[i+1] = DEKF.state.p.sigma
    q_base.append(DEKF.q_base)
    mu[i+1] = imm.mu

    mu_q[i+1] = DEKF.q_base.as_quat()
    mu_omega[i+1] = transformations.inverse_x_transformation(mu_s[i+1])[3:]

change_point = ts[np.nonzero(np.where(mu[:, 1] - mu[:, 0] > 0, 1, 0))[0][0]]
change_point2 = ts[np.nonzero(
    np.where((mu[:, 5] - mu[:, 0] > 0) & ((mu[:, 5] - mu[:, 1] > 0)), 1, 0))[0][0]]

plt.style.use('seaborn-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
ax = axs[0]
ax.set_title(
    'IMM Fault Detection: Multiple Thruster Failure', fontsize=17)
ax.plot([change_point, change_point], [0, 1.1],
        'k--', label=f'Fault 1 Detected @ {change_point:.3f}s')
ax.plot([change_point2, change_point2], [0, 1.1],
        'k--', label=f'Fault 5 Detected @ {change_point2:.3f}s')
ax.plot([t_failure, t_failure], [0, 1.1], 'r--', label='Thruster 0 Fails @ 15s')
ax.plot([t_failure2, t_failure2], [0, 1.1], 'r--', label='Thruster 2 Fails @ 22s')
# ax.plot([20, 20], [0, 1.1], 'r--', label='Thruster 3 Fails @ 20s')
for i, mode in enumerate(modes):
    if i == 0:
        label = 'Mode 0: Nominal Operation'
    elif i == 1:
        label = f'Mode {i}: Thruster 0 Failure'
    elif i == 5:
        label = f'Mode {i}: Thrusters 0 and 2 Failure'
    else:
        label = ''
        # label = f'Mode {i}: Thruster {i-1} Failure'
    if i > 0:
        color = next(ax._get_lines.prop_cycler)['color']
    else:
        color = 'k'
    ax.plot(ts, mu[:,i], label=label, color=color)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('Mode Probability', fontsize=14)
ax.legend(fontsize=14)

sigma_ps = [np.sqrt(sigma_p[:, i, i]) for i in range(len(thrusters))]

ax = axs[1]
ax.set_title(
    'Parameter Estimation', fontsize=17)
for i in range(len(thrusters)):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(ts, theta_profile[:, i], '--', color=color)
    ax.plot(ts, mu_p[:, i], label='Thruster {}'.format(i), color=color)
    ax.fill_between(ts, mu_p[:, i]-sigma_ps[i],
                    mu_p[:, i]+sigma_ps[i], alpha=0.2, color=color)
ax.plot([t_failure, t_failure], [0, 1.075], 'r--', label='Thruster 0 Fails @ 15s')
ax.plot([t_failure2, t_failure2], [0, 1.075],
        'r--', label='Thruster 2 Fails @ 22s')
ax.legend(fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\theta$', fontsize=14)

plt.tight_layout()
plt.show()