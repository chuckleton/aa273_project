import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from tqdm import tqdm

import transformations
from thruster import Thruster
from spacecraft import Spacecraft
from dualEKF import DualEKF, EKFState, DualEKFState

initial_orientation = np.array([35.0, -30.0, 15.0])
R0 = Rot.from_euler('xyz', initial_orientation)
q0 = R0.as_quat()
omega0 = np.zeros(3)

J = np.eye(3)

C = np.eye(6)
# C[:3, :3] = np.zeros((3, 3))

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
    thrusts[3] = thrusts[0]
    # thrusts = np.append(thrusts, thrusts)

    return thrusts

def u_direct(x, t, T):
    tr = T-t
    P = np.block([[12/tr**3*np.eye(3), 6/tr**2*np.eye(3)], [6/tr**2*np.eye(3), 4/tr*np.eye(3)]])
    B = np.block([[np.zeros((3,3))], [np.eye(3)]])
    u = -B.T@P@x
    return u

delta_t = 0.0025
T = 5.0
ts = np.linspace(0, T, int(T/delta_t))


def theta(t):
    thetas = np.array([0.7, 1.35, 1.0, 1.0])
    thetas = thetas - 0.1*np.array([1.7*np.cos(t/12), 1.5, 0.2, -2.3*np.sin(t/5+np.pi/4)])*(t/T)
    # thetas = np.append(thetas, thetas)
    return thetas


# def theta(t):
#     thetas = np.array([0.8, 1.35, 1.0, 0.9])
#     thetas = np.array([1.0, 1.0, 1.0, 1.0])
#     thetas = thetas - 0.1 * \
#         np.array([0.6*np.cos(t/12), 0.75, 0.2, -1.0*np.sin(t/7+np.pi/4)])*(t/T)
#     if t > 15:
#         thetas[0] = 0.0
#     return thetas

# control_profile = np.empty((len(ts), len(u(0))))
theta_profile = np.empty((len(ts), len(theta(0))))
# for i, t in enumerate(ts):
#     control_profile[i] = u(t)
#     theta_profile[i] = theta(t)

control_profile = np.empty((len(ts), 3))


# plt.plot(ts, theta_profile)
# plt.show()

thruster_positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
# thruster_positions = np.vstack((thruster_positions, thruster_positions))
thruster_orientations = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# thruster_orientations = np.vstack((thruster_orientations, -thruster_orientations))
thruster_thrusts = np.ones(thruster_positions.shape[0])

thrusters = [Thruster(pos, ori, thrust) for pos, ori, thrust in zip(thruster_positions, thruster_orientations, thruster_thrusts)]
spacecraft = Spacecraft(q0, np.zeros(3), J, thrusters, C)

q = np.empty((len(ts), 4))
omega = np.empty((len(ts), 3))

s = np.empty((len(ts), 6))
mu_s = np.empty((len(ts), 6))
sigma_s = np.empty((len(ts), 6, 6))
mu_p = np.empty((len(ts), len(thrusters)))
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
Qs[3:,3:] = np.eye(3)*0.02
Qp = np.eye(len(thrusters))*0.0000001
R = np.eye(6)*0.0005

DEKF_initial_state = DualEKFState(EKFState(x_tilde0, 0.01*np.eye(6)), EKFState(np.ones(len(thrusters)), 0.1*np.eye(len(thrusters))))
DEKF = DualEKF(DEKF_initial_state, J, R, Qs, Qp, spacecraft.T)

s[0] = DEKF.state.s.mu
mu_s[0] = DEKF.state.s.mu
sigma_s[0] = DEKF.state.s.sigma
mu_p[0] = DEKF.state.p.mu
sigma_p[0] = DEKF.state.p.sigma
q_base.append(DEKF.q_base)

for i, t in enumerate(tqdm(ts[:-1])):
    # measurements[i+1] = spacecraft.step(u(t), theta(t), Qs, R, delta_t)
    control_profile[i] = u_direct(measurements[i], t, T+0.01)
    control_profile[i] = np.clip(control_profile[i], -1.0, 1.0)
    measurements[i+1] = spacecraft.step_direct(control_profile[i], delta_t)
    state = spacecraft.get_true_state()
    q[i+1] = state[0]
    omega[i+1] = state[1]
    s[i+1] = spacecraft.body.x_tilde
    s[i+1, :3] = q[i+1,:3]

    # DEKF.step(measurements[i+1], u(t), delta_t)
    # mu_s[i+1] = DEKF.state.s.mu
    # sigma_s[i+1] = DEKF.state.s.sigma
    # mu_p[i+1] = DEKF.state.p.mu
    # sigma_p[i+1] = DEKF.state.p.sigma
    # q_base.append(DEKF.q_base)

    # mu_q[i+1] = DEKF.q_base.as_quat()
    # mu_omega[i+1] = transformations.inverse_x_transformation(mu_s[i+1])[3:]
    # mu_s[i+1, :3] = mu_q[i+1,:3]



euler = Rot.from_quat(q).as_euler('xyz', degrees=False)

plt.style.use('seaborn-whitegrid')
# plt.plot(ts, q)
# plt.plot(ts, mu_q, '--')
# sigma_xs = [np.sqrt(sigma_s[:, i, i]) for i in range(6)]
# sigma_ps = [np.sqrt(sigma_p[:,i,i]) for i in range(len(thrusters))]
# fig, ax = plt.subplots(1, 1, figsize=(15, 8))
# for i in range(len(thrusters)):
#     color = next(ax._get_lines.prop_cycler)['color']
#     ax.plot(ts, theta_profile[:,i], '--', color=color)
#     ax.plot(ts, mu_p[:,i], label='Thruster {}'.format(i), color=color)
#     ax.fill_between(ts, mu_p[:,i]-sigma_ps[i], mu_p[:,i]+sigma_ps[i], alpha=0.2, color=color)
# ax.legend(fontsize=14)
# ax.set_xlabel('Time [s]', fontsize=14)
# ax.set_ylabel('$\\theta$', fontsize=14)
# ax.set_title('Parameter Estimation: Unobservable Parameters (Thrusters 1 and 4 equal control and parallel $\\vec{r}\\times \\vec{F}$)', fontsize=17)

fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
ax = axs[1]
for i in range(3,6):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, omega[:, i-3], label=f'$\\omega_{i-2}$', color=color)
    # ax.plot(ts, mu_s[:, i], label=f'$\\hat{{\\tilde{{\\omega}}}}_{i-2}$', color=color, linestyle='--')
    # ax.fill_between(ts, mu_s[:, i]-sigma_xs[i],
    #                 mu_s[:, i]+sigma_xs[i], alpha=0.2, color=color)
ax.legend(loc='upper left', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=14)

ax = axs[0]
for i in range(0,3):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, q[:, i], label=f'$q_{i+1}$', color=color)
    # ax.plot(ts, mu_s[:, i], label=f'$\\hat{{q}}_{i+1}$', color=color, linestyle='--')
    # ax.fill_between(ts, mu_s[:, i]-sigma_xs[i],
    #                 mu_s[:, i]+sigma_xs[i], alpha=0.2, color=color)
ax.legend(loc='upper left', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$q$', fontsize=14)
ax.set_title('Simplified Model Optimal Control with Control Limitation', fontsize=17)

# ax = axs[2]
# for i in range(len(thrusters)):
#     color = next(ax._get_lines.prop_cycler)['color']
#     ax.plot(
#         ts, control_profile[:, i], label=f'$u_{i+1}$', color=color)
# ax.legend(loc='upper right')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('$u(t)$')
# ax.set_title('Control Inputs')

ax = axs[2]
for i in range(3):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, control_profile[:, i], label=f'$v_{i+1}$', color=color)
ax.legend(loc='upper left', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$v(t)$', fontsize=14)
ax.set_title('Control Inputs $(|v_{{max}}|)$ = 1', fontsize=17)

plt.tight_layout()
plt.show()
