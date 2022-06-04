import numpy as np

def forward_transformation(x, u, J):
    q = x[:3]
    q0 = np.sqrt(1-np.dot(q,q))
    omega = x[3:]
    q_tilde = q
    Omega = J@omega

    omega_tilde = 0.5*(q0*Omega + np.cross(q, Omega))
    x_tilde = np.concatenate((q_tilde, omega_tilde))

    omega_dot = np.linalg.inv(J)@(u-np.cross(omega, Omega))
    u_tilde = 0.5*(q0*omega_dot+np.cross(q, omega_dot)) - 0.25*np.dot(omega, omega)*q
    return x_tilde, u_tilde

def forward_x_transformation(x, J):
    q = x[:3]
    q0 = np.sqrt(1-np.dot(q, q))
    omega = x[3:]
    q_tilde = q
    Omega = J@omega

    omega_tilde = 0.5*(q0*Omega + np.cross(q, Omega))
    x_tilde = np.concatenate((q_tilde, omega_tilde))
    return x_tilde

def forward_u_transformation(x_tilde, u, J):
    x = inverse_x_transformation(x_tilde)
    u_tilde = forward_transformation(x, u, J)[1]
    return u_tilde

def inverse_x_transformation(x_tilde):
    x1 = x_tilde[:3]
    x0 = np.sqrt(1-np.dot(x1, x1))
    x2 = x_tilde[3:]

    omega = (2/x0)*(x0**2*x2 + x1*np.dot(x1, x2) - x0*np.cross(x1, x2))
    x = np.concatenate((x1, omega))
    return x

def inverse_transformation(x_tilde, u_tilde, J):
    x1 = x_tilde[:3]
    x0 = np.sqrt(1-np.dot(x1,x1))
    x2 = x_tilde[3:]

    omega = (2/x0)*(x0**2*x2 + x1*np.dot(x1, x2) - x0*np.cross(x1, x2))
    x = np.concatenate((x1, omega))

    Omega = J@omega
    mu = u_tilde + 0.25*(omega.T@omega@x1)
    u = np.cross(omega, Omega) + (2/x0)*J@(x0**2*mu + x1@np.dot(x1, mu) - x0*np.cross(x1, mu))
    return x, u