import numpy as np

class IMM:
    def __init__(self, modes, pi, mu0, mode_zeros):
        self.modes = modes
        self.pi = pi
        self.mu0 = mu0
        self.mu = mu0

        self.N = len(modes)

        self.get_x_hat()
        self.get_P()

        self.nu = np.zeros((self.N, len(self.modes[0].state.mu)))
        self.S = np.zeros((self.N, len(
            self.modes[0].state.mu), len(self.modes[0].state.mu)))

        self.mode_zeros = mode_zeros

    def get_x_hat(self):
        self.x_hat = np.zeros((self.N, len(self.modes[0].state.mu)))
        for i in range(self.N):
            self.x_hat[i] = self.modes[i].state.mu

    def get_P(self):
        self.P = np.zeros((self.N, len(
            self.modes[0].state.mu), len(self.modes[0].state.mu)))
        for i in range(self.N):
            self.P[i] = self.modes[i].state.sigma

    def get_mu_hat(self):
        self.mu_hat = np.zeros_like(self.mu)
        for j in range(self.N):
            for i in range(self.N):
                self.mu_hat[j] += self.pi[i, j]*self.mu[i]

    def mixing_probability(self):
        self.mu_ij = np.zeros((self.N, self.N))
        for j in range(self.N):
            for i in range(self.N):
                self.mu_ij[i, j] = self.pi[i, j]*self.mu[i]/self.mu_hat[j]

    def mixing_estimate(self):
        self.x_hat_0 = np.zeros_like(self.x_hat)
        for j in range(self.N):
            for i in range(self.N):
                self.x_hat_0[j] += self.x_hat[i]*self.mu_ij[i, j]

    def mixing_cov(self):
        P_0 = []
        for j in range(self.N):
            P_0_i = np.zeros_like(self.P[0])
            for i in range(self.N):
                P_0_i += (self.P[i] + (self.x_hat_0[i] - self.x_hat[i])
                           @ (self.x_hat_0[i] - self.x_hat[i]).T)*self.mu_ij[i, j]
            P_0.append(P_0_i)
        self.P_0 = P_0

    def predict(self, u, delta_t):
        for i, mode in enumerate(self.modes):
            mode.state.mu = self.x_hat_0[i]
            mode.state.sigma = self.P_0[i]
            mode.predict(u, delta_t)

    def update(self, y, u, delta_t):
        for i, mode in enumerate(self.modes):
            self.nu[i], self.S[i] = mode.update(y, u, delta_t)
            self.get_x_hat()
            self.get_P()

    def likelihood(self):
        self.L = np.zeros(self.N)
        alpha = 0.25
        for j in range(self.N):
            # self.L[j] = (1/np.sqrt(np.linalg.det(2*np.pi*self.S[j])))*\
            #                 np.exp(-0.5*self.nu[j].T@np.linalg.inv(self.S[j])@self.nu[j])
            self.L[j] = (1/((2*np.pi)**(3)*np.sqrt(np.linalg.det(self.S[j])))) * \
                            np.exp(-alpha*0.5*self.nu[j].T@np.linalg.inv(self.S[j])@self.nu[j])

    def update_mu(self):
        mu_new = np.zeros_like(self.mu)
        mu_new = self.mu_hat*self.L
        mu_new = mu_new/np.sum(mu_new)
        self.mu = mu_new

    def update_theta(self):
        DEKF_theta = self.modes[0].state.p.mu
        for i, mode in enumerate(self.modes):
            if i > 0:
                mode.theta = DEKF_theta.copy()
                for j in self.mode_zeros[i]:
                    mode.theta[j] = 0.0
        # self.modes[0].state.p.mu = self.theta_overall.copy()


    def overall_estimate(self):
        self.x_hat_overall = np.zeros_like(self.x_hat[0])
        self.P_overall = np.zeros_like(self.P[0])
        self.theta_overall = self.modes[0].state.p.mu.copy()*self.mu[0]
        for j in range(self.N):
            self.x_hat_overall += self.mu[j]*self.x_hat[j]
            if j != 0:
                self.theta_overall += self.mu[j]*self.modes[j].theta
        for j in range(self.N):
            self.P_overall += self.mu[j]*(self.P[j] + (
                self.x_hat_overall-self.x_hat[j])@(self.x_hat_overall-self.x_hat[j]).T)
        return self.x_hat_overall, self.P_overall

    def interaction_and_mixing(self):
        self.get_mu_hat()
        self.mixing_probability()
        self.mixing_estimate()
        self.mixing_cov()

    def correction(self):
        self.likelihood()
        self.update_mu()
        self.overall_estimate()
        self.update_theta()

    def step(self, y, u, delta_t):
        self.interaction_and_mixing()
        self.predict(u, delta_t)
        self.update(y, u, delta_t)
        self.correction()