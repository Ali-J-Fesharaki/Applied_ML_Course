import numpy as np

class RR_Robot_Model:
    def __init__(self):
        self.m = np.array([1, 1])  # You can replace these with np.random.randn(2) if needed
        self.l = np.array([1, 1])  # You can replace these with np.random.randn(2) if needed
        self.g = 9.8

    def dynamic_model(self, tau, q, dq, t):
        m = self.m
        l = self.l
        g = self.g

        M = np.zeros((2, 2))
        C = np.zeros(2)
        G = np.zeros(2)

        M[0, 0] = np.sum(m) * l[0]**2
        M[0, 1] = m[1] * np.prod(l) * np.cos(q[0] - q[1])
        M[1, 0] = M[0, 1]
        M[1, 1] = m[1] * l[1]**2

        C[0] = dq[1]**2
        C[1] = -dq[0]**2
        C = m[1] * np.prod(l) * np.sin(q[0] - q[1]) * C

        G[0] = np.sum(m) * l[0] * np.cos(q[0])
        G[1] = m[1] * l[1] * np.cos(q[1])
        G = g * G

        F = 0.1 * np.eye(2) @ dq
        dist = np.zeros(2)
        if 10 < t < 20:
            pass
            #dist = np.array([2, 4])

        D = np.array([0])
        ddq = np.linalg.pinv(M) @(tau - C - G - F - dist - D * dq)

        return ddq

# # Example usage:
# model = RR_Robot_Model()
# tau = np.array([0.5, 0.3])
# q = np.array([0.1, 0.2])
# dq = np.array([0.01, 0.02])
# t = 15
# ddq = model.dynamic_model(tau, q, dq, t)
# print(ddq)
