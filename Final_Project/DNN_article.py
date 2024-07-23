import numpy as np
import matplotlib.pyplot as plt
from Models import RR_Robot_Model 
## Define parameters
alpha = 7
k1 = 10.0
k2 = 5.0
k3 = 1.0
k4 =0.2

Gama_M = 20
Gama_V = 50.8
Gama_G = 20.0
Gama_F = 1.0

n = 2
I_n = np.eye(n)

# Desired trajectory q_d(t)
A = 3 * np.pi / 8
omega = np.pi / 2

dt = 0.01
T = 30
num_steps = int(T / dt)
tt = np.linspace(0, T, num_steps + 1)

q_d_ = (1 - np.exp(-0.1 * tt)) * np.array([A * np.sin(omega * tt) + 1.5, A * np.cos(0.5 * omega * tt) - 1.5])
qd_d_ = np.gradient(q_d_, dt, axis=1)
qdd_d_ = np.gradient(qd_d_, dt, axis=1)

# System initial states
q = np.array([0.4, -0.3])
qd = np.array([0.0, 0.0])

input_size_M = 2
hidden_size_M = 4
output_size_M = 4

input_size_V = 4
hidden_size_V = 4
output_size_V = 4

input_size_G = 2
hidden_size_G = 4
output_size_G = 2

input_size_F = 2
hidden_size_F = 4
output_size_F = 2

# Initialize parameters
def initialize_params(input_size, hidden_size, output_size, bc, bh):
    params = {}
    params['Wf'] = np.random.rand(input_size + hidden_size + 1, hidden_size)
    params['Wp'] = np.random.rand(input_size + hidden_size + 1, hidden_size)
    params['Wc'] = np.random.rand(input_size + hidden_size + 1, hidden_size)
    params['Wo'] = np.random.rand(input_size + hidden_size + 1, hidden_size)
    params['Wh'] = np.random.rand(hidden_size, output_size)
    params['bc_i'] = bc
    params['bh_i'] = bh
    params['input_size'] = input_size
    params['hidden_size'] = hidden_size
    params['output_size'] = output_size

    # DNN Params
    LayerSizes = [input_size + 1, 7, 7, 7, output_size]
    L = len(LayerSizes)
    weightsSizes = [LayerSizes[:-1], LayerSizes[1:]]
    weightsLenght = np.prod(weightsSizes, axis=0)
    weights = [np.random.rand(weightsSizes[0][i], weightsSizes[1][i]) * 0.001 for i in range(L-1)]
    
    params['L'] = L
    params['LayerSizes'] = LayerSizes
    params['weightsSizes'] = weightsSizes
    params['weightsLenght'] = weightsLenght
    params['weights'] = weights

    return params

def lstm_cell(x, c, h, params):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # sigma_g
    tanh_func = np.tanh  # sigma_c

    sigmoid_prime = lambda x: np.diag(sigmoid(x) * (1 - sigmoid(x)))
    tanh_prime = lambda x: np.diag(1 - tanh_func(x) ** 2)

    Wf = params["Wf"]
    Wp = params["Wp"]
    Wc = params["Wc"]
    Wo = params["Wo"]
    Wh = params["Wh"]
    bc_i = params["bc_i"]
    bh_i = params["bh_i"]

    zi = np.concatenate((x, h, [1]))

    f = sigmoid(Wf.T @ zi)
    p = sigmoid(Wp.T @ zi)
    o = sigmoid(Wo.T @ zi)
    c_tilde = tanh_func(Wc.T @ zi)

    Psi_c = f * c + p * c_tilde
    Psi_h = o * tanh_func(Psi_c)

    dc = -bc_i * c + bc_i * Psi_c
    dh = -bh_i * h + bh_i * Psi_h
    output = Wh.T @ Psi_h

    l2 = len(h)
    l_n = params["output_size"]

    Psi_prime_c_Wc = np.diag(sigmoid(Wp.T @ zi)) @ tanh_prime(Wc.T @ zi) @ np.kron(np.eye(l2), zi[:, np.newaxis].T)
    Psi_prime_c_Wp = np.diag(sigmoid(Wc.T @ zi)) @ sigmoid_prime(Wp.T @ zi) @ np.kron(np.eye(l2), zi[:, np.newaxis].T)
    Psi_prime_c_Wf = np.diag(c_tilde) @ sigmoid_prime(Wf.T @ zi) @ np.kron(np.eye(l2), zi[:, np.newaxis].T)

    Psi_prime_h_Wo = np.diag(tanh_func(Psi_c)) @ sigmoid_prime(Wo.T @ zi) @ np.kron(np.eye(l2), zi[:, np.newaxis].T)

    Psi_prime_h_Wc = np.diag(sigmoid(Wo.T @ zi)) @ tanh_prime(Psi_c) @ Psi_prime_c_Wc
    Psi_prime_h_Wp = np.diag(sigmoid(Wo.T @ zi)) @ tanh_prime(Psi_c) @ Psi_prime_c_Wp
    Psi_prime_h_Wf = np.diag(sigmoid(Wo.T @ zi)) @ tanh_prime(Psi_c) @ Psi_prime_c_Wf

    Theta_hat_prim_c = Wh.T @ Psi_prime_h_Wc
    Theta_hat_prim_p = Wh.T @ Psi_prime_h_Wp
    Theta_hat_prim_f = Wh.T @ Psi_prime_h_Wf
    Theta_hat_prim_o = Wh.T @ Psi_prime_h_Wo

    Theta_hat_prim_h = np.kron(np.eye(l_n), Psi_h[:, np.newaxis].T)
    Theta_hat_prim = np.hstack((Theta_hat_prim_c, Theta_hat_prim_p, Theta_hat_prim_f, Theta_hat_prim_o, Theta_hat_prim_h))

    return dc, dh, output, Theta_hat_prim

def vec2param_lstm(v, params):
    Wc_len = np.prod(params['Wc'].shape)
    Wp_len = np.prod(params['Wp'].shape)
    Wf_len = np.prod(params['Wf'].shape)
    Wo_len = np.prod(params['Wo'].shape)
    Wh_len = np.prod(params['Wh'].shape)

    Wc = v[:Wc_len].reshape(params['Wc'].shape)
    v = v[Wc_len:]

    Wp = v[:Wp_len].reshape(params['Wp'].shape)
    v = v[Wp_len:]

    Wf = v[:Wf_len].reshape(params['Wf'].shape)
    v = v[Wf_len:]

    Wo = v[:Wo_len].reshape(params['Wo'].shape)
    v = v[Wo_len:]

    Wh = v[:Wh_len].reshape(params['Wh'].shape)

    return Wc, Wp, Wf, Wo, Wh

import numpy as np

def dnn(DataInput, params):
    weights = params['weights']
    LayerSizes = params['LayerSizes']
    L = params['L']

    Phi = [None] * (L-1)
    Phi[0] = weights[0].T @ np.hstack((DataInput, 1))  # x_a

    for j in range(1, L-1):
        Phi[j] = weights[j].T @ np.tanh(Phi[j-1])

    khi = Phi[-1]

    phi_prime = [None] * (L-2)
    for l in range(L-2):
        phi_prime[l] = np.diag(1 - np.square(Phi[l]))

    Phi_prime = [None] * (L-1)
    Phi_prime[3] = np.kron(np.eye(LayerSizes[4]), np.tanh(Phi[2]).T)
    Phi_prime[2] = weights[3].T @ (phi_prime[2] @ np.kron(np.eye(LayerSizes[3]), np.tanh(Phi[1]).T))
    Phi_prime[1] = weights[3].T @ (phi_prime[2] @ (weights[2].T @ (phi_prime[1] @ np.kron(np.eye(LayerSizes[2]), np.tanh(Phi[0]).T))))
    Phi_prime[0] = weights[3].T @ (phi_prime[2] @ (weights[2].T @ (phi_prime[1] @ (weights[1].T @ (phi_prime[0] @ np.kron(np.eye(LayerSizes[1]), np.hstack((DataInput, 1)).T))))))

    Phi_prime = np.hstack(Phi_prime)
    return khi, Phi_prime

def vec2param_dnn(theta, params):
    L = params['L']
    weightsSizes = params['weightsSizes']
    weightsLenght = params['weightsLenght']
    weights = params['weights']
    startpoint = 0

    for j in range(L - 1):
        endpoint = startpoint + weightsLenght[j]
        weights[j] = theta[startpoint:endpoint].reshape(weightsSizes[1][j], weightsSizes[0][j]).T
        startpoint = endpoint

    return weights


def dnn_param2vec(params):
    w = params['weights']
    theta = []
    for wi in w:
        theta.extend(wi.T.flatten())
    return np.array(theta)

def vec(m):
    return m.T.flatten()


# Initialize parameters
params_M = initialize_params(input_size_M, hidden_size_M, output_size_M, 5.9, 2.7)
params_V = initialize_params(input_size_V, hidden_size_V, output_size_V, 7.6, 7.7)
params_F = initialize_params(input_size_F, hidden_size_F, output_size_F, 1.6, 2)
params_G = initialize_params(input_size_G, hidden_size_G, output_size_G, 1, 1)

c_F = np.random.rand(hidden_size_F, num_steps+1)
h_F = np.random.rand(hidden_size_F, num_steps+1)

q_vec = np.zeros((num_steps, 2))
qd_vec = np.zeros((num_steps, 2))
u_vec = np.zeros((num_steps, 2))
output_M_List = np.zeros((num_steps, output_size_M))
model=RR_Robot_Model()
# Run simulation with Backward Euler integration
for k in range(1, num_steps+1):
    t = k * dt

    q_d = q_d_[:, k]
    qd_d = qd_d_[:, k]
    qdd_d = qdd_d_[:, k]

    #c_prev_F = c_F[:, k-1]
    #h_prev_F = h_F[:, k-1]

    # Compute derivatives and Jacobians
    x0_M = q
    x0_V = np.hstack((q, qd))
    x0_G = q
    x0_F = qd

    #dc_F, dh_F, lstm_output_F, Theta_hat_prim_F = lstm_cell(x0_F, c_prev_F, h_prev_F, params_F)

    dnn_output_M, Phi_hat_prim_M = dnn(x0_M, params_M)
    dnn_output_V, Phi_hat_prim_V = dnn(x0_V, params_V)
    dnn_output_G, Phi_hat_prim_G = dnn(x0_G, params_G)
    dnn_output_F, Phi_hat_prim_F = dnn(x0_F, params_F)

    # Errors
    e = q - q_d
    e_dot = qd - qd_d

    # Auxiliary variables
    r = e_dot + alpha * e

    # Estimates for dynamics (example values)
    hat_Xi_V = dnn_output_V 
    hat_Xi_G = dnn_output_G 
    hat_Xi_F = dnn_output_F 
    hat_Xi_M = dnn_output_M 

    # Compute control law
    tau = (np.kron((qd_d - alpha * e).T, I_n) @ hat_Xi_V + hat_Xi_G + hat_Xi_F
           - k1 * r - e
           + np.kron((qdd_d - alpha * e_dot).T, I_n) @ hat_Xi_M
           - np.sign(r) * (k2 + k3 * np.linalg.norm(np.kron((qd_d - alpha * e).T, I_n))
                           + k4 * np.linalg.norm(np.kron((qdd_d - alpha * e_dot).T, I_n))))

    # Adaptation Law
    Xi_M_prim = Phi_hat_prim_M
    Xi_V_prim =  Phi_hat_prim_V
    Xi_G_prim =Phi_hat_prim_G
    Xi_F_prim = Phi_hat_prim_F


    temp1=(np.kron((qdd_d - alpha * e_dot).T, I_n))
    temp2=(np.kron((qd_d - alpha * e).T, I_n))
    Z_M_dot = -Gama_M * Xi_M_prim.T @ temp1.T @ r
    Z_V_dot = -Gama_V * Xi_V_prim.T @ temp2.T @ r
    Z_G_dot = -Gama_G * Xi_G_prim.T @ r
    Z_F_dot = -Gama_F * Xi_F_prim.T @ r

    vv_M = np.hstack([vec(params_M['Wc']), vec(params_M['Wp']), vec(params_M['Wf']), vec(params_M['Wo']), vec(params_M['Wh'])])
    theta_M = dnn_param2vec(params_M)
    Z_M =  theta_M

    vv_V = np.hstack([vec(params_V['Wc']), vec(params_V['Wp']), vec(params_V['Wf']), vec(params_V['Wo']), vec(params_V['Wh'])])
    theta_V = dnn_param2vec(params_V)
    Z_V = theta_V

    vv_G = np.hstack([vec(params_G['Wc']), vec(params_G['Wp']), vec(params_G['Wf']), vec(params_G['Wo']), vec(params_G['Wh'])])
    theta_G = dnn_param2vec(params_G)
    Z_G =theta_G

    vv_F = np.hstack([vec(params_F['Wc']), vec(params_F['Wp']), vec(params_F['Wf']), vec(params_F['Wo']), vec(params_F['Wh'])])
    theta_F = dnn_param2vec(params_F)
    Z_F = theta_F

    # Update parameters
    Z_M += dt * Z_M_dot
    Z_V += dt * Z_V_dot
    Z_G += dt * Z_G_dot
    Z_F += dt * Z_F_dot

    params_M['weights'] = vec2param_dnn(Z_M, params_M)
    params_G['weights'] = vec2param_dnn(Z_G, params_G)
    params_V['weights'] = vec2param_dnn(Z_V, params_V)

    Wc, Wp, Wf, Wo, Wh = vec2param_lstm(Z_F, params_F)
    params_F['Wc'], params_F['Wp'], params_F['Wf'], params_F['Wo'], params_F['Wh'] = Wc, Wp, Wf, Wo, Wh

    # Dynamics Update
    qdd=model.dynamic_model(tau, q, qd, t)

    #c_F[:, k] = dc_F * dt + c_prev_F
    #h_F[:, k] = dh_F * dt + h_prev_F

    qd += dt * qdd
    q += dt * qd
    # Save data
    q_vec[k - 1, :] = q
    qd_vec[k - 1, :] = qd
    u_vec[k - 1, :] = tau

# Plot Joint Trajectories
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps) * dt, q_vec[:, 0], label='Joint 1')
plt.plot(np.arange(num_steps) * dt, q_vec[:, 1], label='Joint 2')
plt.plot(np.arange(num_steps+1 ) * dt, q_d_[0], 'k--', label='Desired Joint 1')
plt.plot(np.arange(num_steps+1) * dt, q_d_[1], 'r--', label='Desired Joint 2')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.title('Joints Trajectory')
plt.legend()
plt.grid(True)

# Plot Joint Torques
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps) * dt, u_vec[:, 0], label='Torque 1')
plt.plot(np.arange(num_steps) * dt, u_vec[:, 1], label='Torque 2')
plt.xlabel('Time (s)')
plt.ylabel('Torques (N.m)')
plt.title('Joint Torques')
plt.legend()
plt.grid(True)

plt.show()

Z_M_dot