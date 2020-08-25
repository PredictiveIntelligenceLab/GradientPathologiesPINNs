import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from Klein_Gordon_model_tf import Sampler, Klein_Gordon

if __name__ == '__main__':
    def u(x):
        """
        :param x: x = (t, x)
        """
        return x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2])**3

    def u_tt(x):
        return - 25 * np.pi**2 * x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + 6 * x[:,0:1] * x[:,1:2]**3

    def u_xx(x):
        return np.zeros((x.shape[0], 1)) +  6 * x[:,1:2] * x[:,0:1]**3

    def f(x, alpha, beta, gamma, k):
        return u_tt(x) + alpha * u_xx(x) + beta * u(x) + gamma * u(x)**k

    def operator(u, t, x, alpha, beta, gamma, k,  sigma_t=1.0, sigma_x=1.0):
        u_t = tf.gradients(u, t)[0] / sigma_t
        u_x = tf.gradients(u, x)[0] / sigma_x
        u_tt = tf.gradients(u_t, t)[0] / sigma_t
        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
        return residual

    # Parameters of equations
    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    # Domain boundaries
    ics_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    # Create initial conditions samplers
    ics_sampler = Sampler(2, ics_coords, lambda x: u(x), name='Initial Condition 1')

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x), name='Dirichlet BC2')
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, alpha, beta, gamma, k), name='Forcing')

    # Define model
    layers = [2, 50, 50, 50, 50, 50, 1]
    mode = 'M1'          # Method: 'M1', 'M2', 'M3', 'M4'
    stiff_ratio = False  # Log the eigenvalues of Hessian of losses
    model = Klein_Gordon(layers, operator, ics_sampler, bcs_sampler, res_sampler, alpha, beta, gamma, k, mode, stiff_ratio)

    # Train model
    model.train(nIter=40001, batch_size=128)

    # Test data
    nn = 100
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    # Exact solution
    u_star = u(X_star)
    f_star = f(X_star, alpha, beta, gamma, k)

    # Predictions
    u_pred = model.predict_u(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 error_u: {:.2e}'.format(error_u))

    ### Plot ###

    # Test data
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    F_star = griddata(X_star, f_star.flatten(), (t, x), method='cubic')

    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')

    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t, x, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact u(x)')

    plt.subplot(1, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted u(x)')

    plt.subplot(1, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()

    # Loss
    loss_r = model.loss_r_log
    loss_u = model.loss_u_log

    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_r, label='$\mathcal{L}_{r}$')
    ax.plot(loss_u, label='$\mathcal{L}_{u}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Adaptive Constant
    adaptive_constant_ics = model.adaptive_constant_ics_log
    adaptive_constant_bcs = model.adaptive_constant_bcs_log

    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant_ics, label='$\lambda_{u_0}$')
    ax.plot(adaptive_constant_bcs, label='$\lambda_{u_b}$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gradients at the end of training
    data_gradients_ics = model.dict_gradients_ics_layers
    data_gradients_bcs = model.dict_gradients_bcs_layers
    data_gradients_res = model.dict_gradients_res_layers

    num_hidden_layers = len(layers) - 1
    cnt = 1
    fig_4 = plt.figure(4, figsize=(13, 8))
    for j in range(num_hidden_layers):
        ax = plt.subplot(2, 3, cnt)
        gradients_ics = data_gradients_ics['layer_' + str(j + 1)][-1]
        gradients_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]
        gradients_res = data_gradients_res['layer_' + str(j + 1)][-1]

        sns.distplot(gradients_ics, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \lambda_{u_0}\mathcal{L}_{u_0}$')
        sns.distplot(gradients_bcs, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \lambda_{u_b} \mathcal{L}_{u_b}$')
        sns.distplot(gradients_res, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \mathcal{L}_r$')

        ax.set_title('Layer {}'.format(j + 1))
        ax.set_yscale('symlog')
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 500])
        ax.get_legend().remove()
        cnt += 1
    handles, labels = ax.get_legend_handles_labels()
    fig_4.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.3, 0.01),
                 borderaxespad=0, bbox_transform=fig_4.transFigure, ncol=3)
    plt.tight_layout()
    plt.show()

    # Eigenvalues of Hessian of losses if applicable
    if stiff_ratio:
        eigenvalues_list = model.eigenvalue_log
        eigenvalues_ics_list = model.eigenvalue_ics_log
        eigenvalues_bcs_list = model.eigenvalue_bcs_log
        eigenvalues_res_list = model.eigenvalue_res_log

        eigenvalues_ics = eigenvalues_ics_list[-1]
        eigenvalues_bcs = eigenvalues_bcs_list[-1]
        eigenvalues_res = eigenvalues_res_list[-1]

        fig_5 = plt.figure(5)
        ax = fig_5.add_subplot(1, 1, 1)
        ax.plot(eigenvalues_ics, label='$\mathcal{L}_{u_0}$')
        ax.plot(eigenvalues_bcs, label='$\mathcal{L}_{u_b}$')
        ax.plot(eigenvalues_res, label='$\mathcal{L}_r$')
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        ax.set_yscale('symlog')
        plt.legend()
        plt.tight_layout()
        plt.show()
















