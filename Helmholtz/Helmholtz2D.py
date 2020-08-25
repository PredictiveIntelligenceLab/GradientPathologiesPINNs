import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Helmholtz2D_model_tf import Sampler, Helmholtz2D

if __name__ == '__main__':
    
    a_1 = 1
    a_2 = 4
    
    def u(x, a_1, a_2):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_xx(x, a_1, a_2):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_yy(x, a_1, a_2):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    # Forcing
    def f(x, a_1, a_2, lam):
        return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)

    def operator(u, x1, x2, lam, sigma_x1=1.0, sigma_x2=1.0):
        u_x1 = tf.gradients(u, x1)[0] / sigma_x1
        u_x2 = tf.gradients(u, x2)[0] / sigma_x2
        u_xx1 = tf.gradients(u_x1, x1)[0] / sigma_x1
        u_xx2 = tf.gradients(u_x2, x2)[0] / sigma_x2
        residual = u_xx1 + u_xx2 + lam * u
        return residual

    # Parameter
    lam = 1.0

    # Domain boundaries
    bc1_coords = np.array([[-1.0, -1.0],
                           [1.0, -1.0]])
    bc2_coords = np.array([[1.0, -1.0],
                           [1.0, 1.0]])
    bc3_coords = np.array([[1.0, 1.0],
                           [-1.0, 1.0]])
    bc4_coords = np.array([[-1.0, 1.0],
                           [-1.0, -1.0]])

    dom_coords = np.array([[-1.0, -1.0],
                           [1.0, 1.0]])

    # Create initial conditions samplers
    ics_sampler = None

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a_1, a_2, lam), name='Forcing')

    # Define model
    mode = 'M1'            # Method: 'M1', 'M2', 'M3', 'M4'
    stiff_ratio = False    # Log the eigenvalues of Hessian of losses

    layers = [2, 50, 50, 50, 1]
    model = Helmholtz2D(layers, operator, ics_sampler, bcs_sampler, res_sampler, lam, mode, stiff_ratio)

    # Train model
    model.train(nIter=40001, batch_size=128)

    # Test data
    nn = 100
    x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    x1, x2 = np.meshgrid(x1, x2)
    X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

    # Exact solution
    u_star = u(X_star, a_1, a_2)
    f_star = f(X_star, a_1, a_2, lam)

    # Predictions
    u_pred = model.predict_u(X_star)
    f_pred = model.predict_r(X_star)

    # Relative error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_f = np.linalg.norm(f_star - f_pred, 2) / np.linalg.norm(f_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_u: {:.2e}'.format(error_f))

    ### Plot ###

    # Exact solution & Predicted solution
    # Exact soluton
    U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')
    F_star = griddata(X_star, f_star.flatten(), (x1, x2), method='cubic')

    # Predicted solution
    U_pred = griddata(X_star, u_pred.flatten(), (x1, x2), method='cubic')
    F_pred = griddata(X_star, f_pred.flatten(), (x1, x2), method='cubic')

    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Exact $u(x)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Predicted $u(x)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()

    # Residual loss & Boundary loss
    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log

    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_res, label='$\mathcal{L}_{r}$')
    ax.plot(loss_bcs, label='$\mathcal{L}_{u_b}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Adaptive Constant
    adaptive_constant = model.adpative_constant_log

    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant, label='$\lambda_{u_b}$')
    ax.set_xlabel('iterations')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gradients at the end of training
    data_gradients_res = model.dict_gradients_res_layers
    data_gradients_bcs = model.dict_gradients_bcs_layers

    gradients_res_list = []
    gradients_bcs_list = []

    num_hidden_layers = len(layers) - 1
    for j in range(num_hidden_layers):
        gradient_res = data_gradients_res['layer_' + str(j + 1)][-1]
        gradient_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]

        gradients_res_list.append(gradient_res)
        gradients_bcs_list.append(gradient_bcs)

    cnt = 1
    fig_4 = plt.figure(4, figsize=(13, 4))
    for j in range(num_hidden_layers):
        ax = plt.subplot(1, 4, cnt)
        ax.set_title('Layer {}'.format(j + 1))
        ax.set_yscale('symlog')
        gradients_res = data_gradients_res['layer_' + str(j + 1)][-1]
        gradients_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]
        sns.distplot(gradients_bcs, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \lambda_{u_b} \mathcal{L}_{u_b}$')
        sns.distplot(gradients_res, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \mathcal{L}_r$')
      
        ax.get_legend().remove()
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([0,100])
        cnt += 1
    handles, labels = ax.get_legend_handles_labels()
    fig_4.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
               borderaxespad=0, bbox_transform=fig_4.transFigure, ncol=2)
    plt.tight_layout()
    plt.show()

    # Eigenvalues if applicable
    if stiff_ratio:
        eigenvalues_list = model.eigenvalue_log
        eigenvalues_bcs_list = model.eigenvalue_bcs_log
        eigenvalues_res_list = model.eigenvalue_res_log
        eigenvalues_res = eigenvalues_res_list[-1]
        eigenvalues_bcs = eigenvalues_bcs_list[-1]

        fig_5 = plt.figure(5)
        ax = fig_5.add_subplot(1, 1, 1)
        ax.plot(eigenvalues_res, label='$\mathcal{L}_r$')
        ax.plot(eigenvalues_bcs, label='$\mathcal{L}_{u_b}$')
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalue')
        ax.set_yscale('symlog')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
    
        
        
    
     
     
    
    
    
    
    

