import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
from NS_model_tf import Sampler, Navier_Stokes2D

if __name__ == '__main__':
    def U_gamma_1(x):
        num = x.shape[0]
        return np.tile(np.array([1.0, 0.0]), (num, 1))


    def U_gamma_2(x):
        num = x.shape[0]
        return np.zeros((num, 2))


    def f(x):
        num = x.shape[0]
        return np.zeros((num, 2))

    def operator(psi, p, x, y, Re, sigma_x=1.0, sigma_y=1.0):
        u = tf.gradients(psi, y)[0] / sigma_y
        v = - tf.gradients(psi, x)[0] / sigma_x

        u_x = tf.gradients(u, x)[0] / sigma_x
        u_y = tf.gradients(u, y)[0] / sigma_y

        v_x = tf.gradients(v, x)[0] / sigma_x
        v_y = tf.gradients(v, y)[0] / sigma_y

        p_x = tf.gradients(p, x)[0] / sigma_x
        p_y = tf.gradients(p, y)[0] / sigma_y

        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        u_yy = tf.gradients(u_y, y)[0] / sigma_y

        v_xx = tf.gradients(v_x, x)[0] / sigma_x
        v_yy = tf.gradients(v_y, y)[0] / sigma_y

        Ru_momentum = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        Rv_momentum = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re

        return Ru_momentum, Rv_momentum

    # Parameters of equations
    Re = 100.0

    # Domain boundaries
    bc1_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    bc2_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc3_coords = np.array([[1.0, 0.0],
                           [1.0, 1.0]])
    bc4_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: U_gamma_1(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: U_gamma_2(x), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: U_gamma_2(x), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: U_gamma_2(x), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x), name='Forcing')

    # Define model
    mode = 'M1'
    layers = [2, 50, 50, 50, 2]

    model = Navier_Stokes2D(layers, operator, bcs_sampler, res_sampler, Re, mode)

    # Train model
    model.train(nIter=40001, batch_size=128)

    # Test Data
    nx = 100
    ny = 100  # change to 100
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # Predictions
    psi_pred, p_pred = model.predict_psi_p(X_star)
    u_pred, v_pred = model.predict_uv(X_star)
    
    psi_star = griddata(X_star, psi_pred.flatten(), (X, Y), method='cubic')
    p_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    u_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    v_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')

    velocity = np.sqrt(u_pred**2 + v_pred**2)
    velocity_star = griddata(X_star, velocity.flatten(), (X, Y), method='cubic')
    
    # Reference
    u_ref= np.genfromtxt("reference_u.csv", delimiter=',')
    v_ref= np.genfromtxt("reference_v.csv", delimiter=',')
    velocity_ref = np.sqrt(u_ref**2 + v_ref**2)
    
    # Relative error
    error = np.linalg.norm(velocity_star - velocity_ref.T, 2) / np.linalg.norm(velocity_ref, 2)
    print('l2 error: {:.2e}'.format(error))
  
    ### Plot ###
    ###########
    
    # Reference solution & Prediceted solution
    fig_1 = plt.figure(1, figsize=(18, 5))
    fig_1.add_subplot(1, 3, 1)
    plt.pcolor(X.T, Y.T, velocity_ref, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reference Velocity')
    
    fig_1.add_subplot(1, 3, 2)
    plt.pcolor(x, Y, velocity_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted Velocity')
    plt.tight_layout()
    
    fig_1.add_subplot(1, 3, 3)
    plt.pcolor(X, Y, np.abs(velocity_star - velocity_ref.T), cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute Error')
    plt.show()
    

     ## Loss ##
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

    ## Adaptive Constant
    adaptive_constant = model.adpative_constant_bcs_log
     
    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant, label='$\lambda_{u_b}$')
    ax.set_xlabel('iterations')
    plt.legend()
    plt.tight_layout()
    plt.show()

    ## Gradients #
    data_gradients_res = model.dict_gradients_res_layers
    data_gradients_bcs = model.dict_gradients_bcs_layers
    
    num_hidden_layers = len(layers) -1
    cnt = 1
    fig_4 = plt.figure(4, figsize=(13, 4))
    for j in range(num_hidden_layers):
        ax = plt.subplot(1, 4, cnt)
        ax.set_title('Layer {}'.format(j + 1))
        ax.set_yscale('symlog')
        gradients_res = data_gradients_res['layer_' + str(j + 1)][-1]
        gradients_bcs = data_gradients_bcs['layer_' + str(j + 1)][-1]
       
        sns.distplot(gradients_res, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \mathcal{L}_r$')
        sns.distplot(gradients_bcs, hist=False,
                     kde_kws={"shade": False},
                     norm_hist=True, label=r'$\nabla_\theta \mathcal{L}_{u_b}$')

        ax.get_legend().remove()
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([0, 100])
        cnt += 1
    handles, labels = ax.get_legend_handles_labels()

    fig_4.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
               borderaxespad=0, bbox_transform=fig_4.transFigure, ncol=2)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    