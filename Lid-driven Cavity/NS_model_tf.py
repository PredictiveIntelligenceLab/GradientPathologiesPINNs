import tensorflow as tf
import numpy as np
import timeit

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

class Navier_Stokes2D:
    def __init__(self, layers, operator, bcs_sampler, res_sampler, Re, model):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_y, self.sigma_y = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Choose model
        self.model = model

        # Navier Stokes constant
        self.Re = tf.constant(Re, dtype=tf.float32)

        # Adaptive re-weighting constant
        self.beta = 0.9
        self.adaptive_constant_bcs_val = np.array(1.0)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        if model in ['M3', 'M4']:
            # Initialize encoder weights and biases
            self.encoder_weights_1 = self.xavier_init([2, layers[1]])
            self.encoder_biases_1 = self.xavier_init([1, layers[1]])

            self.encoder_weights_2 = self.xavier_init([2, layers[1]])
            self.encoder_biases_2 = self.xavier_init([1, layers[1]])

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U_bc1_tf = tf.placeholder(tf.float32, shape=(None, 2))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U_bc2_tf = tf.placeholder(tf.float32, shape=(None, 2))

        self.x_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U_bc3_tf = tf.placeholder(tf.float32, shape=(None, 2))

        self.x_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U_bc4_tf = tf.placeholder(tf.float32, shape=(None, 2))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.adaptive_constant_bcs_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_bcs_val.shape)

        # Evaluate predictions
        self.u_bc1_pred, self.v_bc1_pred = self.net_uv(self.x_bc1_tf, self.y_bc1_tf)
        self.u_bc2_pred, self.v_bc2_pred = self.net_uv(self.x_bc2_tf, self.y_bc2_tf)
        self.u_bc3_pred, self.v_bc3_pred = self.net_uv(self.x_bc3_tf, self.y_bc3_tf)
        self.u_bc4_pred, self.v_bc4_pred = self.net_uv(self.x_bc4_tf, self.y_bc4_tf)

        self.U_bc1_pred = tf.concat([self.u_bc1_pred, self.v_bc1_pred], axis=1)
        self.U_bc2_pred = tf.concat([self.u_bc2_pred, self.v_bc2_pred], axis=1)
        self.U_bc3_pred = tf.concat([self.u_bc3_pred, self.v_bc3_pred], axis=1)
        self.U_bc4_pred = tf.concat([self.u_bc4_pred, self.v_bc4_pred], axis=1)

        self.psi_pred, self.p_pred = self.net_psi_p(self.x_u_tf, self.y_u_tf)
        self.u_pred, self.v_pred = self.net_uv(self.x_u_tf, self.y_u_tf)
        self.u_momentum_pred, self.v_momentum_pred = self.net_r(self.x_r_tf, self.y_r_tf)

        # Residual loss
        self.loss_u_momentum = tf.reduce_mean(tf.square(self.u_momentum_pred))
        self.loss_v_momentum = tf.reduce_mean(tf.square(self.v_momentum_pred))

        self.loss_res = self.loss_u_momentum + self.loss_v_momentum
        
        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.U_bc1_pred - self.U_bc1_tf))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.U_bc2_pred))
        self.loss_bc3 = tf.reduce_mean(tf.square(self.U_bc3_pred))
        self.loss_bc4 = tf.reduce_mean(tf.square(self.U_bc4_pred))
        
        self.loss_bcs = self.adaptive_constant_bcs_tf * tf.reduce_mean(tf.square(self.U_bc1_pred - self.U_bc1_tf) +
                                                                       tf.square(self.U_bc2_pred) +
                                                                       tf.square(self.U_bc3_pred) +
                                                                       tf.square(self.U_bc4_pred))
        
        # Total loss
        self.loss = self.loss_res + self.loss_bcs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Logger
        self.loss_res_log = []
        self.loss_bcs_log = []
        self.saver = tf.train.Saver()

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_bcs_layers = self.generate_grad_dict(self.layers)

        # Gradients Storage
        self.grad_res = []
        self.grad_bcs = []
        for i in range(len(self.layers) - 1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights[i])[0])
            self.grad_bcs.append(tf.gradients(self.loss_bcs, self.weights[i])[0])

        self.adpative_constant_bcs_list = []
        self.adpative_constant_bcs_log = []

        for i in range(len(self.layers) - 1):
            self.adpative_constant_bcs_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs[i])))
        self.adaptive_constant_bcs = tf.reduce_max(tf.stack(self.adpative_constant_bcs_list))

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict
    
    # Save gradients during training
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            grad_res_value, grad_bcs_value = self.sess.run([self.grad_res[i], self.grad_bcs[i]], feed_dict=tf_dict)

            # save gradients of loss_r and loss_u
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_bcs_layers['layer_' + str(i + 1)].append(grad_bcs_value.flatten())
        return None

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        if self.model in ['M1', 'M2']:
            num_layers = len(self.layers)
            for l in range(0, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = self.weights[-1]
            b = self.biases[-1]
            H = tf.add(tf.matmul(H, W), b)
            return H

        if self.model in ['M3', 'M4']:
            num_layers = len(self.layers)
            encoder_1 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_1), self.encoder_biases_1))
            encoder_2 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_2), self.encoder_biases_2))

            for l in range(0, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                H = tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + \
                    tf.math.multiply(1 - tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_2)

            W = self.weights[-1]
            b = self.biases[-1]
            H = tf.add(tf.matmul(H, W), b)
            return H

    # Forward pass for stream-pressure formulation
    def net_psi_p(self, x, y):
        psi_p = self.forward_pass(tf.concat([x, y], 1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        return psi, p

    # Forward pass for velocities
    def net_uv(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u = tf.gradients(psi, y)[0] / self.sigma_y
        v = - tf.gradients(psi, x)[0] / self.sigma_x
        return u, v

    # Forward pass for residual
    def net_r(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u_momentum_pred, v_momentum_pred = self.operator(psi, p, x, y,
                                                         self.Re,
                                                         self.sigma_x, self.sigma_y)

        return u_momentum_pred, v_momentum_pred

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, _ = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            X_bc4_batch, _ = self.fetch_minibatch(self.bcs_sampler[3], batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc1_tf: X_bc1_batch[:, 0:1], self.y_bc1_tf: X_bc1_batch[:, 1:2],
                       self.U_bc1_tf: u_bc1_batch,
                       self.x_bc2_tf: X_bc2_batch[:, 0:1], self.y_bc2_tf: X_bc2_batch[:, 1:2],
                       self.x_bc3_tf: X_bc3_batch[:, 0:1], self.y_bc3_tf: X_bc3_batch[:, 1:2],
                       self.x_bc4_tf: X_bc4_batch[:, 0:1], self.y_bc4_tf: X_bc4_batch[:, 1:2],
                       self.x_r_tf: X_res_batch[:, 0:1], self.y_r_tf: X_res_batch[:, 1:2],
                       self.adaptive_constant_bcs_tf: self.adaptive_constant_bcs_val
                       }

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_value, loss_r_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)

                if self.model in ['M2', 'M4']:
                    # Compute the adaptive constant
                    adaptive_constant_bcs_val = self.sess.run(self.adaptive_constant_bcs, tf_dict)

                    self.adaptive_constant_bcs_val = adaptive_constant_bcs_val * \
                                            (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_val

                self.adpative_constant_bcs_log.append(self.adaptive_constant_bcs_val)
                self.loss_bcs_log.append(loss_u_value)
                self.loss_res_log.append(loss_r_value)

                print('It: %d, Loss: %.3e, Loss_u: %.3e, Loss_r: %.3e, Time: %.2f' %
                      (it, loss_value, loss_u_value, loss_r_value, elapsed))

                print("constant_bcs_val: {:.3f}".format(self.adaptive_constant_bcs_val))
                start_time = timeit.default_timer()

            # Store gradients
            if it % 10000 == 0:
                self.save_gradients(tf_dict)
                print("Gradients information stored ...")

    # Evaluates predictions at test points
    def predict_psi_p(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2]}
        psi_star = self.sess.run(self.psi_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        return psi_star, p_star

    def predict_uv(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        return u_star, v_star







