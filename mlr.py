import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import mtxutils as mt
import dataproc 

# A = data, b = target, n max number of iterations, epsilon = convergence criteria
def gs(A, b, n, epsilon):
    L = mt.get_lower_triangle(A)
    D = mt.get_diagonal(A)
    U = mt.get_upper_triangle(A)

    invDL = mt.get_matrix_inverse(D + L)
    ib = invDL @ b # @ is matrix multiplication

    # initialize variables
    iterations = 0
    x0 = np.zeros(A.shape[0]) #zero vector
    r = 1

    while r > epsilon:
        x1 = invDL @ b - (invDL @ U) @ x0 # @ is matrix multiplication
        iterations += 1

        # compute for stopping criterion
        r = mt.get_matrix_norm(b - A @ x1)

        #update new solution vector
        x0 = x1

        # if check_divergence(x1) or iterations >= n:
            # return "divergent"
        
        if iterations >= n:
            return x0

    return x0


def plot(data, target, x, cols):
    for i in range(1,data.shape[1]):
        x_plt = data[:,i]
        y_plt = x_plt * x[0,i]
        plt.plot(x_plt,y_plt,label=cols[i-1])
    plt.legend()
    plt.show()
    pass

if __name__ == "__main__":
    data_raw, data, target, datacols = dataproc.load_data('data/combats_training_set.csv')
    x_gs = gs(data.T @ data, data.T @ target.T, 100, 1e-3)
    x_gs = x_gs[np.newaxis]
    x = np.linalg.pinv(data) @ target
    x = x[np.newaxis]
    print(x_gs,x)
    for i in range(30):
        print(target[i], data[i]@x_gs.T, data[i]@x.T)

    plot(data, target, x_gs, datacols)


# # Plotting 2 x vectors
# # x_plt = data_notnorm[:,3]
# # y_plt = data_notnorm[:,2]
# x_plt = data[:10,2]
# y_plt = data[:10,3]
# z_plt = target[:10]
# # print(y_plt)
# # print(x_plt)

# x_plane, y_plane = np.meshgrid(x_plt, y_plt, copy=False)
# x_plane = x_plane.astype(np.float32)
# y_plane = y_plane.astype(np.float32)
# # print(x_plane, y_plane)
# # print(x_gs)
# z_plane = x_gs[0,2] * x_plane + x_gs[0,3] * y_plane + x_gs[0,0]
# # print(z_plane)

# # Create the plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.5)
# ax.scatter(x_plt,y_plt,z_plt)

# plt.show()
