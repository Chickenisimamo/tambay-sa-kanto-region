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
            print("Stopped because of iterations with an r of " +str(r))
            return x0
    else:
        print("Stopped because of epsilon after " + str(iterations) +" iterations")

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
    # Get training and testing data
    train_data, train_target, test_data, test_target, datacols = dataproc.load_data('data/data.csv') 
    
    # ==========================================================================================================
    # Solve for the predictor using Gauss-Seidel Method
    predictor = gs(train_data.T @ train_data, train_data.T @ train_target.T, 10000, 1e-9)

    # Adds a new dimension for numpy matrix multiplication
    predictor = predictor[np.newaxis]

    # Testing the resulting predictor with the test data
    predictions = test_data @ predictor.T

    # If a prediction is confident enough, we set it to 1 or 0
    # A threshold of 0.6 sets everything greater than 0.6 to 1
    #                                    less than 0.4 to 0 
    threshold = 0.6
    clamped = [1 if x>threshold else (0 if x <= 1-threshold else float(x)) for x in predictions]

    # Compute for the accuracy of the clamped values vs ground truth
    accuracy = np.count_nonzero(clamped == test_target) / len(test_target) * 100
    print("Accuracy:", round(accuracy,3), "%")

    # ==========================================================================================================
    # Solve for the predictor using Gauss-Seidel Method
    predictor = np.linalg.pinv(train_data)  @ train_target

    # Adds a new dimension for numpy matrix multiplication
    predictor = predictor[np.newaxis]

    # Testing the resulting predictor with the test data
    predictions = test_data @ predictor.T

    # If a prediction is confident enough, we set it to 1 or 0
    # A threshold of 0.6 sets everything greater than 0.6 to 1
    #                                    less than 0.4 to 0 
    threshold = 0.6
    clamped = [1 if x>threshold else (0 if x <= 1-threshold else float(x)) for x in predictions]

    # Compute for the accuracy of the clamped values vs ground truth
    accuracy = np.count_nonzero(clamped == test_target) / len(test_target) * 100
    print("X Accuracy:", round(accuracy,3), "%")

    #plot(train_data, train_target, x_gs, datacols)