import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(filename="ignore/sample.log", format='%(message)s', level=logging.INFO)

# Input values
D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40
I = 50
K = 50

# Arguments for the Crank-Nicolson scheme
h_x = l / I # 0.5
h_t = T / K # 4
cube_a = D / c
sigma = (h_t * cube_a) / (2 * h_x ** 2)
x_dots = np.linspace(0, l, num=I)
t_dots = np.linspace(0, T, num=K)
psi_of_x = [u_c + (1 + np.cos(np.pi * x / l)) for x in x_dots]

# Arrays with the function values
curr_dots = np.array([i for i in psi_of_x], dtype=float)
new_dots = np.empty_like(curr_dots) 

# Run coeffs vaults
p_coeff = np.zeros(len(x_dots), dtype=float)
q_coeff = np.zeros(len(x_dots), dtype=float)


# Logging thing
def log(smth):
    logging.info('=' * 50)
    logging.info(smth)
    logging.info('=' * 50)
    

# Setting new dots from previouses
def SetNewDots():
    new_dots[0] = curr_dots[0] + 2 * sigma * (curr_dots[1] - curr_dots[0])
    for i in range(1, I - 1):
        new_dots[i] = curr_dots[i] + sigma * (curr_dots[i + 1] - 2 * curr_dots[i] + curr_dots[i - 1])
    new_dots[I - 1] = curr_dots[I - 1] + 2 * sigma * ((-1 - H * h_x) * curr_dots[I - 1] + curr_dots[I - 2])


# SweepMethod for the Crank-Nicolson scheme
def ComputeDots(mat_A):
    SetNewDots()
    # Computing run coeffs 
    p_coeff[0] = -mat_A[0, 1] / mat_A[0, 0]
    q_coeff[0] = new_dots[0] / mat_A[0, 0]
    for i in range(1, I - 1):
        p_coeff[i] = -mat_A[i, i + 1] / (mat_A[i, i] + p_coeff[i - 1] * mat_A[i, i - 1])
        q_coeff[i] = (new_dots[i] - mat_A[i, i - 1] * q_coeff[i - 1]) / (mat_A[i, i] + p_coeff[i - 1] * mat_A[i, i - 1])
    p_coeff[I - 1] = 0
    q_coeff[I - 1] = (new_dots[I - 1] - mat_A[I - 1, I - 2] * q_coeff[I - 2]) / (mat_A[I - 1, I - 1] + mat_A[I - 1, I - 2] * p_coeff[I - 2])
    # Calculating dots values
    new_dots[I - 1] = q_coeff[I - 1]
    for i in range(I - 1, 0, -1):
        new_dots[i - 1] = new_dots[i] * p_coeff[i - 1] + q_coeff[i - 1]
    return new_dots


# Setting matrix "A" values
def setMatrix_A():
    mat_A = np.mat(np.zeros((len(x_dots), len(x_dots)), dtype=float))
    mat_A[0, 0] = 1 + 2 * sigma
    mat_A[0, 1] = -2 * sigma
    for i in range(1, I - 1):
        mat_A[i, i - 1] = -sigma
        mat_A[i, i] = 1 + 2 * sigma
        mat_A[i, i + 1] = - sigma
    mat_A[I - 1, I - 2] = -2 * sigma
    mat_A[I - 1, I - 1] = 1 + 2 * sigma + 2 * sigma * H * h_x
    return mat_A
  

# Adding rows with substance concentration values
def addRow(mat_A):
    global curr_dots
    mat_U = np.mat(np.zeros((len(t_dots), mat_A.shape[1]), dtype=float))
    mat_U[[K - 1]] = psi_of_x
    for k in range(K - 2, -1, -1):
        mat_U[[k]] = ComputeDots(mat_A)
        curr_dots = np.array([x for x in new_dots]) 
    return mat_U
      
    
# Creating 2 plots 
def createPlots(U):
    # Substance concentration by time plot
    plt.figure()
    plt.title('Dynamic of substance concentretion change \nwithin the cylinder by time')
    plt.xlabel('Coords')
    plt.ylabel('Substance concentration')
    plt.grid()
    # Building curves
    for i in range(U.shape[0] - 1, -1, int(-U.shape[0] / 10)):
        fig = plt.subplot()
        fig.plot(x_dots, np.ravel(U[[i]]), label=f'u(x, {T - i * h_t:.2f})')
        fig.legend()
    plt.show()
    
   # Substance concentration by space plot
    plt.figure()
    plt.title('Dynamic of substance concentretion change \nwithin the cylinder by space')
    plt.xlabel('Time')
    plt.ylabel('Substance concentration')
    plt.grid()
    # Building curves 
    for k in range(U.shape[1] - 1, -1, int(-U.shape[1] / 10)):
        fig = plt.subplot()
        fig.plot(t_dots, np.flip(np.ravel(U[:, k])), label=f'u({l - k * h_x:.2f}, t)')
        fig.legend()
    plt.show()
 

# Program start 
U = addRow(setMatrix_A())
createPlots(U)


    
