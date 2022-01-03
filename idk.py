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
h_x = l / I
h_t = T / K
cube_a = D / c
sigma = (h_t * cube_a) / (2 * h_x ** 2)
x_dots = np.linspace(0, l, num=I)
t_dots = np.linspace(0, T, num=K)
psi_of_x = [u_c + (1 + np.cos(np.pi * x / l)) for x in x_dots]

# Arrays with the function values
u_prev = np.array([*psi_of_x], dtype=float)
u_new = np.empty_like(u_prev) 

# Run coeffs vaults
p = np.zeros(len(x_dots), dtype=float)
q = np.zeros(len(x_dots), dtype=float)


# Logging thing
def log(smth):
    logging.info('=' * 50)
    logging.info(smth)
    logging.info('=' * 50)
    

# Setting new dots from previouses
def SetNewDots():
    u_new[0] = u_prev[0] + 2 * sigma * (u_prev[1] - u_prev[0])
    for i in range(1, I - 1):
        u_new[i] = u_prev[i] + sigma * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1])
    u_new[I - 1] = u_prev[I - 1] + 2 * sigma * ((-1 - H * h_x) * u_prev[I - 1] + u_prev[I - 2])


# SweepMethod for the Crank-Nicolson scheme
def ComputeDots(mat_A):
    SetNewDots()
    # Computing run coeffs 
    p[0] = -mat_A[0, 1] / mat_A[0, 0]
    q[0] = u_new[0] / mat_A[0, 0]
    for i in range(1, I - 1):
        p[i] = -mat_A[i, i + 1] / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
        q[i] = (u_new[i] - mat_A[i, i - 1] * q[i - 1]) / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
    p[I - 1] = 0
    q[I - 1] = (u_new[I - 1] - mat_A[I - 1, I - 2] * q[I - 2]) / (mat_A[I - 1, I - 1] + mat_A[I - 1, I - 2] * p[I - 2])
    # Calculating dots values
    u_new[I - 1] = q[I - 1]
    for i in range(I - 1, 0, -1):
        u_new[i - 1] = u_new[i] * p[i - 1] + q[i - 1]
    return u_new


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
    global u_prev
    mat_U = np.mat(np.zeros((len(t_dots), mat_A.shape[1]), dtype=float))
    mat_U[[K - 1]] = psi_of_x
    for k in range(K - 2, -1, -1):
        mat_U[k] = ComputeDots(mat_A)
        u_prev = np.array([*u_new]) 
    return mat_U
      
    
# Creating two plots 
def createPlots(U):
    # Creating one figure (window) which contains two axes (plots)
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, # Number of rows of the subplot grid. 
        ncols=2, # Number of columns of the subplot grid. 
        figsize=(12, 5), # Figure size in inches (size also affected by dpi)
        num='Dynamic of substance concentretion change within the cylinder' # Window title
    )
    # Substance concentration by time plot
    for i in range(U.shape[0] - 1, -1, int(-U.shape[0] / 10)):
        ax1.plot(x_dots, np.ravel(U[i]), label=f'u(x, {T - i * h_t:.2f})')
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.set_title('In time')
    ax1.set_xlabel('Coords')
    ax1.set_ylabel('Substance concentration')
    ax1.grid()
   # Substance concentration by space plot
    for k in range(0, U.shape[1], int(U.shape[1] / 10)):
        ax2.plot(t_dots, np.flip(np.ravel(U[:, k])), label=f'u({k * h_x:.2f}, t)')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.set_title('In space')
    ax2.set_xlabel('Time')
    ax2.grid()
    fig.tight_layout(w_pad=2) # Plots padding (width)
    plt.show()
 

# Program start 
U = addRow(setMatrix_A())
createPlots(U)
