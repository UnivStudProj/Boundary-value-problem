import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd

logging.basicConfig(filename="ignore/sample.log", format='%(message)s', level=logging.INFO)

# Input values
D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40
x_amount = 50
t_amount = 50

# Arguments for the Crank-Nicolson scheme
h_x = l / x_amount 
h_t = T / t_amount
sq_a = D / c
sigma = (h_t * sq_a) / (2 * h_x ** 2)
x_dots = np.linspace(0, l, num=x_amount)
t_dots = np.linspace(0, T, num=t_amount)
psi_of_x = [u_c + (1 + np.cos(np.pi * x / l)) for x in x_dots]

# Using "I", "K" instead of "x_amount - 1", "t_amount - 1" respectively for more readability
I = x_amount - 1
K = t_amount - 1

# Arrays with the dots values
u_prev = np.array([*psi_of_x], dtype=float)
u_new = np.empty_like(u_prev) 

# Sweep coeffs arrays
p = np.empty(len(x_dots), dtype=float)
q = np.empty_like(p)


# Logging thing
def log(smth):
    logging.info('=' * 50)
    logging.info(smth)
    logging.info('=' * 50)
    

# Preparing new dots for the Tomas algorithm
def setNewDots():
    u_new[0] = u_prev[0] + 2 * sigma * (u_prev[1] - u_prev[0])
    for i in range(1, I):
        u_new[i] = u_prev[i] + sigma * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1])
    u_new[I] = u_prev[I] + 2 * sigma * ((-1 - H * h_x) * u_prev[I] + u_prev[I])


# Sweep method for the Crank-Nicolson scheme
def TomasAlgorithm(mat_A):
    setNewDots()
    # The forward sweep consists of the computation of new coefficients
    p[0] = -mat_A[0, 1] / mat_A[0, 0]
    q[0] = u_new[0] / mat_A[0, 0]
    for i in range(1, I):
        p[i] = -mat_A[i, i + 1] / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
        q[i] = (u_new[i] - mat_A[i, i - 1] * q[i - 1]) / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
    p[I] = 0
    q[I] = (u_new[I] - mat_A[I, I - 1] * q[I - 1]) / (mat_A[I, I] + mat_A[I, I - 1] * p[I - 1])
    # The solution is then obtained by back substitution
    u_new[I] = q[I]
    for i in range(I, 0, -1):
        u_new[i - 1] = u_new[i] * p[i - 1] + q[i - 1]
    return u_new


# Setting matrix "A" values
def setMatrix_A():
    mat_A = np.mat(np.empty((len(x_dots), len(x_dots)), dtype=float))
    mat_A[0, 0] = 1 + 2 * sigma
    mat_A[0, 1] = -2 * sigma
    for i in range(1, I):
        mat_A[i, i - 1] = -sigma
        mat_A[i, i] = 1 + 2 * sigma
        mat_A[i, i + 1] = - sigma
    mat_A[I, I - 1] = -2 * sigma
    mat_A[I, I] = 1 + 2 * sigma + 2 * sigma * H * h_x
    return mat_A
  

# Adding rows with substance concentration values
def Solution(mat_A):
    global u_prev
    mat_U = np.mat(np.empty((len(t_dots), mat_A.shape[1]), dtype=float))
    mat_U[K] = psi_of_x
    for k in range(K - 1, -1, -1):
        mat_U[k] = TomasAlgorithm(mat_A)
        u_prev = np.array([*u_new]) 
    return mat_U


# Stability analysis
def isStable():
    return sigma <= 1 / (2 + 2 * h_x * H)
    
  
# Truncation error analysis 
def ErrorAnalysis():
    """ Computing the error analysis

        - Node: u(0, K)
        - Steps of discretization: 4
        - Adding 2 extra elements to some arrays,
            since it need to write the next two values of the node
        
    """
    U_arr = np.empty(6, dtype=float) # "U[0, 0]" values
    I_arr = np.full(4, 50, dtype=int) # "I" values (immutable)
    K_arr = np.empty_like(I_arr) # "K" values (each value multiplied by 2)
    DeltaDiv2 = np.empty_like(U_arr) # Dicretizating step in space/time by 2
    DeltaDiv4 = np.empty_like(U_arr) # Dicretizating step in space/time by 4
    SmallDelta = np.empty_like(U_arr) # The error
    U_arr[0] = Solution(setMatrix_A())[0, 0]
    for i in range(4):
        K_arr[i] = t_amount
        changeTimeInterval(1)
        U_arr[i + 1] = Solution(setMatrix_A())[0, 0]
        DeltaDiv2[i] = U_arr[i] - U_arr[i + 1]
        changeTimeInterval(1)
        U_arr[i + 2] = Solution(setMatrix_A())[0, 0]
        DeltaDiv4[i] = U_arr[i + 1] - U_arr[i + 2]
        SmallDelta[i] = DeltaDiv2[i] / DeltaDiv4[i]
        changeTimeInterval()
    err = np.array([I_arr, K_arr[:4], DeltaDiv2[:4], DeltaDiv4[:4], SmallDelta[:4]])
    ErrorPlot(err)


# Showing the error analysis table
def ErrorPlot(err):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=160)
    ax.axis("tight")
    ax.axis("off")
    df = pd.DataFrame(err.T, columns=["I", "K", "Div2", "Div4", "delta"])
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()
    plt.show()
   
   
# Increasing/Decreasing time interval 
def changeTimeInterval(mode=0):
    global t_amount, h_t, sigma, t_dots
    t_amount = t_amount * 2 if mode == 1 else t_amount // 2
    h_t = T / t_amount
    sigma = (h_t * sq_a) / (2 * h_x ** 2)
    t_dots = np.linspace(0, T, num=t_amount)
    
    
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
# U = Solution(setMatrix_A())
# createPlots(U)
ErrorAnalysis()
