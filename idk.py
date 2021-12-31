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
def ComputeDots():
    SetNewDots()
    for i in range(I):
        if i == 0:
            # Matrix elements
            a = 0
            c = -2 * sigma
            # Setting coeffs for the next run 
            p_coeff[i] = -c / b
            q_coeff[i] = new_dots[i] / b
        elif i >= 1 and i <= I - 2:
            # Matrix elements
            a = -sigma
            c = -sigma
            # Run coefficients
            p_coeff[i] = -c / (b + p_coeff[i - 1] * a)
            q_coeff[i] = (new_dots[i] - a * q_coeff[i - 1]) / (b + p_coeff[i - 1] * a)
        elif i == I - 1:
            # Matrix elements
            a = -2 * sigma
            b += 2 * sigma * H * h_x
            c = 0
            # Run coefficients
            p_coeff[I - 1] = 0
            q_coeff[I - 1] = (new_dots[I - 1] - a * q_coeff[I - 1]) / (b + a * p_coeff[I - 1])
    # Calculating dots values
    new_dots[I - 1] = q_coeff[I - 1]
    for i in range(I - 1, 0, -1):
        new_dots[i - 1] = new_dots[i] * p_coeff[i - 1] + q_coeff[i - 1]
    log(new_dots)


# Setting matrix "A"
def setMatrix_A():
    global mat_A
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
def addRow():
    mat_U = np.mat(np.zeros((len(t_dots), mat_A.shape[1]), dtype=float))
    mat_U[[K - 1]] = psi_of_x
    for k in range(K - 2, -1, -1):
        mat_U[[k]] = ComputeDots()
       
     
# Creating plots

plt.plot(x_dots, curr_dots, label=f'u(x, {t_d:.2f})')
curr_dots = np.array([x for x in new_dots])
    
plt.title('Dynamic of substance concentretion change \nwithin the cylinder by time')
plt.xlabel('Coords')
plt.ylabel('Substance concentration')
plt.grid()
plt.legend()

plt.show()
