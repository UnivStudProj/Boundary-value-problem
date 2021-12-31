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
psi_of_x = [u_c + (1 + np.cos((np.pi * x) / l)) for x in x_dots]

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
    

# FIXME: run coeffs bullshit

# SweepMethod for the Crank-Nicolson scheme
def ComputeDots():
    # Calculating run coeffs
    b = 1 + 2 * sigma
    for i in range(I):
        if i == 0:
            # Matrix elements
            a = 0
            c = -2 * sigma
            # Setting coeffs for the next run 
            p_coeff[i + 1] = -c / b
            q_coeff[i + 1] = curr_dots[i] / b
        elif i >= 1 and i <= I - 2:
            # Matrix elements
            a = -sigma
            c = -sigma
            # Run coefficients
            p_coeff[i + 1] = -c / (b + p_coeff[i] * a)
            q_coeff[i + 1] = (curr_dots[i] - a * q_coeff[i]) / (b + p_coeff[i] * a)
        elif i == I - 1:
            # Matrix elements
            a = -2 * sigma
            b += 2 * sigma * H * h_x
            c = 0
            # Run coefficients
            p_coeff[i] = 0
            q_coeff[i] = (curr_dots[i] - a * q_coeff[i]) / (b + a * p_coeff[i])
    # Calculating dots values
    for i in range(I, -1, -1):
        if i == 0 or (i >= 1 and i <= I - 2):
            new_dots[i] = p_coeff[i + 1] * new_dots[i + 1] + q_coeff[i + 1]
        elif i == I - 1:
            new_dots[i] = q_coeff[i]
    

# Creating plots
for t_d in range(1, T, 4):
    ComputeDots()
    plt.plot(x_dots, curr_dots, label=f'u(x, {t_d:.2f})')
    curr_dots = np.array([x for x in new_dots])
    
    
plt.title('Dynamic of substance concentretion change \nwithin the cylinder by time')
plt.xlabel('Coords')
plt.ylabel('Substance concentration')
plt.grid()
plt.legend()

plt.show()
