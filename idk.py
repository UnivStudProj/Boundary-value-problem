import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(filename="ignore/sample.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40

x_start = 0.00
x_step = 0.10
x_dots = np.arange(x_start, l, x_step)

t_start = 0.80
t_step = 4
t_dots = np.arange(t_start, T, t_step)
psi_of_x = [u_c + (1 + np.cos((np.pi * x) / l)) for x in x_dots]

I = len(x_dots) # 20
K = len(t_dots) # 10
h_x = l / I # 0.5
h_t = T / K # 4
cube_a = D / c
x = [i * h_x for i in x_dots]
t = [k * h_t for k in t_dots]
sigma = (h_t * cube_a) / (2 * h_x ** 2)
start_dots = {f'u({x_start + x_step * i}, t)' : psi_of_x[i] for i in range(len(psi_of_x))}
new_dots = {}


# SweepMethod for the Crank-Nicolson scheme
def SweepMethod(i=0):
    b = 1 + 2 * sigma
    if i == 0:
        # Matrix elements
        a = 0
        c = -2 * sigma
        # Run coefficients
        p_next = -c / b
        q_next = getPrev(i) / b
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_0 = p_next * SweepMethod(i + 1) + q_next
        new_dots[f'u({x_start}, t)'] = u_0
    elif i >= 1 and i <= I - 2:
        # Matrix elements
        a = -sigma
        c = -sigma
        # Run coefficients
        p_next = -c / (b + p_curr * a)
        q_next = (getPrev(i) - a * q_curr) / (b + p_curr * a)
        # Setting coeffs for the next run
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_i = p_next * SweepMethod(i + 1) + q_next
        new_dots[f'u({x_start + x_step * i}, t)'] = u_i
        return u_i
    elif i == I - 1:
        # Matrix elements
        a = -2 * sigma
        b += 2 * sigma * H * h_x
        c = 0
        # Run coefficients
        p_next = 0
        q_next = (getPrev(i) - a * q_curr) / (b + a * p_curr)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_I = q_next
        new_dots[f'u({x_start + x_step * i}, t)'] = u_I 
        return u_I


# Returning previous function value
def getPrev(i):
    return start_dots[f'u({x_start + x_step * i}, t)']


# Set the "p" run coeff 
def setCoeff_p(num):
    global p_curr
    p_curr = num
   

# Set the "q" run coeff 
def setCoeff_q(num):
    global q_curr
    q_curr = num


# Creating plots
for t_d in t_dots:
    SweepMethod()
    n = [v for v in list(start_dots.values())]
    plt.plot(x_dots, n, label=f'u(x, {t_d:.2f})')
    start_dots = {k: v for k, v in reversed(list(new_dots.items()))} 
    
plt.title('Dynamic of substance concentretion change within the cylinder by time')
plt.xlabel('Coords')
plt.ylabel('Substance concentration')
plt.grid()
plt.legend()

plt.show()
    
