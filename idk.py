import matplotlib.pyplot as plt
import numpy as np
import logging

# x = np.linspace(-3, 3)
# plt.plot(x, x**2)

# plt.show()

logging.basicConfig(filename="ignore/sample.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40

x_start = 0
x_step = 1.0
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
sigma = (h_t * cube_a) / (2 * (h_x ** 2))
start_dots = {f'u({x_start + x_step * i}, t)' : psi_of_x[i] for i in range(len(psi_of_x))}
new_dots = {}


# Building Crank-Nicolson scheme
def CrankNicolson_args(i):
    # Sweep method
    b = 1 + 2 * sigma
    if i == 0:
        # Matrix elements
        a = 0
        c = -2 * sigma
        # Run coefficients
        p_next = (2 * sigma) / (1 + 2 * sigma)
        q_next = getPrevLine(a, b, c, i) / (1 + 2 * sigma)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_0 = p_next * CrankNicolson_args(i + 1) + q_next
        new_dots[f'u({x_start}, t)'] = u_0
    elif i >= 1 and i <= I - 2:
        # Matrix elements
        a = -sigma
        c = -sigma
        u_i = -c * (b + p_curr * a) 
        # Run coefficients
        p_next = -sigma * (1 + 2 * sigma - sigma * p_curr)
        q_next = (getPrevLine(a, b, c, i) + sigma * q_curr) / (1 + 2 * sigma - sigma * p_curr)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_i *= CrankNicolson_args(i + 1)
        new_dots[f'u({x_start + x_step * i}, t)'] = u_i
        return u_i
    elif i == I - 1:
        # Matrix elements
        a = -2 * sigma
        b += 2 * sigma * H * h_x
        c = 0 
        u_I = (getPrevLine(a, b, c, i) - a * q_curr) / (b + a * p_curr)
        new_dots[f'u({x_start + x_step * i}, t)'] = u_I 
        return u_I


# Set the "p" run coeff 
def setCoeff_p(num):
    global p_curr
    p_curr = num
   

# Set the "q" run coeff 
def setCoeff_q(num):
    global q_curr
    q_curr = num
    
    
# Returning the previous line    
def getPrevLine(a, b, c, i):
    part1 = 0 if a == 0 else a * start_dots[f'u({x_start + x_step * (i - 1)}, t)']
    part2 = b * start_dots[f'u({x_start + x_step * i}, t)'] 
    part3 = 0 if c == 0 else c * start_dots[f'u({x_start + x_step * (i + 1)}, t)']
    return part1 + part2 + part3


CrankNicolson_args(0)

l = [v for v in list(new_dots.values())]
print(l)
# plt.plot(x_dots, l)

# plt.grid()
# plt.show()
    
