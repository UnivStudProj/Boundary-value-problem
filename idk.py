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

x_dots = np.arange(0, l, 0.5)
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
plots = {'u(x, 0)' : psi_of_x}


# Building Crank-Nicolson scheme
def CrankNicolson_args(i, k=0):
    # Sweep method
    b = 1 + 2 * sigma
    if i == 0:
        # Matrix elements
        a = 0
        c = -2 * sigma
        # Run coefficients
        p_next = (2 * sigma) / (1 + 2 * sigma)
        q_next = [dot / (1 + 2 * sigma) for dot in getPrevDot(a, b, c, i)]
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_0 = [-(c / b) * CrankNicolson_args(i + 1) + dot / b for dot in getPrevDot(a, b, c, i)]
        plots[f'u(x, {t_start + t_step})'] = u_0
    elif i >= 1 and i <= I - 2:
        # Matrix elements
        a = -sigma
        c = -sigma
        # Run coefficients
        p_next = -sigma * (1 + 2 * sigma - sigma * p_curr)
        q_next = [(dot + sigma * q_curr) / (1 + 2 * sigma - sigma * p_curr) for dot in getPrevDot(a, b, c, i)]
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_i = -c * (b + p_next * a) * CrankNicolson_args(i + 1)
        plots[f'u(x, {t_start + t_step * i})'] = u_i
        return u_i
    elif i == I - 1:
        # Matrix elements
        a = -2 * sigma
        b += 2 * sigma * H * h_x
        c = 0 
        # Run coefficients
        p_next = 0
        q_next = [(dot - a * q_curr) / (1 + 2 * sigma - 2 * sigma * H * h_x - 2 * sigma * p_curr) for dot in getPrevDot(a, b, c, i)]
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_I = [(dot - a * q_el) / (b + a * p_next) for dot, q_el in zip(getPrevDot(a, b, c, i), q_next)]
        plots[f'u(x, {t_start + t_step * i})'] = u_I 
        return u_I

   
# Set the "p" run coeff 
def setCoeff_p(num):
    global p_curr
    p_curr = num
   

# Set the "q" run coeff 
def setCoeff_q(num):
    global q_curr
    q_curr = num
    
    
# Returning the previous (already known) dot    
def getPrevDot(a, b, c, i):
    part1 = 0 if a == 0 else [a * dot for dot in plots[f'u(x, {t_start + t_step * (i - 1)})']] 
    part2 = [b * dot for dot in plots[f'u(x, {t_start + t_step * i})']] 
    part3 = 0 if c == 0 else [c * dot for dot in plots[f'u(x, {t_start + t_step * (i + 1)})']]
    return [el1 + el2 + el3 for el1, el2, el3 in zip(part1, part2, part3)]


CrankNicolson_args(0)

plt.plot(x_dots, l)

plt.grid()
plt.show()
    
