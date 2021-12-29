import matplotlib.pyplot as plt
import numpy as np
import logging

# x = np.linspace(-3, 3)
# plt.plot(x, x**2)

# plt.show()

logging.basicConfig(filename="sample.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40

x_dots = np.arange(0, l, 0.5)
t_dots = np.arange(0.8, T, 4)
psi_of_x = [u_c + (1 + np.cos((np.pi * x) / l)) for x in x_dots]

I = len(x_dots) # 100
K = len(t_dots) # 10
h_x = l / I # 0.1
h_t = T / K # 4
cube_a = D / c
x = [i * h_x for i in x_dots]
t = [k * h_t for k in t_dots]
sigma = (h_t * cube_a) / (2 * (h_x ** 2))
dots_bottom = {}
dots_top = dict([('u' + str(i), psi_of_x[i]) for i in range(len(psi_of_x))])


# Building Crank-Nicolson scheme
def CrankNicolson_args(i, k=0):
    global dots_bottom 
    # Sweep method
    if i == 0:
        # Matrix elements
        a_i = 0
        b_i = 1 + 2 * sigma
        c_i = -2 * sigma
        # Run coefficients
        p_next = (2 * sigma) / (1 + 2 * sigma)
        q_next = getPrevDot(a_i, b_i, c_i, i) / (1 + 2 * sigma)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_0 = -(c_i / b_i) * CrankNicolson_args(i + 1) + getPrevDot(a_i, b_i, c_i, i) / b_i
        dots_bottom['u' + str(i)] = [u_0, (x[i], t[k])]
    elif i >= 1 and i <= I - 2:
        # Matrix elements
        a_i = -sigma
        b_i = 1 + 2 * sigma
        c_i = -sigma
        # Run coefficients
        p_next = -sigma * (1 + 2 * sigma - sigma * p_curr)
        q_next = (getPrevDot(a_i, b_i, c_i, i) + sigma * q_curr) / (1 + 2 * sigma - sigma * p_curr)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_i = -c_i * (b_i + p_next * a_i) * CrankNicolson_args(i + 1)
        dots_bottom['u' + str(i)] = [u_i, (x[i], t[k])]
        return u_i
    elif i == I - 1:
        # Matrix elements
        a_i = -2 * sigma
        b_i = 1 + 2 * sigma + 2 * sigma * H * h_x
        c_i = 0 
        # Run coefficients
        p_next = 0
        q_next = (getPrevDot(a_i, b_i, c_i, i) - a_i * q_curr) / (1 + 2 * sigma - 2 * sigma * H * h_x - 2 * sigma * p_curr)
        # Setting coeffs for the next run 
        setCoeff_p(p_next)
        setCoeff_q(q_next)
        u_I = (getPrevDot(a_i, b_i, c_i, i) - a_i * q_next) / (b_i + a_i * p_next)
        dots_bottom['u' + str(i)] = [u_I, (x[i], t[k])]
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
    part1 = 0 if a == 0 else a * dots_top['u' + str(i - 1)] 
    part2 = b * dots_top['u' + str(i)] 
    part3 = 0 if c == 0 else c * dots_top['u' + str(i + 1)]
    return part1 + part2 + part3


CrankNicolson_args(0, 2)
l = [list(dots_bottom.values())[i][0] for i in range(len(dots_bottom))]
print(dots_bottom)

plt.plot(x_dots, l)

plt.grid()
plt.show()
    
