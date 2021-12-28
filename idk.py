import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-3, 3)
# plt.plot(x, x**2)

# plt.show()

D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40

x_dots = np.arange(0, l, 0.1)
t_dots = np.arange(0.8, T, 4)
psi_of_x = [u_c + (1 + np.cos((np.pi * i) / l)) for i in x_dots]

I = len(x_dots) # 100
K = len(t_dots) # 10
h_x = l / I # 0.1
h_t = T / K # 4
cube_a = D / c
x = [i * h_x for i in x_dots]
t = [k * h_t for k in t_dots]
sigma = (h_t * cube_a) / (2 * (h_x ** 2))


def CrankNicolson_args(i, k):
    global line_b, line_t
    # bottom dots
    line_b = {
        'u_curr' : [None, x[i], t[k]],
        'u_next' : [None, x[i + 1], t[k]],
        'u_prev' : [None, x[i - 1], t[k]]
    } 
    # top dots
    line_t = {
        'u_curr' : [None, x[i], t[k + 1]],
        'u_next' : [None, x[i + 1], t[k + 1]],
        'u_prev' : [None, x[i - 1], t[k + 1]]
    }
    # Sweep method
    if i == 0:
        left_side = -2 * sigma * dot(1, 1) + (1 + 2 * sigma) * dot(1, 0)
        right_side = dot(0, 0) + 2 * sigma * (dot(0, 1) - dot(0, 0))
    elif i >= 1 and i <= I -1:
        left_side = -sigma * dot(1, 1) + (1 + 2 * sigma) * dot(1, 0) - sigma * dot(1, 2)
        right_side = dot(0, 0) + sigma * (dot(0, 1) - 2 * dot(0, 0) + dot(0, 2)) 
    elif i == I:
        left_side = (1 + 2 * sigma + 2 * sigma * H * h_x) * dot(1, 0) - 2 * sigma * dot(1, 2)
        right_side = dot(0, 0) + 2 * sigma * ((-1 - H * h_x) * dot(0, 0) + dot(0, 2))
        
CrankNicolson_args(0, 5)


# For text reduction
def dot(line, pos):
    if line == 0:
        l = line_b
    else:
        l = line_t
    # Searching dot positon in the dicitionary
    for d in range(len(l)):
        if d == pos:
            return list(line_b.values())[d][0]
    

print(dot(0, 1))
    
def CrankNicolson_scheme():
    pass  
    

# Substance concentration

# plt.plot(x_dots, psi_of_x)

# plt.show()
    
