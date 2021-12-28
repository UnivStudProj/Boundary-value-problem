x = [0, 1 ,2]
t = [3, 4, 5]
i = 1
k = i
line_b = {
    'u_curr' : [None, x[i], t[k]],
    'u_next' : [None, x[i + 1], t[k]],
    'u_prev' : [None, x[i - 1], t[k]]
} 
for k, v, k1  in line_b.items(), line_b:
     print(k1)