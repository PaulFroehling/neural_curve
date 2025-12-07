import numpy as np
from random import random
import matplotlib.pyplot as plt

pi_vals = np.linspace(0,2*np.pi, 5000)

r_min = 2
ring_width = 2

x, y, z = [],[],[]
radius =  []

for i in pi_vals:
    r_offset=ring_width*random()
    x_val = (r_min + r_offset) * np.cos(i)
    x.append(x_val)
    y.append((r_min + r_offset) * np.sin(i))
    z.append(r_offset**2 - x_val**2)
    radius.append(r_min+r_offset)

normed_vals = np.array(list(zip(radius, x,y,z)))

min_val = np.min(normed_vals, axis=0)
max_val = np.max(normed_vals, axis=0)

normed_vals = np.array((normed_vals-min_val)/(max_val-min_val)) *15
print(normed_vals)
query =[]
data =[]
for i, rad in enumerate(radius):
    if rad < r_min + ring_width/3:
        query.append([normed_vals[i,1], normed_vals[i,2], normed_vals[i,3]])
    else:
        data.append([normed_vals[i,1], normed_vals[i,2], normed_vals[i,3]])

dataset = np.column_stack((pi_vals, normed_vals))

query, data = np.array(query), np.array(data) 




#np.savetxt("/home/paul-froehling/Dokumente/Code/neural_curve/data/in/saddle/saddle_3d.csv", dataset, delimiter=",")

print(np.array(dataset).shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], color="blue", s=.9)
plt.show()