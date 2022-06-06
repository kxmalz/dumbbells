# Post-processing of the evensen_2008.py code

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

radius = 0.55e-6

with open("heights_approx_small.dat") as f:
	size = sum(1 for _ in f)
f.close()

heights = np.empty(size)
angles = np.empty(size)

h = open("heights_approx_small.dat", "r")
a = open("angles_approx_small.dat", "r")

for i, l in enumerate(h):
	heights[i] = l

h.close()

for i, l in enumerate(a):
	angles[i] = l
	
a.close()

with open('Verweij_heights_small.dat') as f:
    size = sum(1 for _ in f)
f.close()

Verweij_heights_x = np.zeros(size)
Verweij_heights_y = np.zeros(size)

V = open("Verweij_heights_small.dat", "r")

for i, line in enumerate(V):
    l = line.split()
    Verweij_heights_x[i], Verweij_heights_y[i] = l[0], l[1]
    
V.close()

with open('Verweij_angles_small.dat') as f:
    size = sum(1 for _ in f)
f.close()

Verweij_angles_x = np.zeros(size)
Verweij_angles_y = np.zeros(size)

V = open("Verweij_angles_small.dat", "r")

for i, line in enumerate(V):
    l = line.split()
    Verweij_angles_x[i], Verweij_angles_y[i] = 0.01745 * float(l[0]), l[1]
    
V.close()

with open('heights_theor_range.dat') as h:
    size = sum(1 for _ in h)
h.close()

h_r = open("heights_theor_range.dat", "r")
h_v = open("heights_theor_values.dat", "r")

heights_range = np.empty(size)
heights_theor = np.empty(size)

for i, l in enumerate(h_r):
	heights_range[i] = l
	
h_r.close()
	
for i, l in enumerate(h_v):
	heights_theor[i] = l
	
h_v.close()

with open('angles_theor_range.dat') as a:
    size = sum(1 for _ in a)
a.close()

a_r = open("angles_theor_range.dat", "r")
a_v = open("angles_theor_values.dat", "r")

angles_range = np.empty(size)
angles_theor = np.empty(size)

for i, l in enumerate(a_r):
	angles_range[i] = l
	
a_r.close()
	
for i, l in enumerate(a_v):
	angles_theor[i] = l
	
a_v.close()

fig, ax1 = plt.subplots()
ax1.hist(heights, bins = 30, range = (0.0, 4.0), density = True)
ax1.set_xlabel("Height $h_{c.m.}$ [$\mu$m]")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
ax2.plot(1e6 * heights_range, heights_theor / max(heights_theor), 'r', label = 'Theoretical predictions')
ax2.plot(Verweij_heights_x, Verweij_heights_y, 'orange', label = 'Verweij et al. [2021]')
ax2.set_ylim(bottom = 0.0)
fig.legend()
plt.title("Heights distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("Verweij_heights_approx_small.png")

fig, ax1 = plt.subplots()
ax1.hist(angles, bins = 30, density = True)
ax1.set_xlabel("Angle $\Theta_p$")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
angles_theor -= angles_theor[0]
ax2.plot(angles_range, angles_theor / max(angles_theor), 'r')
ax2.plot(Verweij_angles_x, Verweij_angles_y / max(Verweij_angles_y), 'orange')
ax2.set_ylim(bottom = 0.0)
plt.title("Angular distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("Verweij_angles_approx_small.png")

fig, ax1 = plt.subplots()
ax1.plot(heights, angles, 'bo', markersize = 1)
geometric_confinement = np.linspace(-np.pi/2, np.pi/2, 100)
ax1.plot((1 + np.abs(np.sin(geometric_confinement))) * 1E6 * radius, geometric_confinement, 'r')
ax1.set_xlim(1e6 * radius, 5)
ax1.set_ylim(-np.pi/2.0, np.pi/2.0)
ax1.set_xlabel("Height $h_{c.m.}$ [$\mu$m]")
ax1.set_ylabel("Angle $\Theta_p$")
plt.title("Heights-angles distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("Verweij_heights_angles_approx_small.png")
