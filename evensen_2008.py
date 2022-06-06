# Following code is implementation of simulation published in:
# "Brownian Dynamics Simulations of Rotational Diffusion Using
# the Cartesian Components of the Rotation Vector as Generalized Coordinates"
# T. R. Evensen, S. N. Naess & A. Elgsaeter
# Macromol. Theory Simul. (2008)
# doi:10.1002/mats.200800031

import pychastic
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

# Font parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'cm'

radius = 0.55e-6  # particle radius		[m]

rho_p = 2.0e3  # particle density		[kg/m^3]
rho_f = 997  # fluid density			[kg/m^3]

T = 300  # fluid temperature			[K]
kB = 1.38e-23  # Boltzmann constant		[J/K]
kBT = kB * T

g = 9.81  # gravitational acceleration	[m/s^2]
eps_0 = 8.854e-12  # vacuum permitivity	[F/m]
epsilon = 80 * eps_0  # water permitivity 	[F/m]
kappa = 1 / 103e-9  # 1 / Debye length	[1/m]
e = 1.602e-19  # elementary charge		[C]
eta = 1e-3  # dyn. viscosity of the fluid	[Pa*s]

zeta_w = -54e-3  # wall-corresp. zeta potential [V]
zeta_p = -30e-3  # par.-corresp. zeta potential [V]

# wall's Stern potential	[V]
#psi_w = zeta_w * jnp.exp(1 - kappa * radius) / (radius * kappa)
psi_w = zeta_w / ((kappa * radius) / ((1.0 + kappa * radius) * jnp.e))
# particle's Stern potential	[V]
#psi_p = zeta_p * jnp.exp(1 - kappa * radius) / (radius * kappa)
psi_p = zeta_p / ((kappa * radius) / ((1.0 + kappa * radius) * jnp.e))

print(psi_w)
print(psi_p)

def mobility(xq):
    H = xq[2] * 1e6
    unsafe_phi_squared = jnp.sum(xq[3:] ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.001) ** 2)
    phi = jnp.sqrt(phi_squared)
    cos = 1.0 - (1.0 - jnp.cos(phi)) * (1.0 - (xq[5] / phi) ** 2)
    sin = jnp.sqrt(1.0 - cos ** 2)

    Xtt = 3.8708425598363618
    Ytt = 4.3479650066773061
    Xrr = 1.8030828271581760
    Yrr = 3.7400782411154063
    Xrd = -1.5710943251115512
    Xdr = 1.5710943251115512

    resistance_matrix = jnp.array([[Xtt-(Xtt**2*((-3*cos**2)/2-(3*(1-cos**2))/4))/(8*H)+(Xtt*(4*Xtt**2*((-3*cos**2)/2-(3*(1-cos**2))/4)**2+(9*Xtt*Ytt*cos**2*sin**2)/4))/(256*H**2),
	0,(-3*Xtt*Ytt*cos*sin)/(32*H)+(Ytt*(3*Xtt**2*cos*((-3*cos**2)/2-(3*(1-cos**2))/4)*sin+3*Xtt*Ytt*cos*sin*((-3*sin**2)/2-(3*(1-sin**2))/4)))/(256*H**2),
	0,(Xdr*Xtt*((-9*cos**2*sin)/2+3*(-sin/2+cos**2*sin)))/(32*H**2),0],
	[0,Ytt+(3*Ytt**2)/(32*H)+(9*Ytt**3)/(1024*H**2),0,0,0,(-3*Xdr*Ytt*cos)/(64*H**2)],
	[(-3*Xtt*Ytt*cos*sin)/(32*H)+(Xtt*(3*Xtt*Ytt*cos*((-3*cos**2)/2-(3*(1-cos**2))/4)*sin+3*Ytt**2*cos*sin*((-3*sin**2)/2-(3*(1-sin**2))/4)))/(256*H**2),
	0,Ytt-(Ytt**2*((-3*sin**2)/2-(3*(1-sin**2))/4))/(8*H)+(Ytt*((9*Xtt*Ytt*cos**2*sin**2)/4+4*Ytt**2*((-3*sin**2)/2-(3*(1-sin**2))/4)**2))/(256*H**2),
	0,(Xdr*Ytt*((9*cos*sin**2)/2+3*(cos/2-cos*sin**2)))/(32*H**2),0],
	[0,0,0,Xrr-(Xrr**2*(-cos**2-(5*(1-cos**2))/2))/(64*H**3),0,-((3*Xdr*Xrr*cos*sin)/2-(3*Xrr*Yrr*cos*sin)/2)/(64*H**3)],
	[-(Xrd*Xtt*((-9*cos**2*sin)/2+3*(-sin/2+cos**2*sin)))/(32*H**2),0,-(Xrd*Ytt*((9*cos*sin**2)/2+3*(cos/2-cos*sin**2)))/(32*H**2),
	0,Yrr-((-5*Yrr**2)/2+3*Xdr*Yrr*(cos**2/2-sin**2/2)+3*Xrd*Yrr*(-cos**2/2+sin**2/2)-Xdr*Xrd*(-18*cos**2*sin**2-3*(1/2+(cos**2*sin**2)/2+(-cos**2-sin**2)/2)-18*(-(cos**2*sin**2)+(cos**2+sin**2)/4)))/(64*H**3),0],
	[0,(3*Xrd*Ytt*cos)/(64*H**2),0,-((-3*Xrd*Xrr*cos*sin)/2-(3*Xrr*Yrr*cos*sin)/2)/(64*H**3),0,Yrr-((3*Xdr*Yrr*cos**2)/2-(3*Xrd*Yrr*cos**2)/2-Xdr*Xrd*((-9*cos**2)/2-3*(1/2-cos**2/2))+Yrr**2*(-sin**2-(5*(1-sin**2))/2))/(64*H**3)]])

    mobility_unnormalized = jnp.linalg.inv(resistance_matrix)

    tt_norm = 1 / (3 * jnp.pi * eta * 2 * radius)
    rr_norm = 1 / (jnp.pi * eta * (2 * radius) ** 3)
    tr_norm = 1 / (jnp.sqrt(3) * jnp.pi * eta * (2 * radius) ** 2)

    norms_matrix = jnp.concatenate(
        (
            jnp.concatenate(
                (jnp.full((3, 3), tt_norm), jnp.full((3, 3), tr_norm)), axis=1
            ),
            jnp.concatenate(
                (jnp.full((3, 3), tr_norm), jnp.full((3, 3), rr_norm)), axis=1
            ),
        ),
        axis=0,
    )

    return mobility_unnormalized * norms_matrix


def spin_matrix(q):
    # Antisymmetric matrix dual to q
    return jnp.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])


def rotation_matrix(xq):
    # Compare with equation: Evensen2008.11
    q = xq[3:]
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    rot = jnp.where(
        phi_squared == unsafe_phi_squared,
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        (1.0 - 0.5 * unsafe_phi_squared) * jnp.eye(3)
        + spin_matrix(q)
        + 0.5 * q.reshape(1, 3) * q.reshape(3, 1),
    )

    return jnp.concatenate(
        (
            jnp.concatenate((rot, jnp.zeros((3, 3))), axis=1),
            jnp.concatenate((jnp.zeros((3, 3)), rot), axis=1),
        ),
        axis=0,
    )


def rotation_matrix_r(q):
    # Compare with equation: Evensen2008.11
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    rot = jnp.where(
        phi_squared == unsafe_phi_squared,
        (jnp.sin(phi) / phi) * spin_matrix(q)
        + jnp.cos(phi) * jnp.eye(3)
        + ((1.0 - jnp.cos(phi)) / phi ** 2) * q.reshape(1, 3) * q.reshape(3, 1),
        (1.0 - 0.5 * unsafe_phi_squared) * jnp.eye(3)
        + spin_matrix(q)
        + 0.5 * q.reshape(1, 3) * q.reshape(3, 1),
    )

    return rot


def transformation_matrix(xq):
    # Compare with equation: Evensen2008.12 - there are typos!
    # Compare with equation: Ilie2014.A9-A10 - no typos (except from Taylor-expanded terms)
    q = xq[3:]
    unsafe_phi_squared = jnp.sum(q ** 2)
    phi_squared = jnp.maximum(unsafe_phi_squared, jnp.array(0.01) ** 2)
    phi = jnp.sqrt(phi_squared)

    c = phi * jnp.sin(phi) / (1.0 - jnp.cos(phi))

    trans = jnp.where(
        phi_squared == unsafe_phi_squared,
        ((1.0 - 0.5 * c) / (phi ** 2)) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + 0.5 * c * jnp.eye(3),
        (1.0 / 12.0) * q.reshape(1, 3) * q.reshape(3, 1)
        + 0.5 * spin_matrix(q)
        + jnp.eye(3),
    )

    return jnp.concatenate(
        (
            jnp.concatenate((jnp.eye(3), jnp.zeros((3, 3))), axis=1),
            jnp.concatenate((jnp.zeros((3, 3)), trans), axis=1),
        ),
        axis=0,
    )


def force(xq):
    # Compare with equations: Evensen2008.10; Verweij2021.1-4,8,9
    q = xq[3:]
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    scale = jnp.where(
        phi == unsafephi,
        jnp.sin(phi) / (1.0 - jnp.cos(phi)) - 2.0 / phi,
        -unsafephi / 6.0,
    )

    metric_force = jnp.where(
        phi == unsafephi, kBT * (q / phi) * scale, jnp.array([0.0, 0.0, 0.0])
    )

    gravity_force = jnp.array(
        [0.0, 0.0, -(4.0 / 3.0) * jnp.pi * g * (rho_p - rho_f) * radius ** 3]
    )

    position_1 = (
        radius * rotation_matrix_r(q)[:, 2]	# position of a single sphere to c.m. !
    )
    
    position_2 = -position_1

    electrostatic_force_1 = jnp.array(
        [
            0.0,
            0.0,
            64
            * jnp.pi
            * epsilon
            * kappa
            * radius
            * (kBT / e) ** 2
            * jnp.tanh(e * psi_w / (4 * kBT))
            * jnp.tanh(e * psi_p / (4 * kBT))
            * jnp.exp(-kappa * (xq[2] + position_1[2])),
        ]
    )
    electrostatic_force_2 = jnp.array(
        [
            0.0,
            0.0,
            64
            * jnp.pi
            * epsilon
            * kappa
            * radius
            * (kBT / e) ** 2
            * jnp.tanh(e * psi_w / (4 * kBT))
            * jnp.tanh(e * psi_p / (4 * kBT))
            * jnp.exp(-kappa * (xq[2] + position_2[2])),
        ]
    )

    trans_force_1 = electrostatic_force_1 + gravity_force
    trans_force_2 = electrostatic_force_2 + gravity_force

    torque = (
        transformation_matrix(xq)[3:, 3:]
        @ rotation_matrix_r(q).T
        @ (jnp.cross(position_1, trans_force_1) + jnp.cross(position_2, trans_force_2))
    )

    return jnp.concatenate(
        (trans_force_1 + trans_force_2, metric_force + torque), axis=None
    )
    
def dumbbell_potential(h, theta):
    # Potential on the dumbbell.
    # Compare with Verweij2021.10-13
    gravity_force = (- 4.0 / 3.0) * jnp.pi * g * (rho_p - rho_f) * radius ** 3
            
    electrostatic_force = (64.0
            * jnp.pi
            * epsilon
            * kappa
            * radius
            * (kBT / e) ** 2.0
            * jnp.tanh(e * psi_w / (4.0 * kBT))
            * jnp.tanh(e * psi_p / (4.0 * kBT))
            * jnp.exp(-kappa * h))
            
    return 2.0 * (- gravity_force * h 
        + (electrostatic_force 
        * np.cosh(kappa * radius * np.sin(theta))) / kappa)

        
def h_probability(h_range):
    h_prob = np.zeros(len(h_range))
    
    for i, h in enumerate(h_range):
        if h > 2 * radius:
            theta_range = np.linspace(-np.pi/2, np.pi/2, 100)
        elif h < radius:
            theta_range = np.linspace(0, 0, 100)
        else:
            th_crit = np.arcsin(-1 + h/radius)
            theta_range = np. linspace(-th_crit, th_crit, 100)
        dtheta = theta_range[1] - theta_range[0]
        for j, theta in enumerate(theta_range):
    	    h_prob[i] += dtheta * np.cos(theta) * np.exp(-dumbbell_potential(h, theta)/kBT)
    	    
    return h_prob
    
def theta_probability(theta_range):
    theta_prob = np.zeros(len(theta_range))
    
    for i, theta in enumerate(theta_range):
        h_range = np.linspace(radius * (1 + np.abs(np.sin(theta))), 5.0e-6, 200)
        dh = h_range[1] - h_range[0]
        for j, h in enumerate(h_range):
    	    theta_prob[i] += dh * np.exp(-dumbbell_potential(h, theta)/kBT)
    	    
    return theta_prob


def t_mobility(xq):
    # Mobility matrix transformed to coordinates.
    # Compare with equation: Evensen2008.2
    return (transformation_matrix(xq) 
    @ (rotation_matrix(xq).T)
    @ mobility(xq) 
    @ rotation_matrix(xq)
    @ (transformation_matrix(xq).T))


def drift(xq):
    # Drift term.
    # Compare with equation: Evensen2008.5
    # jax.jacobian has differentiation index last (like mu_ij d_k) so divergence is contraction of first and last axis.
    return t_mobility(xq) @ force(xq) + kBT * jnp.einsum(
        "iji->j", jax.jacobian(t_mobility)(xq)
    )


def noise(xq):
    # Noise term.
    # Compare with equation: Evensen2008.5
    mobility_d = jnp.linalg.cholesky(mobility(xq))
    return jnp.sqrt(2 * kBT) * transformation_matrix(xq) @ (rotation_matrix(xq).T) @ mobility_d


def canonicalize_coordinates(xq):
    q = xq[3:]
    unsafephi = jnp.sqrt(jnp.sum(q ** 2))
    phi = jnp.maximum(unsafephi, jnp.array(0.01))

    max_phi = jnp.pi
    canonical_phi = jnp.fmod(phi + max_phi, 2.0 * max_phi) - max_phi

    q = jax.lax.select(
        phi > max_phi, (canonical_phi / phi) * q, q  # and phi == unsafephi
    )

    return jnp.concatenate((xq[:3], q), axis=None)

# Setting an initial position of the particle
p_th = jnp.pi / 4.0
p_phi = jnp.pi
p_phia = 0.
initial_position = jnp.array([0., 0., 2.5e-6, p_phi * jnp.sin(p_th) * jnp.cos(p_phia),
    p_phi * jnp.sin(p_th) * jnp.sin(p_phia),
    p_phi * jnp.cos(p_th)])

# Stating the problem and solving its SDEs
problem = pychastic.sde_problem.SDEProblem(
    drift, noise, tmax=20.0, x0=initial_position
)


solver = pychastic.sde_solver.SDESolver(dt=0.01, scheme="euler")

trajectories = solver.solve_many(
    problem,
    step_post_processing=canonicalize_coordinates,
    n_trajectories=10000,
    chunk_size=8,
    chunks_per_randomization=2,
)

t_n = trajectories["time_values"][0]
t_t = jnp.arange(0.0, trajectories["time_values"][0][-1], 0.005)

# Sample trajectories
plt.figure(0)

plt.plot(t_n, 1E6 * trajectories["solution_values"][0, :, 2])
plt.plot(t_n, 1E6 * trajectories["solution_values"][1, :, 2])
plt.plot(t_n, 1E6 * trajectories["solution_values"][2, :, 2])
plt.plot(t_n, 1E6 * trajectories["solution_values"][3, :, 2])
plt.plot(t_n, 1E6 * trajectories["solution_values"][4, :, 2])
plt.plot(t_n, 1E6 * trajectories["solution_values"][5, :, 2])
plt.hlines(0, 0, t_n[-1], linewidth=2, color="k")

plt.xlabel("Time $t$ [s]")
plt.xlim(0, trajectories["time_values"][0, -1])
plt.ylabel("Height $h_{c.m.}$ [$\mu$m]")
plt.title("Sample trajectories")

plt.savefig("sample_heights_approx_small.png")

# Heights histogram
fig, ax1 = plt.subplots()
heights = np.zeros(len(trajectories["solution_values"][:,-1,2]))
for i in range(len(trajectories["solution_values"][:,-1,2])):
    heights[i] = 1E6 * trajectories["solution_values"][i,-1,2]
ax1.hist(heights, bins = 30, range = (0.0, 4.0), density = True)
ax1.set_xlabel("Height $h_{c.m.}$ [$\mu$m]")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
h_range = np.linspace(0.0, 1e-6 * 4.0, 1000)
h_theor = h_probability(h_range)
ax2.plot(1e6 * h_range, h_theor, 'r')
ax2.set_ylim(bottom = 0.0)
#plt.xlim(0.0, 4.0)
plt.title("Heights distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("heights_approx_small.png")

# Angles histogram
fig, ax1 = plt.subplots()
thetas = np.zeros(len(trajectories["solution_values"][:,-1,2]))
for i in range(len(trajectories["solution_values"][:,-1,2])):
    vec = rotation_matrix_r(trajectories["solution_values"][i,-1,3:])[:,2]
    rho = np.maximum(np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2), np.array(0.001))
    thetas[i] = np.arccos(vec[2] / rho) - np.pi/2
ax1.hist(thetas, bins = 30, density = True)
ax1.set_xlabel("Angle $\Theta_p$")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
theta_range = np.linspace(-np.pi/2.0, np.pi/2.0, 100)
theta_theor = theta_probability(theta_range)
ax2.plot(theta_range, theta_theor, 'r')
ax2.set_ylim(bottom = min(theta_theor))
plt.title("Angular distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("angles_approx_small.png")

plt.figure(3)
plt.plot(heights, thetas, 'bo', markersize = 1)
angles = np.linspace(-np.pi/2, np.pi/2, 100)
plt.plot((1 + np.abs(np.sin(angles))) * 1E6 * radius, angles, 'r')
plt.xlim(1e6 * radius, 5)
plt.ylim(-np.pi/2.0, np.pi/2.0)
plt.xlabel("Height $h_{c.m.}$ [$\mu$m]")
plt.ylabel("Angle $\Theta_p$")
plt.title("Heights-angles distribution for $d=1.1$ $\mu$m, analytical approx.")
plt.savefig("heights_angles_approx_small.png")

with open('trajectories_approx_small.dat', 'w') as f:
	for i in range(len(trajectories["solution_values"][:, 1, 1])):
		f.write(str(trajectories["solution_values"][i, -1, :]) + '\n')
	f.close()
	
with open('heights_approx_small.dat', 'w') as f:
	for i in range(len(heights)):
		f.write(str(heights[i]) + '\n')
	f.close()
	
with open('angles_approx_small.dat', 'w') as f:
	for i in range(len(thetas)):
		f.write(str(thetas[i]) + '\n')
	f.close()
	
with open('heights_theor_range.dat', 'w') as f:
	for i in range(len(h_range)):
		f.write(str(h_range[i]) + '\n')
	f.close()
	
with open('heights_theor_values.dat', 'w') as f:
	for i in range(len(h_theor)):
		f.write(str(h_theor[i]) + '\n')
	f.close()

with open('angles_theor_range.dat', 'w') as f:
	for i in range(len(theta_range)):
		f.write(str(theta_range[i]) + '\n')
	f.close()
	
with open('angles_theor_values.dat', 'w') as f:
	for i in range(len(theta_theor)):
		f.write(str(theta_theor[i]) + '\n')
	f.close()

'''
fig, ax1 = plt.subplots()
for i in range(len(heights) - 1, 0, -1):
    if heights[i] < 1e6 * radius * (1 + np.sin(np.absolute(thetas[i]))) :
    	heights = np.delete(heights, i)
    	thetas = np.delete(thetas, i)
ax1.hist(heights, bins = 30, range = (0.0, 4.0), density = True)
ax1.set_xlabel("Height $h_{c.m.}$ [$\mu$m]")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
ax2.plot(1e6 * h_range, h_theor, 'r')
ax2.set_ylim(bottom = 0.0)
plt.savefig("z_hist_popped.png")  

plt.figure(5)
plt.plot(heights, thetas, 'bo', markersize = 1)
angles = np.linspace(-np.pi/2, np.pi/2, 100)
plt.plot((1 + np.sin(np.absolute(angles))) * 1E6 * radius, angles, 'r')
plt.savefig("h_theta_popped.png")

fig, ax1 = plt.subplots()
ax1.hist(thetas, bins = 30, density = True)
ax1.set_xlabel("Angles $\Theta_p$")
ax1.set_ylabel("PDF")
ax2 = plt.twinx()
ax2.plot(theta_range, theta_theor, 'r')
ax2.set_ylim(bottom = min(theta_theor))
plt.title("Angle distribution")
plt.savefig("thetas_p_hist_popped.png")


# Rotational-rotational correlations.
# Compare with equation: Cichocki2015.71
rotation_matrices = jax.vmap(jax.vmap(rotation_matrix_r))(trajectories["solution_values"][:,:,3:])
rotation_matrices = jnp.einsum(
    "ij,abjk", (rotation_matrix_r(problem.x0[3:]).T), rotation_matrices
)

epsilon_tensor = jnp.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
    ]
)

delta_u = -0.5 * jnp.einsum("kij,abij->abk", epsilon_tensor, rotation_matrices)
r_cor = jnp.mean(delta_u ** 2, axis=0)

plt.figure(0)

plt.plot(t_n, r_cor[:, 0], label = "00-num.")
plt.plot(t_n, r_cor[:, 1], label = "11-num.")
plt.plot(t_n, r_cor[:, 2], label = "22-num.")

D1 = mu[5,5]
D3 = mu[3,3]

plt.plot(
    t_t,
    1.0 / 6.0
    + (1.0 / 12.0) * jnp.exp(-6.0 * D1 * t_t)
    - (1.0 / 2.0) * jnp.exp(-(2.0 * D1 + 4.0 * D3) * t_t)
    + (1.0 / 4.0) * jnp.exp(-2.0 * D1 * t_t),
    label="00-theor.",
)

plt.plot(
    t_t,
    1.0 / 6.0
    - (1.0 / 6.0) * jnp.exp(-6.0 * D1 * t_t)
    - (1.0 / 4.0) * jnp.exp(-(5.0 * D1 + D3) * t_t)
    + (1.0 / 4.0) * jnp.exp(-(D1 + D3) * t_t),
    label="11,22-theor.",
)

plt.title("$<\Delta u(t) \Delta u(t)>_0$")
plt.legend()

plt.savefig("dudu.png")

# Translational-translational correlations.
# Compare with equation: Cichocki2015.57
x_0 = jnp.swapaxes(
    jnp.full(
        (len(t_n), trajectories["solution_values"].shape[0], 3),
        problem.x0[:3]
    ),
    0, 1
    )

delta_R = trajectories["solution_values"][:,:,:3] - x_0
t_cor = jnp.mean(delta_R ** 2, axis = 0)

plt.figure(1)
plt.plot(t_n, t_cor[:, 0], label = "00-num.")
plt.plot(t_n, t_cor[:, 1], label = "11-num.")
plt.plot(t_n, t_cor[:, 2], label = "22-num.")

D3 = mu[0,0]
D1 = mu[2,2]
D = (2.0 * D1 + D3)/3.0

plt.plot(
    t_t,
    2.0 * D * t_t
    + 2.0 * (1.0 - jnp.exp(-6.0 * D1 * t_t)) * (D1 - D3) / (18.0 * D1),
    label = "00-theor."
)

plt.plot(
    t_t,
    2.0 * D * t_t
    - 2.0 * (1.0 - jnp.exp(-6.0 * D1 * t_t)) * (D1 - D3) / (9.0 * D1),
    label = "11,22-theor."
)

plt.title("$<\Delta R(t) \Delta R(t)>_0$")
plt.legend()

plt.savefig("dRdR.png")

# Translational-rotational correlations.
# Compare with equation: Cichocki2015.81
delta_u_delta_R = delta_u * delta_R
tr_cor = jnp.mean(delta_u_delta_R, axis = 0)

plt.figure(2)
plt.plot(t_n, tr_cor[:, 0], label = "00-num.")
plt.plot(t_n, tr_cor[:, 1], label = "11-num.")
plt.plot(t_n, tr_cor[:, 2], label = "22-num.")

plt.plot(
    t_t,
    0 * t_t,
    label = "theoretical"
)

plt.title("$<\Delta u(t) \Delta R(t)>_0$")
plt.legend()

plt.savefig("dudR.png")
'''
