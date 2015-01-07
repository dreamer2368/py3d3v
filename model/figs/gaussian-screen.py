
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# Generate short and long range force values
r = np.linspace(.01, 1, 100)
beta_vals = np.arange(1, 5)
sqpi = np.sqrt(np.pi)
E_vals = []
F_vals = []
for beta in beta_vals:
    E = sqpi*erf(r*beta)/(2*r**2)-beta*np.exp(-beta**2*r**2)/r
    E = E/(2*sqpi**3)
    E_vals.append((E, beta))
    F = 1/(4*np.pi*r**2) - E
    F_vals.append((F, beta))

# Plot short range forces
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1,2,1)
for E, beta in E_vals:
    ax.plot(r, E, label="$\\beta=%i$"%(beta,))
ax.set_title("Field produced by Gaussian screen")
ax.set_xlabel("$r$")
ax.set_ylabel("$E(r)$")
ax.legend()

# Plot long range forces
#fig = plt.figure()
ax = fig.add_subplot(122)
for F, beta in F_vals:
    ax.semilogy(r, F, label="$\\beta=%i$"%(beta,))
ax.set_title("Short range force using Gaussian screen")
ax.set_xlabel("$r$")
ax.set_ylabel("$E(r)$")
ax.legend(loc="lower left")
fig.savefig("p3m-gaussian-fields.pdf")

# CIC
x_vals = np.linspace(-1, 1, 1000)
s_vals = np.zeros_like(x_vals)

s_vals[np.abs(x_vals)<.5] = 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_vals, s_vals)
ax.set_ylim((0, 1.1))
ax.set_title("CIC Particle Shape")
fig.savefig("cic.pdf")
