#!/usr/bin/env python
# coding: utf-8

# # Diffusion equation:
# This is the diffusion equation: $$ \frac{\partial z}{\partial t} = D\frac{\partial^2 z}{\partial x^2} $$ and $$ z^{t+1}_x = z^t_x + {D \Delta t \over \Delta x^2} (z^t_{x+1} - 2z^t_x + z^t_{x-1}) $$Assuming constant diffusivity, a regular grid, and a step function for the initial profile. Here, $z$ is elevation, $t$ is time, $x$ is the spatial dimension and $D$ is a transport rate coefficient, sometimes called hillslope diffusivity.
# 
# This is the forward in time centered-in-space discretization scheme.

# In[2]:


import os
import sys
import tomllib

import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def calculate_stable_time_step(dx, diffusivity):
    return 0.5 * dx**2 / diffusivity


def step_like(x, step_at=0):
    y = np.empty_like(x, dtype=float)

    y[:step_at] = 0.0
    y[step_at] = 0.5
    y[step_at + 1 :] = 1.0

    return y


def boxcar_like(x, step_at=0):
    return step_like(x, step_at=step_at) - step_like(x, step_at=len(x) - step_at - 1)


def new_profile(x, form="step"):
    match form:
        case "step":
            return step_like(x, step_at=len(x) // 2)
        case "boxcar":
            return boxcar_like(x, step_at=len(x) // 4)
        case "bowl":
            return 1.0 - boxcar_like(x, step_at=len(x) // 4)
        case "wedding":
            return boxcar_like(x, step_at=len(x) // 8) + boxcar_like(
                x, step_at=3 * len(x) // 8
            )
        case _:
            raise ValueError(f"unknown profile type ({form!r})")


def plot_profile(x, elevation, color="r"):
    plt.figure()
    plt.plot(x, elevation, color)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Hillslope profile")


def calculate_second_derivative(y, dx=1.0):
    d2y_dx2 = np.empty_like(y)

    d2y_dx2[1:-1] = (y[:-2] - 2 * y[1:-1] + y[2:]) / dx**2
    d2y_dx2[0] = 0.0
    d2y_dx2[-1] = 0.0

    return d2y_dx2


def diffuse_until(y_initial, stop_time, dx=1.0, diffusivity=1.0):
    stable_dt = 0.9 * calculate_stable_time_step(dx, diffusivity)

    y = y_initial.copy()

    time_remaining = stop_time
    while time_remaining > 0.0:
        dt = min(time_remaining, stable_dt)

        y += diffusivity * dt * calculate_second_derivative(y, dx=dx)

        time_remaining -= dt

    return y


def run_diffusion_model(
    diffusivity=100.0, width=100.0, stop_time=1.0, n_points=81, profile="step"
):
    x, dx = np.linspace(0, width, num=n_points, retstep=True)
    initial_elevation = new_profile(x, form=profile)

    elevation = diffuse_until(
        initial_elevation, stop_time, dx=dx, diffusivity=diffusivity
    )

    plot_profile(x, initial_elevation, "g")
    plot_profile(x, elevation, "r")
    plt.show()

    return elevation


def load_params_from_path(filepath):
    with open(filepath, "rb") as stream:
        params = tomllib.load(stream)
    return params


# In[14]:


get_ipython().run_line_magic('pip', 'install mpl_ascii')


# In[4]:


if __name__ == "__main__":
    import matplotlib as mpl
    import mpl_ascii

    mpl_ascii.AXES_WIDTH = 70
    mpl_ascii.AXES_HEIGHT = 15

    mpl.use("module://mpl_ascii")

    filepath = "diffusion.toml"

    if os.path.isfile(filepath):
        params = load_params_from_path(filepath)
    else:
        params = {}
    elevation = run_diffusion_model(**params)

    np.savetxt(sys.stdout, elevation, fmt="%.6f")


# In[7]:


D = 100
Lx= 300


# In[8]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x) # number of grid nodes

print(x[:100])
print(nx)


# Setting up the initial conditions.

# In[9]:


z = np.zeros_like(x)
z_left = 500
z_right = 0
z[x <= Lx / 2] = z_left
z[x > Lx / 2] = z_right

print(f"{z[:100]}" + "\n" + f"{z[500:]}")


# Plot the initial profile.

# In[10]:


plt.figure()
plt.plot(x, z, "r")
plt.ylabel("z")
plt.xlabel("x")
plt.title("Initial hillslope profile")
plt.show()


# In[12]:


nt = 5000
dt = 0.5 * dx**2 / D
print(dt)


# In[ ]:


Loop over the time steps of the model.


# In[13]:


for t in range(0, nt):
	z[1:-1] += D * dt / dx ** 2 * (z[:-2] - 2*z[1:-1] + z[2:])


# In[15]:


plt.figure()
plt.plot(x, z, "r")
plt.ylabel("z")
plt.xlabel("x")
plt.title("Final hillslope profile")
plt.show()


# In[ ]:




