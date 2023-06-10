#!/usr/bin/env python3
##### This file contains a script to test run cooling simulations defined in the CoolingSimulation.jl module
include("../CoolingSimulation.jl")

# Define dipole polarizailities
const alpha_H2 = 0.787*1e-30 # [4*pi*eps0*m**3]
const alpha_H2O = 1.501*1e-30 # [4*pi*eps0*m**3]
const alpha_N2 = 1.71*1e-30 # [4*pi*eps0*m**3]
const alpha_CO = 1.953*1e-30 # [4*pi*eps0*m**3]
const alpha_CO2 = 2.507*1e-30 # [4*pi*eps0*m**3]

# Define global parameters
const B = [0.,0.,7.]

# Define ion initial conditions
q = e.val
m = 23*m_u.val
N_ions = 100
μ_z0 = -0.125
σ_z0 = 0.005
σ_xy0 = 0.001
μ_E0_par, σ_E0_par = 80., 8.
σ_E0_perp = 0.5

# Define plasma parameters
n_b = 1e10*1e06*0.1
T_b = 300.
q_b = -e.val 
m_b = m_e.val

# Define residual gas parameters
neutral_masses = [28*m_u.val]
neutral_pressures_mbar = [5e-09*20.]
alphas = [alpha_N2]
CX_fractions = [0.0] 
T_n = 300. 

# Define run parameters
t_end = 370.0e-06
dt = 1e-08
sample_every = 1000
velocity_diffusion = true

##### Run test simulation
@time ion_orbits = integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                        μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q=q, m=m, N_ions=N_ions, B=B, 
                                        n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                        neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                        alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                        t_end=t_end, dt=dt, sample_every=sample_every, 
                                        velocity_diffusion=velocity_diffusion);