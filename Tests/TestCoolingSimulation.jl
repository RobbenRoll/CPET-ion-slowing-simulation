#!/usr/bin/julia
##### This script is to test run cooling simulations defined in the CoolingSimulation.jl module
using ProfileView
# using Base.Threads
# println("No. of threads: " * string(nthreads()))
using Dates
using Distributed
addprocs(9)
println("number of cores = ", nprocs())
println("number of workers = ", nworkers())
workers()
@everywhere include("../CoolingSimulation.jl")

# Define global parameters
const B = [0.,0.,7.]

# Define ion initial conditions
q = e.val
m = 23*m_u.val
const N_ions = 100
const μ_z0 = -0.125
const σ_z0 = 0.005
const σ_xy0 = 0.001
const μ_E0_par, σ_E0_par = 80., 8.
const σ_E0_perp = 0.5

# Define plasma parameters
n_b = 1e07*1e06
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val

# Define residual gas parameters
neutral_masses = [28*m_u.val]
neutral_pressures_mbar = [5e-09]
alphas = [alpha_N2]
CX_fractions = [0.0] 
T_n = 300. 

# Define run parameters
t_end = 3700e-03
dt = 1e-08
sample_every = 10000 #1000
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")
fname = "Tests/OutputFiles/" * datetime * "test_run_plasma_on"

##### Run test simulation
using ProfileView

println("Starting compilation run")
# ProfileView.@profview
@time ion_orbits = integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                        μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q=q, m=m, N_ions=N_ions, B=B, 
                                        n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                        neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                        alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                        t_end=sample_every*dt, dt=dt, sample_every=sample_every, 
                                        velocity_diffusion=velocity_diffusion);
println("Starting profiling run")
@time ion_orbits = integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                        μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q=q, m=m, N_ions=N_ions, B=B, 
                                        n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                        neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                        alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                        t_end=t_end, dt=dt, sample_every=sample_every, 
                                        velocity_diffusion=velocity_diffusion, fname=fname);