#!/usr/bin/julia
##### This script is to test run cooling simulations defined in the CoolingSimulation.jl module
using ProfileView
using Dates
using Distributed
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
include("../IonNeutralCollisions.jl")

# Compile atomic masses from https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html (AME2020)
# Masses given to micro-u precision
alkali_mass_data = Dict("Na" => ([22.989769], [1.]), 
                        "K"  => ([38.963706, 40.961825], [0.933, 0.067]),
                        "Rb" => ([84.911790, 86.909180], [0.722, 0.278]) ) 

# Define global parameters
const B = [0.,0.,7.]
const m_e_u = m_e.val/m_u.val

# Define ion initial conditions
q0 = e.val
species = "K"
m0_u, m0_probs = alkali_mass_data[species]  
m0_u = m0_u .- round(q0/e.val)*m_e_u # correct for electron mass, neglect atomic binding energies
const N_ions = 8
const μ_z0 = -0.125
const σ_z0 = 0.003
const σ_xy0 = 0.00025
const μ_E0_par, σ_E0_par = 83., 8.
const σ_E0_perp = 0.5

# Define plasma parameters
n_b = 1e07*1e06 #
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val
const r_b = 0.0 #0.001 # plasma radius for grabbing potential and electron density data

# Define residual gas parameters
neutral_masses = [2*m_u.val, 18*m_u.val, 28*m_u.val, 44*m_u.val] # H2, H2O, N2, CO2 # H2, H2O, CO, CO2 
neutral_pressures_mbar = [5.719607782346591e-9, 1.9596895785235832e-10, 1.1069845782347077e-10, 3.270300387651214e-10]#), 2.990275706930693 #[0.3, 0.38, 0.3, 0.02]*2.78e-09 #[0.35, 0.25, 0.35, 0.05]*2.87e-09  #[0.3, 0.2, 0.3, 0.02]*1.8e-09  #[0.80, 0.10, 0.05, 0.02]*4.4e-09 #[0.80, 0.10, 0.05, 0.05]*3.15e-09 #[0.70, 0.10, 0.10, 0.10]*3.7e-09 #[0.38, 0.20, 0.32, 0.10]*2.45e-09 #[5e-10, 5e-10, 4e-10, 4e-10]
alphas = [alpha_H2, alpha_H2O, alpha_N2, alpha_CO2]
CX_fractions = [0., 0., 0., 0.] 
T_n = 300. 

# Define run parameters
n_procs = 9
t_end = 3700e-03
dt = 1e-08
sample_every = 5000
seed = 85383
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")
output_path = "Tests/OutputFiles/" * datetime * "test_run_" * species * "_ions_plasma_off_test_n_e_map"

##### Run test simulation
addprocs(n_procs)
@everywhere include("../CoolingSimulation.jl")

println("Starting compilation run")
#using ProfileView
# ProfileView.@profview
@time integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                           μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q0, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, 
                           B=B, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b,
                           neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                           alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                           t_end=3*dt, dt=dt, sample_every=1, 
                           velocity_diffusion=velocity_diffusion);
println("Starting simulation run")
@time integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                           μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q0, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, 
                           B=B, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b, 
                           neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                           alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                           t_end=t_end, dt=dt, sample_every=sample_every, seed=seed,
                           velocity_diffusion=velocity_diffusion, fname=output_path);