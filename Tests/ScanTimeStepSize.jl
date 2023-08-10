#!/usr/bin/julia
##### This script is to test run cooling simulations defined in the CoolingSimulation.jl module
#using ProfileView
using Dates
using Formatting
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
include("../IonNeutralCollisions.jl")
mbar_to_Torr = 0.7500616827 

# Compile atomic masses from https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html (AME2020)
# Masses given to micro-u precision - List isotopic masses and respective relative natural abundances for 
# each isotope of an element (neglecting isotopes with < 1% abundance)
alkali_mass_data = Dict("Na" => ([22.989769], [1.]), 
                        "K"  => ([38.963706, 40.961825], [0.933, 0.067]),
                        "Rb" => ([84.911790, 86.909180], [0.722, 0.278]) ) 

# Define global parameters
const B = [0.,0.,7.]
const m_e_u = m_e.val/m_u.val

# Define ion initial conditions
q0 = e.val
species = "Na"
m0_u, m0_probs = alkali_mass_data[species]  
m0_u = m0_u .- round(q0/e.val)*m_e_u # correct for electron mass, neglect atomic binding energies
const N_ions = 1
const μ_z0 = -0.123
const σ_z0 = 0.0
const σ_xy0 =  0.00029 #0.00033 #0.00039 #0.00018 #0.0001796875 #0.00025
const μ_E0_par, σ_E0_par = 80., 16.
const σ_E0_perp = 0.5

# Define plasma parameters
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val
const r_b = 0.0 #0.001 # plasma radius for grabbing potential and electron density data (set to 0 to turn plasma off)

# Define residual gas parameters
neutral_masses = [2*m_u.val, 18*m_u.val, 28*m_u.val, 44*m_u.val] # H2, H2O, N2, CO2 # H2, H2O, CO, CO2 
neutral_pressures_mbar = [0., 0., 0., 0.] #[3.96e-9, 6.42e-10, 4.63e-10, 1.55e-10] #[5.43e-09, 1.60e-10, 1.46e-10, 1.39e-10] #[5.43e-9, 1.6e-10, 1.5e-10, 1.4e-10] #[4.2765625e-9, 7.890624999999999e-10, 4.828125e-10, 4.5078125e-10] #[5.719607782346591e-9, 1.9596895785235832e-10, 1.1069845782347077e-10, 3.270300387651214e-10]#), 2.990275706930693 #[0.3, 0.38, 0.3, 0.02]*2.78e-09 #[0.35, 0.25, 0.35, 0.05]*2.87e-09  #[0.3, 0.2, 0.3, 0.02]*1.8e-09  #[0.80, 0.10, 0.05, 0.02]*4.4e-09 #[0.80, 0.10, 0.05, 0.05]*3.15e-09 #[0.70, 0.10, 0.10, 0.10]*3.7e-09 #[0.38, 0.20, 0.32, 0.10]*2.45e-09 #[5e-10, 5e-10, 4e-10, 4e-10]
gas_corr_factors = [0.46,1.12,1.00,1.42] # ion gauge gas correction factors from https://www.duniway.com/images/_pg/ion-gauge-gas-correction-factors.pdf
alphas = [alpha_H2, alpha_H2O, alpha_N2, alpha_CO2]
CX_fractions = [0., 0., 0., 0.] 
T_n = 300. 
total_pressure_mbar = sum(neutral_pressures_mbar)
detectable_total_pressure_mbar = sum(neutral_pressures_mbar .* gas_corr_factors)
printfmt("True total pressure:        {:.2e} mbar / {:.2e} Torr \n", total_pressure_mbar, total_pressure_mbar*mbar_to_Torr)
printfmt("Detectable total pressure:  {:.2e} mbar / {:.2e} Torr \n", detectable_total_pressure_mbar, detectable_total_pressure_mbar*mbar_to_Torr)

# Define trajectory tracing parameters
n_workers = 1
t_end = 1.000e-03
dt = 1e-08
sample_every = 1
seed = 85383
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")

##### Run test simulation
include("../CoolingSimulation.jl")

println("Starting compilation run")
#using ProfileView
# ProfileView.@profview
@time integrate_ion_orbits(;μ_E0_par=μ_E0_par, σ_E0_par=σ_E0_par, σ_E0_perp=σ_E0_perp, 
                           μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q0, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, 
                           B=B, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b,
                           neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                           alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, seed=seed,
                           t_end=3*dt, dt=dt, sample_every=1, n_workers=n_workers,
                           velocity_diffusion=velocity_diffusion);

time_step_sizes = [0.5e-08, 1.0e-08, 1.5e-08, 2.0e-08, 2.5e-08]
for dt in time_step_sizes
    output_path = "Tests/OutputFiles/" * datetime * "test_run_" * species * "_ions_r_b_" * string(r_b*1e03) * "mm_dt_" * string(round(dt*1e09)) * "ns"

    println("Starting simulation run")
    @time integrate_ion_orbits(;μ_E0_par=μ_E0_par, σ_E0_par=σ_E0_par, σ_E0_perp=σ_E0_perp,
                                μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q0, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, 
                                B=B, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b, 
                                neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, seed=seed,
                                t_end=t_end, dt=dt, sample_every=sample_every, n_workers=n_workers, 
                                velocity_diffusion=velocity_diffusion, fname=output_path);
end