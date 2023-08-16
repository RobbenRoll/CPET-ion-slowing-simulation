#!/usr/bin/julia
##### Optimization based on Surrogates.jl - Kriging
using Surrogates
using Plots
using HDF5
include("../ParameterOptimization.jl")

# Define global parameters
const B = [0.,0.,7.]

# Define ion initial conditions
q0 = e.val
m0_u = [23]
m0_probs = [1.]
const N_ions = 50
const μ_z0 = -0.125
const σ_z0 = 0.003
const μ_E0_par, σ_E0_par = 80., 16.
const σ_E0_perp = 0.5

# Define plasma parameters
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val
const r_b = 0.0 # plasma radius for grabbing potential data

# Define residual gas parameters
neutral_masses = [2*m_u.val, 18*m_u.val, 28*m_u.val, 44*m_u.val] # H2, H20, N2, CO2 
neutral_pressures_mbar = [0.80, 0.10, 0.05, 0.05]*3.15e-09 #[0.70, 0.10, 0.10, 0.10]*3.7e-09 #[0.38, 0.20, 0.32, 0.10]*2.45e-09 #[5e-10, 5e-10, 4e-10, 4e-10]
alphas = [alpha_H2, alpha_H2O, alpha_N2, alpha_CO2]
CX_fractions = [0., 0., 0., 0.] 
T_n = 300. 

# Define run parameters
n_workers = 15
t_end = 3700e-03
dt = 1e-08 
sample_every = 5000
seed = nothing # SET SEED TO NOTHING TO ALLOW FOR RANDOM FLUCTUATIONS IN INITIAL CONDITIONS
velocity_diffusion = true
max_detectable_r = 8e-04
n_smooth_E = 51 

orbit_tracing_kws = Dict(:μ_E0_par => μ_E0_par, :σ_E0_par => σ_E0_par, :σ_E0_perp => σ_E0_perp, 
                         :μ_z0 => μ_z0, :σ_z0 => σ_z0, :q0 => q0, :m0_u => m0_u, :m0_probs => m0_probs, 
                         :N_ions => N_ions, :B => B, :T_b => T_b, :q_b => q_b, :m_b => m_b, :r_b => r_b,
                         :neutral_masses => neutral_masses, :neutral_pressures_mbar => neutral_pressures_mbar, 
                         :alphas => alphas, :CX_fractions => CX_fractions, :T_n => T_n, :seed => nothing, 
                         :t_end => t_end, :dt => dt, :sample_every => sample_every, 
                         :velocity_diffusion => velocity_diffusion, :n_workers => n_workers)


# Define loss function 
loss(x) = eval_combined_plasma_off_loss(x; orbit_tracing_kws=orbit_tracing_kws, 
                                        max_detectable_r=max_detectable_r, 
                                        n_smooth_E=n_smooth_E, seed=seed)

# Define surrogate optimization parameters
n_samples = 20
lower_bounds = [5e-10, 1e-10, 1e-10, 5e-11, 0.0001] # [5e-10, 0.0001] 
upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09, 0.0004] # [1e-08, 0.0004]
p = [1.5, 1.5, 1.5, 1.5, 1.5] # [1.5, 1.5]
sampling_func = GoldenSample() # SobolSample() #  # LatinHypercubeSample()
acquisition_func = SRBF() #EI()

output_fname = "results_evaluate_cooling_sim_noise_50_ions"

u = LinRange(lower_bounds[1], upper_bounds[1], 100)
xs = [(ui,x0[2],x0[3],x0[4],x0[5]) for ui in u] 
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, sampling_func)
println(x[1])
x_test = [(3.96e-9, 6.42e-10, 4.63e-10, 1.55e-10, 0.000116) for _ in range(1,n_samples)] #[x[1] for _ in range(1,n_samples)]
println(x_test)
y = loss.(x_test)
println("Loss function values: ", y)
println("Std. dev. of loss function values: ", std(y))

# # Evaluate elementary effects of initial samples and calculate corresponding statistical measures
# println(elementary_effect(loss_func, x0, lower_bounds, upper_bounds))
# x_samples = [[xi[k] for k in range(1,length(xi))] for xi in xs]
# elem_effects = [elementary_effect(xi, lower_bounds, upper_bounds) for xi in x_samples]
# mean_elem_effects = [mean(getindex.(elem_effects,i)) for i in range(1,length(elem_effects[1]))]
# std_elem_effects = [std(getindex.(elem_effects,i)) for i in range(1,length(elem_effects[1]))]
# println(mean_elem_effects)
# println(std_elem_effects)

# f = plot(mean_elem_effects, std_elem_effects, xlabel="Mean of elementary effects", ylabel="Std. dev. of elementary effects")
# savefig(f, "elementary_effects_map.png")