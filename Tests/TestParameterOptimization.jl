#!/usr/bin/julia
##### Optimization based on Surrogates.jl - Kriging
include("../ParameterOptimization.jl")
using Surrogates
using Plots
using HDF5
using BlackBoxOptim

# Define global parameters
const B = [0.,0.,7.]
const max_detectable_r = 8e-04

# Define ion initial conditions
q0 = e.val
m0_u = [23]
m0_probs = [1.]
const N_ions = 15
const μ_z0 = -0.125
const σ_z0 = 0.003
const μ_E0_par, σ_E0_par = 84., 13.
const σ_E0_perp = 0.5

# Define plasma parameters
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val
const r_b = 0.001 # plasma radius for grabbing potential data

# Define residual gas parameters
neutral_masses = [2*m_u.val, 18*m_u.val, 28*m_u.val, 44*m_u.val] # H2, H20, N2, CO2 
neutral_pressures_mbar = [0.80, 0.10, 0.05, 0.05]*3.15e-09 #[0.70, 0.10, 0.10, 0.10]*3.7e-09 #[0.38, 0.20, 0.32, 0.10]*2.45e-09 #[5e-10, 5e-10, 4e-10, 4e-10]
alphas = [alpha_H2, alpha_H2O, alpha_N2, alpha_CO2]
CX_fractions = [0., 0., 0., 0.] 
T_n = 300. 

# Define run parameters
n_workers = 15
t_end = 3700e-03
dt = 1e-08 # applies to Na only if scale_time_step == true
scale_time_step = true
sample_every = 200
seed = 85383
velocity_diffusion = true
n_smooth_E = 51

orbit_tracing_kws = Dict(:μ_E0_par => μ_E0_par, :σ_E0_par => σ_E0_par, :σ_E0_perp => σ_E0_perp, 
                         :μ_z0 => μ_z0, :σ_z0 => σ_z0, :q0 => q0, :m0_u => m0_u, :m0_probs => m0_probs, 
                         :N_ions => N_ions, :B => B, :T_b => T_b, :q_b => q_b, :m_b => m_b, :r_b => r_b,
                         :neutral_masses => neutral_masses, :neutral_pressures_mbar => neutral_pressures_mbar, 
                         :alphas => alphas, :CX_fractions => CX_fractions, :T_n => T_n, :seed => seed, 
                         :t_end => t_end, :dt => dt, :sample_every => sample_every, 
                         :velocity_diffusion => velocity_diffusion, :n_workers => n_workers)


# Define surrogate optimization parameters
n_samples = 50
lower_bounds = [5e-10, 1e-10, 1e-10, 1e-10, 0.0001] # [5e-10, 0.0001] 
upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09, 0.0004] # [1e-08, 0.0004]
p = [1.99, 1.99, 1.99, 1.99, 1.99] # [1.5, 1.5]
sampling_func = SobolSample() #LatinHypercubeSample() # GoldenSample() 
acquisition_func = SRBF() # EI()
maxiters = 30 
num_new_samples = 300
noise_variance = 1.66^2 #0.1^2 

output_fname = "optimization_results" 

# Define loss function 
loss(x) = eval_combined_plasma_off_loss(x; orbit_tracing_kws=orbit_tracing_kws, 
                                        max_detectable_r=max_detectable_r, 
                                        n_smooth_E=n_smooth_E, seed=seed, 
                                        scale_time_step=scale_time_step)
#loss(x) = mock_loss(x; noise_variance=noise_variance) # for sped-up testing 

# Create samples
u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
xs = [(ui,x0[2],x0[3],x0[4],x0[5]) for ui in u] 
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, sampling_func)
y = loss.(x) 

# Build Kriging surrogate
println( [0.5 / std(x_i[i] for x_i in x)^p[i] for i in 1:length(x[1])] )
println(0.5 / (1e-6 * norm(upper_bounds .- lower_bounds)) )
theta = [0.5 / max(1e-6 * norm(upper_bounds[i] - lower_bounds[i]), std(x_i[i] for x_i in x))^p[i] for i in 1:length(x[1])] # default from Kriging.jl
initial_surrogate = Kriging(x, y, lower_bounds, upper_bounds, p=p, theta=theta, noise_variance=noise_variance)
println("Samples:")
println(initial_surrogate.x)
println("Loss values:")
println(initial_surrogate.y)

# Optimize Kriging hyperparameters (keep noise-variance constant)
N_dims = length(x[1])
#p_bounds = [(1.0,1.99) for _ in range(1,N_dims)]
theta = [1. / (1e-09)^p[i] for i in range(1,N_dims)]
theta[5] = 1. / (1e-04)^p[5]
theta_bounds = [(1. / (5e-09)^p[i], 1. / (1e-09)^p[i]) for i in range(1,N_dims)] #[(0.1*theta[i], 1.1*theta[i]) for i in range(1,N_dims)]  #[(1e-01*theta[i], 1e06*theta[i]) for i in range(1,N_dims)] 
#theta_bounds[1] = ( 1. / (5e-09)^p[1], 1. / (1e-09)^p[1])
#theta_bounds[2] = ( 1. / (5e-09)^p[2], 1. / (1e-09)^p[2])
#theta_bounds[3] = ( 1. / (5e-09)^p[3], 1. / (1e-09)^p[3])
#theta_bounds[4] = ( 1. / (5e-09)^p[4], 1. / (1e-09)^p[4])
theta_bounds[5] = ( 1. / (1e-03)^p[5], 1. / (5e-05)^p[5])
ranges = theta_bounds #append!(p_bounds, theta_bounds)
initial_guess = theta #append!(p, theta)
#initial_guess = append!(initial_guess, noise_variance)
#ranges = append!(ranges, (1e-06, 1e03))

"""Log-likelihood for Kriging hyperparameter optimization"""
function log_likelihood(hyper_pars; x=x, y=y, p=p, noise_variance=noise_variance)
    #p = hyper_pars[1:5]
    #theta = hyper_pars[6:10]
    theta = hyper_pars[1:5]
    #noise_variance = hyper_pars[11]
    return Surrogates.kriging_log_likelihood(x, y, p, theta, noise_variance)
end

println("Initial hyperparameters [theta's]: ", initial_guess)
println("1/sqrt(theta):", 1 ./ sqrt.(theta))
println("Initial log-likelihood: ", log_likelihood(initial_guess))

hyper_par_res = bboptimize(log_likelihood, initial_guess; SearchRange = ranges, 
                           NumDimensions = 2*N_dims, 
                           MaxFuncEvals = 1000, TraceInterval = 0.05)
best_hyper_pars = best_candidate(hyper_par_res)
#p = best_hyper_pars[1:N_dims]
theta = best_hyper_pars[1:N_dims] #best_hyper_pars[N_dims+1:2*N_dims] #####################################
println("Optimized hyperparameters [theta's]: ", best_hyper_pars)
println("1/sqrt(theta):", 1 ./ sqrt.(theta))

# Plot initial samples from surrogate
u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[2], upper_bounds[2], 1000)
ZZ = [initial_surrogate([ui,vi,x0[3],x0[4],x0[5]]) for vi in v, ui in u] 
function get_contour_levels(zz, n_levels = 60) 
    levels = Vector(0.0:1.1*maximum(zz)/n_levels:1.1*maximum(zz))
    return levels
end 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", 
             xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"H_2O \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(initial_surrogate.x,1), getindex.(initial_surrogate.x,2), marker_z=initial_surrogate.y, label="Samples")
savefig(f, "Surrogate_loss_contour_H2_and_H20_pressure_before_hyper_parameter_optimization.png") 

# Re-build surrogate with optimized hyperparameters
surrogate = Kriging(x, y, lower_bounds, upper_bounds, theta=theta, noise_variance=noise_variance)

# Write initial sample data to attributes 
fid = h5open(output_fname * ".h5", "w")
create_group(fid, "optimizer_pars")
optimizer_pars = fid["optimizer_pars"]
optimizer_pars["lower_bounds"] = lower_bounds
optimizer_pars["upper_bounds"] = upper_bounds
optimizer_pars["p"] = p
optimizer_pars["theta"] = theta
optimizer_pars["noise_variance"] = noise_variance
optimizer_pars["sampling_func"] = string(sampling_func)
optimizer_pars["acquisition_func"] = string(acquisition_func)
optimizer_pars["maxiters"] = maxiters
optimizer_pars["num_new_samples"] = num_new_samples
create_group(fid, "initial_samples")
init_samples = fid["initial_samples"]
init_samples["inputs"] = surrogate.x
init_samples["combined_loss_vals"] = surrogate.y
close(fid)
println("Initial sample data written to " * output_fname * ".h5")

# Plot initial samples from surrogate
function get_contour_levels(zz, n_levels = 60) 
    levels = Vector(0.0:1.1*maximum(zz)/n_levels:1.1*maximum(zz))
    return levels
end 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[2], upper_bounds[2], 1000)
ZZ = [surrogate([ui,vi,x0[3],x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, 
             margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             ylabel=L"H_2O \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,2), marker_z=surrogate.y, label="Samples")
scatter!([x0[1]], [x0[2]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_loss_contour_H2_and_H20_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[3], upper_bounds[3], 1000)
ZZ = [surrogate([ui,x0[2],vi,x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, lw=0, title=L"\mathrm{Loss}", 
             xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"N_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,3), marker_z=surrogate.y, label="Samples")
scatter!([x0[1]], [x0[3]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_loss_contour_H2_and_N2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[4], upper_bounds[4], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],vi,x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, 
             margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             ylabel=L"CO_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,4), marker_z=surrogate.y, label="Samples")
scatter!([x0[1]], [x0[4]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_loss_contour_H2_and_CO2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[5], upper_bounds[5], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],x0[4],vi]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, lw=0, title=L"\mathrm{Loss}", 
             xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"σ_{xy,0} \;\mathrm{(m)}", 
             left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,5), marker_z=surrogate.y, label="Samples")
scatter!([x0[1]], [x0[5]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_loss_contour_H2_pressure_and_sigma_xy.png") 

ys = surrogate.(xs)
yerrs = std_error_at_point.(surrogate, xs)
f = scatter(getindex.(surrogate.x,1), surrogate.y, label="Sampled points", ylim=(1e-03,1e02),
            xlabel="H2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
plot!(getindex.(xs,1), ys, label="Surrogate function", legend=:top, ribbon=yerrs)
plot!(getindex.(xs,1), mock_loss.(xs), label="True function")
savefig(f, "Surrogate_samples_H2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,2), surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_samples_H2O.png") 
#display(f)

f = scatter(getindex.(surrogate.x,3), surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_samples_N2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,4), surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_samples_CO2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,5), surrogate.y, label="Sampled points", 
            xlabel="σ_xy0 (m)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_samples_sigma_xy0.png") 

# Run surrogate optimization
println(pathof(Surrogates))
@time res = surrogate_optimize(loss, acquisition_func, lower_bounds, upper_bounds, surrogate, 
                               UniformSample(); maxiters=maxiters, num_new_samples=num_new_samples)
best_values = res[1]

println(res)
#println(x)
#println(y)

# Plot final samples from surrogate GaussianProcess
u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[2], upper_bounds[2], 1000)
ZZ = [surrogate([ui,vi,x0[3],x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             ylabel=L"H_2O \;\mathrm{pressure}\;\mathrm{(mbar)}", margin=10Plots.mm, left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,2), marker_z=surrogate.y, label="Samples")
scatter!([best_values[1]], [best_values[2]], marker=:circ, markercolor=:blue, label="Best values")
scatter!([x0[1]], [x0[2]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_final_loss_contour_H2_and_H20_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[3], upper_bounds[3], 1000)
ZZ = [surrogate([ui,x0[2],vi,x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, 
             margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             ylabel=L"N_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,3), marker_z=surrogate.y, label="Samples")
scatter!([best_values[1]], [best_values[3]], marker=:circ, markercolor=:blue, label="Best values")
scatter!([x0[1]], [x0[3]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_final_loss_contour_H2_and_N2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[4], upper_bounds[4], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],vi,x0[5]]) for vi in v, ui in u]
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, margin=10Plots.mm, title=L"\mathrm{Loss}", 
             xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"CO_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", 
             left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,4), marker_z=surrogate.y, label="Samples")
scatter!([best_values[1]], [best_values[4]], marker=:circ, markercolor=:blue, label="Best values")
scatter!([x0[1]], [x0[4]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_final_loss_contour_H2_and_CO2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[5], upper_bounds[5], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],x0[4],vi]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, lw=0, margin=10Plots.mm, title=L"\mathrm{Loss}", 
             xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"σ_{xy0} \;\mathrm{(m)}", 
             left_margin=18Plots.mm, dpi=300)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,5), marker_z=surrogate.y, label="Samples")
scatter!([best_values[1]], [best_values[5]], marker=:circ, markercolor=:blue, label="Best values")
scatter!([x0[1]], [x0[5]], marker=:cross, markercolor=:red, label="True values")
savefig(f, "Surrogate_final_loss_contour_H2_pressure_and_sigma_xy.png") 

yerrs = std_error_at_point.(surrogate, xs)
f = scatter(getindex.(surrogate.x,1), surrogate.y, label="Sampled points", ylim=(1e-03,1e02),
            xlabel="H2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
plot!(getindex.(xs,1), surrogate.(xs), label="Surrogate function", legend=:top, ribbon=yerrs)
plot!(getindex.(xs,1), mock_loss.(xs), label="True function")
savefig(f, "Surrogate_final_samples_H2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,2), surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_final_samples_H2O.png") 
#display(f)

f = scatter(getindex.(surrogate.x,3), surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_final_samples_N2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,4), surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_final_samples_CO2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,5), surrogate.y, label="Sampled points", 
            xlabel="σ_xy0 (m)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm, dpi=300)
savefig(f, "Surrogate_final_samples_sigma_xy0.png") 

# Write final sample data to attributes 
fid = h5open(output_fname * ".h5", "r+")
create_group(fid, "samples")
samples = fid["samples"]
samples["inputs"] = surrogate.x
samples["combined_loss_vals"] = surrogate.y
close(fid)
println("Final sample data written to " * output_fname * ".h5")