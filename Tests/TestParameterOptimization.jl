#!/usr/bin/julia
##### Optimization based on Surrogates.jl - Kriging
include("../ParameterOptimization.jl")
using Surrogates
using Plots
using HDF5

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


# Define loss function 
loss(x) = eval_combined_plasma_off_loss(x; orbit_tracing_kws=orbit_tracing_kws, 
                                        max_detectable_r=max_detectable_r, 
                                        n_smooth_E=n_smooth_E, seed=seed, 
                                        scale_time_step=scale_time_step)
#loss(x) = g(x) # for sped-up testing 

# Define surrogate optimization parameters
n_samples = 50
lower_bounds = [5e-10, 1e-10, 1e-10, 5e-11, 0.0001] # [5e-10, 0.0001] 
upper_bounds =  [1e-08, 5e-09, 5e-09, 1e-09, 0.0004] # [1e-08, 0.0004]
p = [1.5, 1.5, 1.5, 1.5, 1.5] # [1.5, 1.5]
sampling_func = GoldenSample() # SobolSample() #  # LatinHypercubeSample()
acquisition_func = SRBF() #EI()
maxiters = 50 
num_new_samples = 300
noise_variance = 1.23^2

output_fname = "optimization_results" 

# Create samples
u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
xs = [(ui,x0[2],x0[3],x0[4],x0[5]) for ui in u] 
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, sampling_func)
println(x)
y = loss.(x)
println(y)

# Build surrogate
theta = [0.5 / max(1e-6 * norm(upper_bounds .- lower_bounds), std(x_i[i] for x_i in x))^p[i] for i in 1:length(x[1])] # default from Kriging.jl
surrogate = Kriging(x, y, lower_bounds, upper_bounds, p=p, theta=theta, noise_variance=noise_variance)
println("Samples:")
println(surrogate.x)
println("Loss values:")
println(surrogate.y)
println(xs)
ys = surrogate.(xs)
println(ys)
println(1e-6 * norm(maximum(surrogate.y) - minimum(surrogate.y)))

# Write initial sample data to attributes 
fid = h5open(output_fname * ".h5", "w")
create_group(fid, "optimizer_pars")
optimizer_pars = fid["optimizer_pars"]
optimizer_pars["lower_bounds"] = lower_bounds
optimizer_pars["upper_bounds"] = upper_bounds
optimizer_pars["p"] = p
optimizer_pars["theta"] = theta
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
function get_contour_levels(zz, n_levels = 30) 
    levels = Vector(0.0:1.1*maximum(zz)/n_levels:1.1*maximum(zz))
    return levels
end 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[2], upper_bounds[2], 1000)
ZZ = [surrogate([ui,vi,x0[3],x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"H_2O \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,2), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_loss_contour_H2_and_H20_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[3], upper_bounds[3], 1000)
ZZ = [surrogate([ui,x0[2],vi,x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"N_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,3), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_loss_contour_H2_and_N2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[4], upper_bounds[4], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],vi,x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels,
             margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"CO_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,4), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_loss_contour_H2_and_CO2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[5], upper_bounds[5], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],x0[4],vi]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"σ_{xy,0} \;\mathrm{(m)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,5), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_loss_contour_H2_pressure_and_sigma_xy.png") 

yerrs = std_error_at_point.(surrogate, xs)
f = scatter(getindex.(surrogate.x,1), surrogate.y, label="Sampled points", ylim=(1e-03,1e02),
            xlabel="H2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm)
plot!(getindex.(xs,1), ys, label="Surrogate function", legend=:top, ribbon=yerrs)
plot!(getindex.(xs,1), g.(xs), label="True function")
savefig(f, "Surrogate_samples_H2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,2), surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_samples_H2O.png") 
#display(f)

f = scatter(getindex.(surrogate.x,3), surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_samples_N2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,4), surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_samples_CO2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,5), surrogate.y, label="Sampled points", 
            xlabel="σ_xy0 (m)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_samples_sigma_xy0.png") 

# Run surrogate optimization
println(pathof(Surrogates))
@time res = surrogate_optimize(loss, acquisition_func, lower_bounds, upper_bounds, surrogate, sampling_func; maxiters=maxiters, num_new_samples=num_new_samples)

println(res)
#println(x)
#println(y)


# Plot final samples from surrogate GaussianProcess
u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[2], upper_bounds[2], 1000)
ZZ = [surrogate([ui,vi,x0[3],x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"H_2O \;\mathrm{pressure}\;\mathrm{(mbar)}", margin=10Plots.mm, left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,2), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_final_loss_contour_H2_and_H20_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[3], upper_bounds[3], 1000)
ZZ = [surrogate([ui,x0[2],vi,x0[4],x0[5]]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels,
    margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"N_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,3), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_final_loss_contour_H2_and_N2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[4], upper_bounds[4], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],vi,x0[5]]) for vi in v, ui in u]
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"CO_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,4), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_final_loss_contour_H2_and_CO2_pressure.png") 

u = LinRange(lower_bounds[1], upper_bounds[1], 1000)
v = LinRange(lower_bounds[5], upper_bounds[5], 1000)
ZZ = [surrogate([ui,x0[2],x0[3],x0[4],vi]) for vi in v, ui in u] 
levels = get_contour_levels(ZZ)
f = contourf(u, v, ZZ, levels=levels, margin=10Plots.mm, title=L"\mathrm{Loss}", xlabel=L"H_2 \;\mathrm{pressure}\;\mathrm{(mbar)}", ylabel=L"σ_{xy0} \;\mathrm{(m)}", left_margin=18Plots.mm)
scatter!(getindex.(surrogate.x,1), getindex.(surrogate.x,5), marker_z=surrogate.y, label="Samples")
savefig(f, "Surrogate_final_loss_contour_H2_pressure_and_sigma_xy.png") 

yerrs = std_error_at_point.(surrogate, xs)
f = scatter(getindex.(surrogate.x,1), surrogate.y, label="Sampled points", ylim=(1e-03,1e02),
            xlabel="H2 pressure (mbar)", ylabel="Loss", #yscale=:log10, 
            margin=10Plots.mm)
plot!(getindex.(xs,1), surrogate.(xs), label="Surrogate function", legend=:top, ribbon=yerrs)
plot!(getindex.(xs,1), g.(xs), label="True function")
savefig(f, "Surrogate_final_samples_H2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,2), surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_final_samples_H2O.png") 
#display(f)

f = scatter(getindex.(surrogate.x,3), surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_final_samples_N2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,4), surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_final_samples_CO2.png") 
# #display(f)

f = scatter(getindex.(surrogate.x,5), surrogate.y, label="Sampled points", 
            xlabel="σ_xy0 (m)", ylabel="Loss", yscale=:log10, margin=10Plots.mm)
savefig(f, "Surrogate_final_samples_sigma_xy0.png") 

# Write final sample data to attributes 
fid = h5open(output_fname * ".h5", "r+")
create_group(fid, "samples")
samples = fid["samples"]
samples["inputs"] = surrogate.x
samples["combined_loss_vals"] = surrogate.y
close(fid)
println("Final sample data written to " * output_fname * ".h5")