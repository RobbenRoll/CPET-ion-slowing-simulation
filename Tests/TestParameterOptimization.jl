using LinearAlgebra
include("../Diagnostics.jl")

# Define convenience functions for grabbing data from position and velocity histories
get_detectable_N_ions(A) = [countnotnans(A[:,it,:]) for it in range(1, size(A)[2])]
get_mean(A) = transpose(nanmean(A, dims=1))
get_std(A) = transpose(nanstd(A, dims=1))
get_err_of_mean(A) = get_std(A)./sqrt.(get_detectable_N_ions(A))
get_RMSD(A) = transpose(sqrt.(nanmean(A.^2, dims=1)))

"""Nansum with penalty of +10 for each NaN in input vector"""
function penalized_nansum(v) 
    s = nansum(v)
    for vi in v 
        if isnan(vi)
            s += 10.
        end
    end
    return s 
end 

function loss(sample_times, position_hists, velocity_hists, 
              charge_hists, mass_hists; q_b=-e.val, r_b=0.0, T_b=300, max_detectable_r=8e-04, 
              n_smooth_E=51, ramp_correction=true, exp_data_fname=nothing)
    """Calculate loss function"""
    get_E_par(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,3])^2/charge_hists[ion_id,it] + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)
    get_E_tot(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,:])^2/charge_hists[ion_id,it]  + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)


    if mod(n_smooth_E,2) == 1
        n_smooth_half = Int64(floor(n_smooth_E/2)) # half-length of sliding smoothing window
    else
        throw("The sample number `n_smooth_E` must be an odd integer.")
    end
    
    # Determine ion species to use for fetching ramp correction data
    mean_m0_u = mean(mass_hists[:,1])/m_u.val
    if isapprox(mean_m0_u, 23, atol=0.49)
        species = "23Na"
    elseif isapprox(mean_m0_u, 39.1, atol=0.49)
        species = "39K"
    elseif isapprox(mean_m0_u, 85.3, atol=0.49)
        species = "85Rb"
    else 
        species = ""
    end
    V_sitp = get_V_sitp(r_b) # load potential map
    N_ions = size(position_hists)[1]
    
    # Fetch and prepare experimental data from .npy file
    if !isnothing(exp_data_fname)
        exp_data = get_exp_data(exp_data_fname)
        times_exp = get(exp_data,"Interaction times")/1000 # [s]
        if r_b > 0 # fetch plasma-on data
            mean_E_par_exp = get(exp_data,"mean_E unmuted") # [eV/q]
            err_mean_E_par_exp = get(exp_data,"Error mean_E unmuted") # [eV/q]
            std_E_par_exp = get(exp_data,"sigma_E unmuted") # [eV/q]
            N_ions_exp = get(exp_data, "N_ions unmuted")
            err_N_ions_exp = get(exp_data, "Error N_ions unmuted")
        else # fetch plasma-off data
            mean_E_par_exp = get(exp_data,"mean_E muted") # [eV/q]
            err_mean_E_par_exp = get(exp_data,"Error mean_E muted") # [eV/q]
            std_E_par_exp = get(exp_data,"sigma_E muted") # [eV/q]
            N_ions_exp = get(exp_data, "N_ions muted")
            err_N_ions_exp = get(exp_data, "Error N_ions muted")
        end
    end
     
    # Determine effective nest depth and set threshold energy for determining 
    # ions localized in entrance-side potential well
    V_nest_eff = mean(V_itp_on_axis(Vector(0:0.001:0.025), V_sitp))
    V_thres = V_nest_eff - 5*k_B.val*T_b/abs(q_b) 
    
    t_end = sample_times[end]
    it_eval = [argmin(abs.(sample_times .- t)) for t in times_exp if t <= t_end]
    E_par = []
    for i in range(1,N_ions)
        E_par_i = [mean(get_E_par.(i, it-n_smooth_half:it+n_smooth_half)) for it in it_eval]  
        if ramp_correction # correct ion energies for endcap voltage ramp
            E_par_i = apply_ramp_correction(E_par_i, V_nest_eff, species=species)
        end
        push!(E_par, E_par_i)
    end 
    E_par = transpose(stack(E_par)) # idx1: ion number, idx2: time step
    z = [mean(position_hists[i,it-n_smooth_half:it+n_smooth_half,3]) for it in it_eval, i in range(1,N_ions)]
    r = [mean(sqrt.(position_hists[i,it-n_smooth_half:it+n_smooth_half,1].^2 .+ position_hists[i,it-n_smooth_half:it+n_smooth_half,2].^2)) for it in it_eval, i in range(1,N_ions)]
    z = transpose(z)
    r = transpose(r)
          
    detectable = ((E_par .> V_thres .|| z .> 0.0) .&& r .<= max_detectable_r) # bool-mask for detectable ions
    detectable_E_par = nanmask(E_par, @. !detectable)
    detectable_N_ions = get_detectable_N_ions(detectable_E_par) 
    detectable_mean_E_par = get_mean(detectable_E_par) 
    detectable_std_E_par = get_std(detectable_E_par) 
    
    E_par_loss = penalized_nansum( ((detectable_mean_E_par .- mean_E_par_exp[1:length(it_eval)])./mean_E_par_exp[1:length(it_eval)]).^2 )
    std_E_par_loss = penalized_nansum( ((detectable_std_E_par .- std_E_par_exp[1:length(it_eval)])./std_E_par_exp[1:length(it_eval)]).^2 )
    N_ions_loss = sum( ((detectable_N_ions .- N_ions_exp[1:length(it_eval)])./N_ions_exp[1:length(it_eval)]).^2 )
    total_loss = sum( E_par_loss + std_E_par_loss + 0.1*N_ions_loss )

    println(E_par_loss, " ", std_E_par_loss, " ", N_ions_loss)
    return total_loss #E_par_loss, std_E_par_loss, N_ions_loss
end

# Compile atomic masses from https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html (AME2020)
# Masses given to micro-u precision
alkali_mass_data = Dict("Na" => ([22.989769], [1.]), 
                        "K"  => ([38.963706, 40.961825], [0.933, 0.067]),
                        "Rb" => ([84.911790, 86.909180], [0.722, 0.278]) ) 
                        
exp_data_fname_Na = "RFA_results_run04343_Na23_final.npy"
exp_data_fname_K = "RFA_results_run04354_K39_final.npy"
exp_data_fname_Rb = "RFA_results_run04355_Rb85_final.npy"

# Test loss function
# using HDF5
# include("../Diagnostics.jl")
# rel_path="/OutputFiles/"
# fname = "2023-06-28_1636_test_run_Na_ions_plasma_off.h5"

# run_info = get_run_info(fname; rel_path="/Tests/" * rel_path)

# path = string(@__DIR__) * rel_path * fname 
# fid = h5open(path, "r")
# orbs = fid["IonOrbits"] 
# sample_times = read(orbs["sample_time_hists"])[1,:]
# position_hists = read(orbs["position_hists"])
# velocity_hists = read(orbs["velocity_hists"])
# charge_hists = read(orbs["charge_hists"])
# mass_hists = read(orbs["mass_hists"])
                        
# @time loss(sample_times, position_hists, velocity_hists, 
#            charge_hists, mass_hists, q_b=-e.val, r_b=0.0, T_b=300,
#            max_detectable_r=8e-04, n_smooth_E=51, exp_data_fname=exp_data_fname_Na)

using Dates
using Distributed
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
include("../IonNeutralCollisions.jl")
include("../CoolingSimulation.jl")

# Define global parameters
const B = [0.,0.,7.]

# Define ion initial conditions
q = e.val
m0_u = [23]
m0_probs = [1.]
const N_ions = 8
const μ_z0 = -0.125
const σ_z0 = 0.003
const σ_xy0 = 0.00025
const μ_E0_par, σ_E0_par = 84., 13.
const σ_E0_perp = 0.5

# Define plasma parameters
n_b = 0.0 #1e07*1e06
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
n_procs = 8
t_end = 37.00e-03
dt = 3e-08 # TODO: Reduce again!
sample_every = 200
seed = 85383
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")
output_path = "Tests/OutputFiles/" * datetime * "test_run_plasma_off"

##### Run test simulation
function eval_plasma_off_loss(x; q=e.val, m0_u=[23], m0_probs=[1.], q_b=-e.val, r_b=0.0, T_b=300, 
                              exp_data_fname=nothing, max_detectable_r=8e-04, n_smooth_E=51, n_procs=n_procs)    
    neutral_pressures_mbar = x[1:4]
    σ_xy0 = x[5]

    orbits =  integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                   μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, B=B, 
                                   n_b=0.0, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b,
                                   neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                   alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, seed=seed, 
                                   t_end=t_end, dt=dt, sample_every=sample_every, n_procs=n_procs,
                                   velocity_diffusion=velocity_diffusion, fname=nothing)
      
    sample_times = orbits[1][1,:]
    position_hists = orbits[2]
    velocity_hists = orbits[3]
    charge_hists = orbits[4]
    mass_hists = orbits[5]
    
    loss_val = loss(sample_times, position_hists, velocity_hists, 
                    charge_hists, mass_hists, q_b=q_b, r_b=r_b, T_b=T_b,
                    n_smooth_E=n_smooth_E, exp_data_fname=exp_data_fname)

    return loss_val
end

function eval_combined_plasma_off_loss(x; q=q, q_b=q_b, T_b=T_b, max_detectable_r=8e-04, n_smooth_E=51, n_procs=n_procs)
    r_b=0.0 # turn plasma off 
    loss_val_Na = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["Na"][1], m0_probs=alkali_mass_data["Na"][2],               
                                       exp_data_fname=exp_data_fname_Na, q_b=q_b, r_b=r_b, T_b=T_b,
                                       max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_procs=n_procs)
    
    loss_val_K = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["K"][1], m0_probs=alkali_mass_data["K"][2],               
                                      exp_data_fname=exp_data_fname_K, q_b=q_b, r_b=r_b, T_b=T_b,
                                      max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_procs=n_procs)
    
    loss_val_Rb = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["Rb"][1], m0_probs=alkali_mass_data["Rb"][2],               
                                       exp_data_fname=exp_data_fname_Rb, q_b=q_b, r_b=r_b, T_b=T_b,
                                       max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_procs=n_procs)
    println()
    println(x)
    println([loss_val_Na, loss_val_K, loss_val_Rb])
    return sum([loss_val_Na, loss_val_K, loss_val_Rb])
end

x = push!(neutral_pressures_mbar, σ_xy0)
@time eval_combined_plasma_off_loss(x; max_detectable_r=8e-04, n_smooth_E=51, n_procs=n_procs)


# # ###### BlackBoxOptim 
# # ### multiobjective.jl
# # using BlackBoxOptim, Gadfly
# # using LinearAlgebra

# # # run Borg MOAE
# # guess = neutral_pressures_mbar
# # res = bboptimize(eval_plasma_off_orbits, guess; Method=:borg_moea,
# #                  FitnessScheme=ParetoFitnessScheme{3}(is_minimizing=true),
# #                  SearchRange=(1e-10, 1e-07), NumDimensions=length(guess), ϵ=0.1,
# #                  MaxSteps=15, TraceInterval=1.0, TraceMode=:verbose);

# ##### Surrogates.jl - AbstractGPs
# ### Optimization example
# using Surrogates
# using Plots
# using AbstractGPs
# using SurrogatesAbstractGPs

# n_samples = 30
# lower_bounds = [1e-10,1e-10,1e-10,5e-11, 0.0001]
# upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09, 0.0005]
# #xs = minimum(lower_bounds):5e-10:maximum(upper_bounds)
# x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, SobolSample())
    
# y = eval_combined_plasma_off_loss.(x)
# gp_surrogate = AbstractGPSurrogate(x,y)

# # Plot samples from surrogate GaussianProcess
# f = scatter(getindex.(gp_surrogate.x,1), gp_surrogate.y, label="Sampled points", 
#             xlabel="H2 pressure (mbar)", ylabel="Loss")
# #plot!(xs[1], gp_surrogate.(xs), label="Surrogate function", ribbon=p->std_error_at_point(gp_surrogate, p), legend=:top)
# savefig(f, "SurrogateGC_samples_H2.png") 
# #display(f)

# f = scatter(getindex.(gp_surrogate.x,2), gp_surrogate.y, label="Sampled points", 
#             xlabel="H2O pressure (mbar)", ylabel="Loss")
# savefig(f, "SurrogateGC_samples_H2O.png") 
# #display(f)

# f = scatter(getindex.(gp_surrogate.x,3), gp_surrogate.y, label="Sampled points", 
#             xlabel="N2 pressure (mbar)", ylabel="Loss")
# savefig(f, "SurrogateGC_samples_N2.png") 
# #display(f)

# f = scatter(getindex.(gp_surrogate.x,4), gp_surrogate.y, label="Sampled points", 
#             xlabel="CO2 pressure (mbar)", ylabel="Loss")
# savefig(f, "SurrogateGC_samples_CO2.png") 
# #display(f)

# f = scatter(getindex.(gp_surrogate.x,5), gp_surrogate.y, label="Sampled points", 
#             xlabel="σ_xy0 (m)", ylabel="Loss")
# savefig(f, "SurrogateGC_samples_sigma_xy0.png") 

# @time best_values = surrogate_optimize(eval_combined_plasma_off_loss, SRBF(), lower_bounds, upper_bounds, gp_surrogate, SobolSample())

# println(best_values)


##### Surrogates.jl - Kriging
### Optimization example
using Surrogates
using Plots
using HDF5

n_samples = 10
lower_bounds = [1e-10, 1e-10, 1e-10, 5e-11, 0.0001]
upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09, 0.0004]
p = [1.5, 1.5, 1.5, 1.5, 1.5]
sampling_func = SobolSample()
acquisition_func = EI()
maxiters = 30 
num_new_samples = 30

output_fname = "optimization_results"

#xs = minimum(lower_bounds):5e-10:maximum(upper_bounds)
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, sampling_func)
println(x)
y = eval_combined_plasma_off_loss.(x)
println(y)

# Build surrogate
theta = 0.5 / max(1e-6 * abs(upper_bounds - lower_bounds), std(x))^p) # default from Kriging.jl
surrogate = Kriging(x, y, lower_bounds, upper_bounds, p=p, , theta=theta, sampling_func)
println(surrogate.x)
println(surrogate.y)

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

# Plot samples from surrogate GaussianProcess
f = scatter(getindex.(surrogate.x,1), surrogate.y, label="Sampled points", 
            xlabel="H2 pressure (mbar)", ylabel="Loss")
#plot!(xs[1], gp_surrogate.(xs), label="Surrogate function", ribbon=p->std_error_at_point(gp_surrogate, p), legend=:top)
savefig(f, "Surrogate_samples_H2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,2), surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss")
savefig(f, "Surrogate_samples_H2O.png") 
#display(f)

f = scatter(getindex.(surrogate.x,3), surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss")
savefig(f, "Surrogate_samples_N2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,4), surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss")
savefig(f, "Surrogate_samples_CO2.png") 
#display(f)

f = scatter(getindex.(surrogate.x,5), surrogate.y, label="Sampled points", 
            xlabel="σ_xy0 (m)", ylabel="Loss")
savefig(f, "Surrogate_samples_sigma_xy0.png") 

@time res = surrogate_optimize(eval_combined_plasma_off_loss, acquisition_func, lower_bounds, upper_bounds, surrogate, sampling_func;         
                               maxiters=maxiters, num_new_samples=num_new_samples)

println(res)
println(x)
println(y)

# Write final sample data to attributes 
fid = h5open(output_fname * ".h5", "r+")
create_group(fid, "samples")
samples = fid["samples"]
samples["inputs"] = surrogate.x
samples["combined_loss_vals"] = surrogate.y
close(fid)
println("Final sample data written to " * output_fname * ".h5")