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
    println("mean_m0_u: ",mean_m0_u)
    # Increased atol as compared to Diagnostics.jl to account for sampling error:
    if isapprox(mean_m0_u, 23, atol=0.99)
        species = "23Na"
    elseif isapprox(mean_m0_u, 39.1, atol=0.99) 
        species = "39K"
    elseif isapprox(mean_m0_u, 85.3, atol=0.99)
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
const N_ions = 15
const μ_z0 = -0.125
const σ_z0 = 0.003
const σ_xy0 = 0.00025
const μ_E0_par, σ_E0_par = 78., 16.
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
dt = 3e-08 # TODO: Reduce again!
sample_every = 5000
seed = nothing # SET SEED TO NOTHING TO ALLOW FOR RANDOM FLUCTUATIONS IN INITIAL CONDITIONS
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")
output_path = "Tests/OutputFiles/" * datetime * "test_run_plasma_off"

using Distributions: Normal
import Random
x0 =  [3.2e-09, 1e-09, 8e-10, 2e-10, 0.00020] # [3.2e-09, 0.00020] # True parameter values
function g(x; σ_noise=0.5)
    scale = [1e-09, 1e-09, 1e-09, 1e-09, 1e-04] # [1e-09, 1e-04]
    noise = rand(Normal(0, σ_noise))
    return sum( ((x .- x0) ./ scale ).^2) + noise
end

##### Run test simulation
function eval_plasma_off_loss(x; q=e.val, m0_u=[23.], m0_probs=[1.], q_b=-e.val, r_b=0.0, T_b=300, seed=seed, 
                              exp_data_fname=nothing, max_detectable_r=8e-04, n_smooth_E=51, n_workers=n_workers)    
   neutral_pressures_mbar = x[1:4]
   σ_xy0 = x[5]

    orbits =  integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                   μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, B=B, 
                                   T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b,
                                   neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                   alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, seed=seed, 
                                   t_end=t_end, dt=dt, sample_every=sample_every, n_workers=n_workers,
                                   velocity_diffusion=velocity_diffusion, fname=nothing)
      
    sample_times = orbits[1][1,:]
    position_hists = orbits[2]
    velocity_hists = orbits[3]
    charge_hists = orbits[4]
    mass_hists = orbits[5]
    
    loss_val = loss(sample_times, position_hists, velocity_hists, 
                    charge_hists, mass_hists, q_b=q_b, r_b=r_b, T_b=T_b,
                    n_smooth_E=n_smooth_E, exp_data_fname=exp_data_fname)
    
    # loss_val = g(x) # for sped-up testing 
    
    # Clean up shared arrays 
    finalize(sample_times)
    finalize(position_hists)
    finalize(velocity_hists)
    finalize(charge_hists)
    finalize(mass_hists)
    @everywhere GC.gc() # To prevent memory leakage and overfilling of /dev/shm

    return loss_val
end

function eval_combined_plasma_off_loss(x; q=q, q_b=q_b, T_b=T_b, max_detectable_r=8e-04, n_smooth_E=51, seed=seed, n_workers=n_workers)
    r_b=0.0 # turn plasma off 
    loss_val_Na = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["Na"][1], m0_probs=alkali_mass_data["Na"][2],               
                                       exp_data_fname=exp_data_fname_Na, q_b=q_b, r_b=r_b, T_b=T_b, seed=seed, 
                                       max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_workers=n_workers)
    
    loss_val_K = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["K"][1], m0_probs=alkali_mass_data["K"][2],               
                                      exp_data_fname=exp_data_fname_K, q_b=q_b, r_b=r_b, T_b=T_b, seed=seed, 
                                      max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_workers=n_workers)
    
    loss_val_Rb = eval_plasma_off_loss(x; q=q, m0_u=alkali_mass_data["Rb"][1], m0_probs=alkali_mass_data["Rb"][2],               
                                       exp_data_fname=exp_data_fname_Rb, q_b=q_b, r_b=r_b, T_b=T_b, seed=seed, 
                                       max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E, n_workers=n_workers)
    println()
    println(x)
    println([loss_val_Na, loss_val_K, loss_val_Rb])
    return sum([loss_val_Na, loss_val_K, loss_val_Rb])
end


##### Surrogates.jl - Kriging
### Optimization example
using Surrogates
using Plots
using HDF5

n_samples = 20
lower_bounds = [5e-10, 1e-10, 1e-10, 5e-11, 0.0001] # [5e-10, 0.0001] 
upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09, 0.0004] # [1e-08, 0.0004]
p = [1.5, 1.5, 1.5, 1.5, 1.5] # [1.5, 1.5]
sampling_func = GoldenSample() # SobolSample() #  # LatinHypercubeSample()
acquisition_func = SRBF() #EI()
maxiters = 50 
num_new_samples = 300

output_fname = "optimization_results"

u = LinRange(lower_bounds[1], upper_bounds[1], 100)
xs = [(ui,x0[2],x0[3],x0[4],x0[5]) for ui in u] 
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, sampling_func)
println(x[1])
x_test = [(5.43e-09, 1.60e-10, 1.46e-10, 1.39e-10, 0.00033) for _ in range(1,n_samples)] #[x[1] for _ in range(1,n_samples)]
println(x_test)
y = eval_combined_plasma_off_loss.(x_test)
println("Loss function values: ", y)
println("Std. dev. of loss function values: ", std(y))

# Define elementary effects
function elementary_effect(x0, lower_bounds, upper_bounds; delta=1/100)
    y0 = eval_combined_plasma_off_loss(x0)
    function shift_element(x, index, delta)
        x_new = deepcopy(x)
        x_new[index] += delta*(upper_bounds[index] - lower_bounds[index])
        return x_new
    end
    shifted_x = [shift_element(x0, i, delta) for i in range(1,length(x0))]
    d = (eval_combined_plasma_off_loss.(shifted_x) .- y0) / delta
    return d
end 

# # Evaluate elementary effects of initial samples and calculate corresponding statistical measures
# println(elementary_effect(x0, lower_bounds, upper_bounds))
# x_samples = [[xi[k] for k in range(1,length(xi))] for xi in xs]
# elem_effects = [elementary_effect(xi, lower_bounds, upper_bounds) for xi in x_samples]
# mean_elem_effects = [mean(getindex.(elem_effects,i)) for i in range(1,length(elem_effects[1]))]
# std_elem_effects = [std(getindex.(elem_effects,i)) for i in range(1,length(elem_effects[1]))]
# println(mean_elem_effects)
# println(std_elem_effects)

# f = plot(mean_elem_effects, std_elem_effects, xlabel="Mean of elementary effects", ylabel="Std. dev. of elementary effects")
# savefig(f, "elementary_effects_map.png")