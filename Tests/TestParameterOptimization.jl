using LinearAlgebra
function loss(sample_times, position_hists, velocity_hists, 
              charge_hists, mass_hists; r_b=0.0, max_detectable_r=8e-04, 
              n_smooth_E=51, ramp_correction=true, exp_data_fname=nothing)
    """Calculate loss function"""
    
    # Define convenience functions for grabbing data from position and velocity histories
    get_detectable_N_ions(A) = [countnotnans(A[:,it,:]) for it in range(1, size(A)[2])]
    get_mean(A) = transpose(nanmean(A, dims=1))
    get_std(A) = transpose(nanstd(A, dims=1))
    get_err_of_mean(A) = get_std(A)./sqrt.(get_detectable_N_ions(A))
    get_RMSD(A) = transpose(sqrt.(nanmean(A.^2, dims=1)))
    get_E_par(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,3])^2/charge_hists[ion_id,it] + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)
    get_E_tot(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,:])^2/charge_hists[ion_id,it]  + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)
    
    if mod(n_smooth_E,2) == 1
        n_smooth_half = Int64(floor(n_smooth_E/2)) # half-length of sliding smoothing window
    else
        throw("The sample number `n_smooth_E` must be an odd integer.")
    end
    
    # Determine ion species to use for fetching ramp correction data
    mean_m0_u = sum(run_info.m0_u .* run_info.m0_probs)/sum(run_info.m0_probs)
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
        if run_info.n_b > 0 # fetch plasma-on data
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
    V_thres = V_nest_eff - 5*k_B.val*run_info.T_b/abs(run_info.q_b) 
                    
    it_eval = [argmin(abs.(sample_times .- t)) for t in times_exp]
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
    mean_E_par = get_mean(E_par) 
    std_E_par = get_std(E_par) 
    detectable_N_ions = get_detectable_N_ions(detectable_E_par) 
    
                        
    E_par_loss = sum( ((mean_E_par .- mean_E_par_exp)./mean_E_par_exp).^2 )
    std_E_par_loss = sum( ((std_E_par .- std_E_par_exp)./std_E_par_exp).^2 )
    N_ions_loss = sum( ((detectable_N_ions .- N_ions_exp)./N_ions_exp).^2 )
    total_loss = sum( E_par_loss + std_E_par_loss + 0.1*N_ions_loss )
                        
    return total_loss #E_par_loss, std_E_par_loss, N_ions_loss
end

                        
exp_data_fname_Na = "RFA_results_run04343_Na23_final.npy"
exp_data_fname_K = "RFA_results_run04354_K39_final.npy"
exp_data_fname_Rb = "RFA_results_run04355_Rb85_final.npy"

using HDF5
include("../Diagnostics.jl")
rel_path="/OutputFiles/"
fname = "2023-06-28_1636_test_run_Na_ions_plasma_off.h5"

run_info = get_run_info(fname; rel_path="/Tests/" * rel_path)

path = string(@__DIR__) * rel_path * fname 
fid = h5open(path, "r")
orbs = fid["IonOrbits"] 
sample_times = read(orbs["sample_time_hists"])[1,:]
position_hists = read(orbs["position_hists"])
velocity_hists = read(orbs["velocity_hists"])
charge_hists = read(orbs["charge_hists"])
mass_hists = read(orbs["mass_hists"])
                        
@time loss(sample_times, position_hists, velocity_hists, 
           charge_hists, mass_hists, r_b=0.0, max_detectable_r=8e-04, 
           n_smooth_E=51, exp_data_fname=exp_data_fname_Na)

using Dates
using Distributed
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
include("../IonNeutralCollisions.jl")

# Define global parameters
const B = [0.,0.,7.]

# Define ion initial conditions
q = e.val
m0_u = [23]
m0_probs = [1.]
const N_ions = 16
const μ_z0 = -0.125
const σ_z0 = 0.003
const σ_xy0 = 0.00025
const μ_E0_par, σ_E0_par = 83., 8.
const σ_E0_perp = 0.5

# Define plasma parameters
n_b = 0.0 #1e07*1e06
T_b = 300.
const q_b = -e.val 
const m_b = m_e.val
const r_b = 0.001 # plasma radius for grabbing potential data

# Define residual gas parameters
neutral_masses = [2*m_u.val, 18*m_u.val, 28*m_u.val, 44*m_u.val] # H2, H20, C0, CO2 
neutral_pressures_mbar = [0.80, 0.10, 0.05, 0.05]*3.15e-09 #[0.70, 0.10, 0.10, 0.10]*3.7e-09 #[0.38, 0.20, 0.32, 0.10]*2.45e-09 #[5e-10, 5e-10, 4e-10, 4e-10]
alphas = [alpha_H2, alpha_H2O, alpha_CO, alpha_CO2]
CX_fractions = [0., 0., 0., 0.] 
T_n = 300. 

# Define run parameters
n_procs = 16
t_end = 3700e-03
dt = 3e-08 # TODO: Reduce again!
sample_every = 5000
seed = 85383
velocity_diffusion = true
now = Dates.now()
datetime = Dates.format(now, "yyyy-mm-dd_HHMM_")
output_path = "Tests/OutputFiles/" * datetime * "test_run_plasma_off"

##### Run test simulation
addprocs(n_procs)
@everywhere include("../CoolingSimulation.jl")


function eval_plasma_off_orbits(neutral_pressures_mbar;q=e.val, m=23*m_u.val, exp_data_fname=exp_data_fname_Na, 
                                max_detectable_r=8e-04, n_smooth_E=51)
    orbits =  integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                                   μ_z0=μ_z0, σ_z0=σ_z0, σ_xy0=σ_xy0,  q0=q, m0_u=m0_u, m0_probs=m0_probs, N_ions=N_ions, B=B, 
                                   n_b=0.0, T_b=T_b, q_b=q_b, m_b=m_b, r_b=0.0,
                                   neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                   alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                   t_end=t_end, dt=dt, sample_every=sample_every, seed=seed,
                                   velocity_diffusion=velocity_diffusion, fname=nothing)
      
    sample_times = orbits[1][1,:]
    position_hists = orbits[2]
    velocity_hists = orbits[3]
    charge_hists = orbits[4]
    mass_hists = orbits[5]
    
    out = loss(sample_times, position_hists, velocity_hists, 
                charge_hists, mass_hists, n_smooth_E=n_smooth_E, 
                exp_data_fname=exp_data_fname)

    return out
end


@time eval_plasma_off_orbits(neutral_pressures_mbar; q=e.val, m=23*m_u.val, exp_data_fname=exp_data_fname_Na, 
                             max_detectable_r=8e-04, n_smooth_E=51)


# ###### BlackBoxOptim 
# ### multiobjective.jl
# using BlackBoxOptim, Gadfly
# using LinearAlgebra

# # run Borg MOAE
# guess = neutral_pressures_mbar
# res = bboptimize(eval_plasma_off_orbits, guess; Method=:borg_moea,
#                  FitnessScheme=ParetoFitnessScheme{3}(is_minimizing=true),
#                  SearchRange=(1e-10, 1e-07), NumDimensions=length(guess), ϵ=0.1,
#                  MaxSteps=15, TraceInterval=1.0, TraceMode=:verbose);

##### Surrogates.jl
### Optimization example
using Surrogates
using Plots
using AbstractGPs
using SurrogatesAbstractGPs

n_samples = 30
lower_bounds = [1e-10,1e-10,1e-10,5e-11]
upper_bounds = [1e-08, 5e-09, 5e-09, 1e-09]
#xs = minimum(lower_bounds):5e-10:maximum(upper_bounds)
x = Surrogates.sample(n_samples, lower_bounds, upper_bounds, SobolSample())
    
y = eval_plasma_off_orbits.(x)
gp_surrogate = AbstractGPSurrogate(x,y)

# Plot samples from surrogate GaussianProcess
f = scatter(getindex.(gp_surrogate.x,1), gp_surrogate.y, label="Sampled points", 
            xlabel="H2 pressure (mbar)", ylabel="Loss")
#plot!(xs[1], gp_surrogate.(xs), label="Surrogate function", ribbon=p->std_error_at_point(gp_surrogate, p), legend=:top)
savefig(f, "SurrogateGC_samples_H2.png") 
#display(f)

f = scatter(getindex.(gp_surrogate.x,2), gp_surrogate.y, label="Sampled points", 
            xlabel="H2O pressure (mbar)", ylabel="Loss")
savefig(f, "SurrogateGC_samples_H2O.png") 
#display(f)

f = scatter(getindex.(gp_surrogate.x,3), gp_surrogate.y, label="Sampled points", 
            xlabel="N2 pressure (mbar)", ylabel="Loss")
savefig(f, "SurrogateGC_samples_N2.png") 
#display(f)

f = scatter(getindex.(gp_surrogate.x,4), gp_surrogate.y, label="Sampled points", 
            xlabel="CO2 pressure (mbar)", ylabel="Loss")
savefig(f, "SurrogateGC_samples_CO2.png") 
#display(f)

@show surrogate_optimize(eval_plasma_off_orbits, SRBF(), lower_bounds, upper_bounds, gp_surrogate, SobolSample())