##### Helper functions for parameter optimization with Surrogates.jl
using LinearAlgebra
using Dates
using Distributed
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
include("IonNeutralCollisions.jl")
include("CoolingSimulation.jl")
include("Diagnostics.jl")

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

"""Calculate loss function for a given ion species"""
function single_species_loss(sample_times, position_hists, velocity_hists, 
                             charge_hists, mass_hists; q_b=-e.val, r_b=0.0, T_b=300, max_detectable_r=8e-04, 
                             n_smooth_E=51, ramp_correction=true, exp_data_fname=nothing)
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

# Define mock loss function for sped-up testing
x0 =  [3.2e-09, 1e-09, 8e-10, 2e-10, 0.00020] # [3.2e-09, 0.00020] # True parameter values
function g(x)
    scale = [1e-09, 1e-09, 1e-09, 1e-09, 1e-04] # [1e-09, 1e-04]
    return sum( ((x .- x0) ./ scale).^2 )
end

orbit_tracing_kws = Dict(:μ_E0_par => 84., :σ_E0_par => 13., :σ_E0_perp => 0.5, 
                         :μ_z0 => -0.125, :σ_z0 => 0.003, :q0 => e.val, :m0_u => [23.], :m0_probs => [1.], 
                         :N_ions => 50, :B => [0.,0.,7.], :T_b => 300, :q_b => -e.val, :m_b => m_e.val, :r_b => 0.0,
                         :neutral_masses => [2*m_u.val], :neutral_pressures_mbar => [3e-09], 
                         :alphas => [alpha_H2], :CX_fractions => [0.], :T_n => 300, :seed => 85383, 
                         :t_end => 3.7, :dt => 1e-08, :sample_every => 200, :velocity_diffusion => true, :n_workers => 1)

##### Run test simulation
function eval_single_species_loss(x; orbit_tracing_kws::Dict, exp_data_fname=nothing, 
                                  max_detectable_r=8e-04, ramp_correction=true, n_smooth_E=51)    
    orbit_tracing_kws[:neutral_pressures_mbar] = x[1:4]
    orbit_tracing_kws[:σ_xy0] = x[5] 

    orbits = integrate_ion_orbits(;orbit_tracing_kws...)
      
    sample_times = orbits[1][1,:]
    position_hists = orbits[2]
    velocity_hists = orbits[3]
    charge_hists = orbits[4]
    mass_hists = orbits[5]
    
    loss_val = single_species_loss(sample_times, position_hists, velocity_hists, 
                                   charge_hists, mass_hists, q_b=orbit_tracing_kws[:q_b], 
                                   r_b=orbit_tracing_kws[:r_b], T_b=orbit_tracing_kws[:T_b],
                                   max_detectable_r=max_detectable_r, ramp_correction=ramp_correction,
                                   n_smooth_E=n_smooth_E, exp_data_fname=exp_data_fname)
    
    # Clean up shared arrays 
    finalize(sample_times)
    finalize(position_hists)
    finalize(velocity_hists)
    finalize(charge_hists)
    finalize(mass_hists)
    @everywhere GC.gc() # To prevent memory leakage and overfilling of /dev/shm

    return loss_val
end

"""Evalute combined loss for Na, K and Rb ions"""
function eval_combined_plasma_off_loss(x; orbit_tracing_kws::Dict, max_detectable_r=8e-04, n_smooth_E=51, seed=85383, scale_time_step=false)
    tracing_kws = deepcopy(orbit_tracing_kws)
    tracing_kws[:r_b] = 0.0 # turn plasma off 
    tracing_kws[:seed] = seed # TODO: Consider using different seed for each species

    tracing_kws[:m0_u] = alkali_mass_data["Na"][1]
    tracing_kws[:m0_probs] = alkali_mass_data["Na"][2]
    loss_val_Na = eval_single_species_loss(x; orbit_tracing_kws=tracing_kws, exp_data_fname=exp_data_fname_Na, 
                                           max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E)
    
    tracing_kws[:m0_u] = alkali_mass_data["K"][1]
    tracing_kws[:m0_probs] = alkali_mass_data["K"][2]
    if scale_time_step
        dt_Na = deepcopy(orbit_tracing_kws[:dt])
        tracing_kws[:dt] = round(dt_Na*sqrt(mean(alkali_mass_data["K"][1].*alkali_mass_data["K"][2])/alkali_mass_data["Na"][1][1]), 
                                  digits=Int(floor(abs(log10(dt_Na))))+1)
        println(tracing_kws[:dt])
    end
    loss_val_K = eval_single_species_loss(x; orbit_tracing_kws=tracing_kws, exp_data_fname=exp_data_fname_K, 
                                          max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E)
    
    tracing_kws[:m0_u] = alkali_mass_data["Rb"][1]
    tracing_kws[:m0_probs] = alkali_mass_data["Rb"][2]
    if scale_time_step
        tracing_kws[:dt] = round(dt_Na*sqrt(mean(alkali_mass_data["Rb"][1].*alkali_mass_data["Rb"][2])/alkali_mass_data["Na"][1][1]), 
                                 digits=Int(floor(abs(log10(dt_Na))))+1)
        println(tracing_kws[:dt])
    end
    loss_val_Rb = eval_single_species_loss(x; orbit_tracing_kws=tracing_kws, exp_data_fname=exp_data_fname_Rb, 
                                           max_detectable_r=max_detectable_r, n_smooth_E=n_smooth_E)

    println()
    println("Input parameters: ",x)
    println("Loss [Na / K / Rb]: ",[loss_val_Na, loss_val_K, loss_val_Rb])
    return sum([loss_val_Na, loss_val_K, loss_val_Rb])
end

"""Elementary effects of parameters on loss function"""
function elementary_effect(loss_func, x0, lower_bounds, upper_bounds; delta=1/100)
    y0 = loss_func(x0)
    function shift_element(x, index, delta)
        x_new = deepcopy(x)
        x_new[index] += delta*(upper_bounds[index] - lower_bounds[index])
        return x_new
    end
    shifted_x = [shift_element(x0, i, delta) for i in range(1,length(x0))]
    d = (eval_combined_plasma_off_loss.(shifted_x) .- y0) / delta
    return d
end 
