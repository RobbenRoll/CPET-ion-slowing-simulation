#!/usr/bin/julia
using Plots
using HDF5
using Printf
using LaTeXStrings
using LinearAlgebra
using NaNStatistics
using PyCall
import Base.@kwdef
import PhysicalConstants.CODATA2018: m_u
default(fmt = :png) # prevent slowdown from dense figures
include("ParticlePushers.jl")

##### Define particle trajectory diagnostics
function plot_trajectory_data(times, positions, velocities; q=e.val, m=m_u.val, r_b=0.011, n_smooth_E=1)
    plot(times, 10*getindex.(positions,1), xlabel="t (s)", ylabel="x, y, z (m)", label="x*10")
    plot!(times, 10*getindex.(positions,2), label="y*10")
    display(plot!(times, getindex.(positions,3), label="z"))

    display(plot(getindex.(positions,1), getindex.(positions,2), xlabel="x", ylabel="y", label=""))

    display(plot(getindex.(positions,1), getindex.(positions,3), xlabel="x", ylabel="z", label=""))

    display(plot(times, getindex.(velocities,2), xlabel="t", ylabel="vy", label=""))

    display(plot(times, getindex.(velocities,3), xlabel="t", ylabel="vz", label=""))
    
    V_sitp = get_V_sitp(r_b)
    E_kins = [0.5*m*norm(vel)^2 for vel in velocities]/q
    E_pots = V_itp.(positions, V_sitp=V_sitp)
    E_tots = E_kins + E_pots
    n_smooth_half = Int64(floor(n_smooth_E/2))
    
    smoothed_E_kins = [mean(E_kins[i-n_smooth_half:i+n_smooth_half]) for i in range(1+n_smooth_half,step=1,stop=length(E_tots)-n_smooth_half)]
    display(plot(times[1+n_smooth_half:end-n_smooth_half], smoothed_E_kins, xlabel="t (s)", 
            ylabel="Kinetic ion energy (eV/q)"))
    
    smoothed_E_tots = [mean(E_tots[i-n_smooth_half:i+n_smooth_half]) for i in range(1+n_smooth_half,step=1,stop=length(E_tots)-n_smooth_half)]
    display(plot(times[1+n_smooth_half:end-n_smooth_half], smoothed_E_tots, xlabel="t (s)", 
            ylabel="Total ion energy (eV/q)"))
end

function plot_adiabatic_invariants(times, positions, velocities; m=m_u.val, B=1.0)
    v_p = [norm(v[1:2]) for v in velocities] # velocity of reduced cyclotron motion 
    μ = 0.5*m*v_p.^2/norm(B)
    f1 = plot(times, μ/μ_B.val, xlabel="t (s)", ylabel="μ (μ_B)")
    display(f1)

    vz_half_osc = [] # velocities along magnetic field line for one axial half oscillation
    ds_half_osc = [] # path lengths along magnetic field line for one axial half oscillation
    reflection_times = []
    J = []
    refl_count = 0
    for i in range(2,length(velocities))
        if sign(velocities[i][3])*sign(velocities[i-1][3]) < 0 # reflection at axial turning point 
            refl_count += 1
            if refl_count > 1
                push!(reflection_times, times[i])
                push!(J, sum(vz_half_osc.*ds_half_osc))
                vz_half_osc = [] # reset
                ds_half_osc = [] # reset
            end
        end
        if refl_count == 0
            continue # discard steps up to first reflection
        end
        ds = positions[i][3] - positions[i-1][3]
        push!(vz_half_osc, velocities[i][3])
        push!(ds_half_osc, ds)
    end
    f2 = plot(reflection_times, J, xlabel="t (s)", ylabel="J (m^2/s)")
    display(f2) 
end

@kwdef struct RunInfo
    filepath::String
    datetime::String 
    t_end::Float64
    dt::Float64 
    sample_every::Int64 
    N_ions::Int64
    μ_E0_par::Float64
    σ_E0_par::Float64
    σ_E0_perp::Float64
    μ_z0::Float64
    σ_z0::Float64
    σ_xy0::Float64
    q0::Float64
    m0_u::Vector{Float64}
    m0_probs::Vector{Float64}
    B::Vector{Float64}
    n_b::Float64
    T_b::Float64
    q_b::Float64
    m_b::Float64
    r_b::Float64
    neutral_masses::Vector{Float64}
    neutral_pressures_mbar::Vector{Float64}
    alphas::Vector{Float64}
    CX_fractions::Vector{Float64}
    T_n::Float64
    coll_types::Vector{String}
    velocity_diffusion::Bool
    seed::Any
end 

##### Load and plot HDF5 data
function get_run_info(fname; rel_path="/Tests/OutputFiles/")
    """Plot diagnostics for ion orbit data stored in HDF5 file"""
    path = string(@__DIR__) * rel_path * fname 
    fid = h5open(path, "r")
    info = fid["RunInfo"]
    run_info = RunInfo(
                        filepath = read(info["filepath"]),
                        datetime = read(info["datetime"]), 
                        t_end = read(info["t_end"]),
                        dt = read(info["dt"]),
                        sample_every = read(info["sample_every"]),
                        N_ions= read(info["N_ions"]),
                        μ_E0_par = read(info["μ_E0_par"]),
                        σ_E0_par = read(info["σ_E0_par"]),
                        σ_E0_perp = read(info["σ_E0_perp"]),
                        μ_z0 = read(info["μ_z0"]),
                        σ_z0 = read(info["σ_z0"]),
                        σ_xy0 = read(info["σ_xy0"]),
                        q0 = read(info["q0"]),
                        m0_u = read(info["m0_u"]),
                        m0_probs = read(info["m0_probs"]),
                        B = read(info["B"]),
                        n_b = read(info["n_b"]),
                        T_b = read(info["T_b"]),
                        q_b = read(info["q_b"]),
                        m_b = read(info["m_b"]),
                        r_b = read(info["r_b"]),
                        neutral_masses = read(info["neutral_masses"]),
                        neutral_pressures_mbar = read(info["neutral_pressures_mbar"]),
                        alphas = read(info["alphas"]),
                        CX_fractions = read(info["CX_fractions"]),
                        T_n = read(info["T_n"]),
                        coll_types = read(info["coll_types"]),
                        velocity_diffusion = read(info["velocity_diffusion"]),
                        seed = read(info["seed"]),
                      )
    close(fid)

    return run_info
end

function print_RunInfo(run_info)
    """Print formatted representation of RunInfo"""
    for name in fieldnames(typeof(run_info))
        val = getfield(run_info, name)
        @printf("%25s:  %s \n", name, val)
    end
end

function get_ion_orbits(fname; rel_path="/Tests/OutputFiles/")
    """Extract ion orbit data from HDF5 file"""
    run_info = get_run_info(fname; rel_path=rel_path)

    path = string(@__DIR__) * rel_path * fname 
    fid = h5open(path, "r")
    orbs = fid["IonOrbits"] 
    sample_times = read(orbs["sample_time_hists"])[1,:]
    position_hists = read(orbs["position_hists"])
    velocity_hists = read(orbs["velocity_hists"])
    close(fid)

    return sample_times, position_hists, velocity_hists
end

function get_exp_data(exp_data_fname, rel_path="/ExpDatasets/")
    """Load experimental ion energy data from .npy file"""
    np = pyimport("numpy")
    exp_data = np.load(string(@__DIR__) * rel_path * exp_data_fname, allow_pickle=true)[1]
    return exp_data
end 

function get_default_exp_data_fname(m_u; atol=0.49, verbose=true)
    """Get filename for default experimental data from ion mass"""
    if isapprox(m_u, 23, atol=atol)
        exp_data_fname = "RFA_results_run04343_Na23_final.npy"
    elseif isapprox(m_u, 39.1, atol=atol)
        exp_data_fname = "RFA_results_run04354_K39_final.npy"
    elseif isapprox(m_u, 85.3, atol=atol)
        exp_data_fname = "RFA_results_run04355_Rb85_final.npy"
    end
    if verbose 
        println("Fetching exp. data from " * exp_data_fname)
    end
    
    return exp_data_fname
end

function nanmask(A, m)
    B = copy(A)
    B[m] .= NaN
    return B 
 end

 function apply_ramp_correction(energies, V_nest_eff; species="23Na")
    """Correct ion energies for adiabatic cooling correction"""
    path = string(@__DIR__) * "/VoltageRampCorrection/"
    corr_energies = []
    initial_energies = []
    eff_nest_depths = Vector(28:2:40)
    for nest_depth in eff_nest_depths
        fname = "phase-averaged_ion_energies_after_extraction_"*species*"_"*string(nest_depth)*"V_eff_nest_depth_0V_RFA_barrier.csv"
        csv_file = CSV.File(path * fname)
        df = DataFrame(csv_file)
        rename!(df,[:E_initial,:E_final,:std_E_final,:extraction_prob])
        
        # Add in missing data at even V_nest_eff -0.01 / +0.025 eV
        for V in eff_nest_depths
            for E_init in [V - 0.01, V + 0.025]
                if !(E_init in df.E_initial)
                    E_below = V - mod(V, 5)
                    E_above = E_below + 5
                    if E_below in df.E_initial && E_above in df.E_initial 
                        interp_row = Matrix(df[(df.E_initial.==E_below),:]) + (Matrix(df[(df.E_initial.==E_above),:]) - Matrix(df[(df.E_initial.==E_below),:]))*mod(V,5)/(E_above - E_below)
                    elseif !(E_below in df.E_initial)
                        E_below = V - mod(V, 5) + 0.025
                        E_above = E_below + 4.975
                        interp_row = Matrix(df[(df.E_initial.==E_below),:]) + (Matrix(df[(df.E_initial.==E_above),:]) - Matrix(df[(df.E_initial.==E_below),:]))*(mod(V,5) - 0.025)/(E_above - E_below)
                    elseif !(E_above in df.E_initial)
                        E_below = V - mod(V, 5)
                        E_above = round(E_below + 4.99, digits=3)
                        interp_row = Matrix(df[(df.E_initial.==E_below),:]) + (Matrix(df[(df.E_initial.==E_above),:]) - Matrix(df[(df.E_initial.==E_below),:]))*(mod(V,5) + 0.010)/(E_above - E_below)
                    end 
                    interp_row[1] = E_init
                    push!(df, interp_row)
                end
            end
        end
        # Add interpolations for missing rows at 30V and 40V 
        if !(30. in df.E_initial)
            E_below = 29.99
            E_above = 30.025
            interp_row = Matrix(df[(df.E_initial.==E_below),:]) + (Matrix(df[(df.E_initial.==E_above),:]) - Matrix(df[(df.E_initial.==E_below),:]))*0.010/(E_above - E_below)
            push!(df, interp_row)
        elseif !(40. in df.E_initial)
            E_below = 39.99
            E_above = 40.025
            interp_row = Matrix(df[(df.E_initial.==E_below),:]) + (Matrix(df[(df.E_initial.==E_above),:]) - Matrix(df[(df.E_initial.==E_below),:]))*0.010/(E_above - E_below)
            push!(df, interp_row)
        end
        sort!(df, :E_initial)
        push!(corr_energies, df.E_final)
        initial_energies = df.E_initial
    end
    corr_energies = stack(corr_energies)
    itp = interpolate( (initial_energies, eff_nest_depths), corr_energies, Gridded(Linear()) ) 
    exitp = extrapolate(itp, Line())
    return exitp(energies, V_nest_eff)    
end

function print_collision_stats(fname; rel_path="/Tests/OutputFiles/")
    run_info = get_run_info(fname; rel_path=rel_path)

    path = string(@__DIR__) * rel_path * fname 
    fid = h5open(path, "r")
    orbs = fid["IonOrbits"] 
    coll_counts = read(orbs["coll_counts"])
    println("\n### Ion-neutral collision counts for all ions ###")
    for target_id in range(1,length(run_info.neutral_masses))
        println("\nNeutral target mass: ", run_info.neutral_masses[target_id]/m_u.val," u")
        for type_id in range(1,length(run_info.coll_types))
            coll_type = run_info.coll_types[type_id] 
            println(coll_type, " collisions: ", sum(coll_counts[:,target_id,type_id]))
        end 
    end
end

function plot_run_results(fname; rel_path="/Tests/OutputFiles/", max_detectable_r=8e-04, 
                          ramp_correction=true, exp_data_fname="default", n_smooth_E=1)
    """Plot diagnostics for ion orbit data stored in HDF5 file"""
    run_info = get_run_info(fname; rel_path=rel_path)

    if mod(n_smooth_E,2) == 1
        n_smooth_half = Int64(floor(n_smooth_E/2)) # half-length of sliding smoothing window
    else
        throw("The sample number `n_smooth_E` must be an odd integer.")
    end
    path = string(@__DIR__) * rel_path * fname 
    fid = h5open(path, "r")
    orbs = fid["IonOrbits"] 
    sample_times = read(orbs["sample_time_hists"])[1,:]
    smoothed_sample_times = @views sample_times[1+n_smooth_half:end-n_smooth_half]
    position_hists = read(orbs["position_hists"])
    velocity_hists = read(orbs["velocity_hists"])
    charge_hists = read(orbs["charge_hists"])
    mass_hists = read(orbs["mass_hists"])
    N_ions = run_info.N_ions
    q = run_info.q0
    mean_m0_u = sum(run_info.m0_u .* run_info.m0_probs)/sum(run_info.m0_probs)

    # Determine ion species to use for fetching ramp correction data
    if isapprox(mean_m0_u, 23, atol=0.49)
        species = "23Na"
    elseif isapprox(mean_m0_u, 39.1, atol=0.49)
        species = "39K"
    elseif isapprox(mean_m0_u, 85.3, atol=0.49)
        species = "85Rb"
    else 
        species = ""
    end
    V_sitp = get_V_sitp(run_info.r_b) # load potential map

    # Fetch and prepare experimental data from .npy file
    if exp_data_fname == "default" 
        exp_data_fname = get_default_exp_data_fname(mean_m0_u, atol=0.49, verbose=true)
    end
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

    # Define convenience functions for grabbing data from position and velocity histories
    get_detectable_N_ions(A) = [countnotnans(A[:,it,:]) for it in range(1, size(A)[2])]
    get_mean(A) = transpose(nanmean(A, dims=1))
    get_std(A) = transpose(nanstd(A, dims=1))
    get_err_of_mean(A) = get_std(A)./sqrt.(get_detectable_N_ions(A))
    get_RMSD(A) = transpose(sqrt.(nanmean(A.^2, dims=1)))
    get_E_par(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,3])^2/charge_hists[ion_id,it] + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)
    get_E_tot(ion_id, it) = 0.5*mass_hists[ion_id,it]*norm(@views velocity_hists[ion_id,it,:])^2/charge_hists[ion_id,it]  + @views V_itp(position_hists[ion_id,it,:], V_sitp=V_sitp)

    # Determine effective nest depth and set threshold energy for determining 
    # ions localized in entrance-side potential well
    V_nest_eff = mean(V_itp_on_axis(Vector(0:0.001:0.025), V_sitp))
    V_thres = V_nest_eff - 5*k_B.val*run_info.T_b/abs(run_info.q_b) 

    # Collect longitudinal energy data 
    E_par = []
    for i in range(1,N_ions)
        E_par_i = [mean(get_E_par.(i, it-n_smooth_half:it+n_smooth_half)) for it in range(1+n_smooth_half,length(sample_times)-n_smooth_half)]  
        if ramp_correction # correct ion energies for endcap voltage ramp
            E_par_i = apply_ramp_correction(E_par_i, V_nest_eff, species=species)
        end
        push!(E_par, E_par_i)
    end 
    E_par = transpose(stack(E_par)) # idx1: ion number, idx2: time step
    z = [mean(position_hists[i,it-n_smooth_half:it+n_smooth_half,3]) for it in range(1+n_smooth_half,length(sample_times)-n_smooth_half), i in range(1,N_ions)]
    r = [mean(sqrt.(position_hists[i,it-n_smooth_half:it+n_smooth_half,1].^2 .+ position_hists[i,it-n_smooth_half:it+n_smooth_half,2].^2)) for it in range(1+n_smooth_half,length(sample_times)-n_smooth_half), i in range(1,N_ions)]
    z = transpose(z)
    r = transpose(r)
    detectable = ((E_par .> V_thres .|| z .> 0.0) .&& r .<= max_detectable_r) # bool-mask for detectable ions
    detectable_E_par = nanmask(E_par, @. !detectable)
    mean_E_par = get_mean(E_par) 
    std_E_par = get_std(E_par) 
    err_mean_E_par = get_err_of_mean(E_par) 
    
    detectable_N_ions = get_detectable_N_ions(detectable_E_par) 
    detectable_mean_E_par = get_mean(detectable_E_par) 
    detectable_std_E_par = get_std(detectable_E_par) 
    detectable_err_mean_E_par = get_err_of_mean(detectable_E_par) 
    
    # Detectable ion fraction
    function plot_detectable_ion_number(smoothed_sample_times, detectable_N_ions, exp_data_fname)
        f = plot(smoothed_sample_times, detectable_N_ions/N_ions, xlabel="Time (s)", 
                 ylabel="Detectable ion fraction", label="model")
        if !isnothing(exp_data_fname)
            scatter!(times_exp, N_ions_exp/maximum(N_ions_exp), yerror=err_N_ions_exp/maximum(N_ions_exp), 
                     markershape=:circle, markercolor=:black, label="exp. data")
        end 
        display(f)
    end
    plot_detectable_ion_number(smoothed_sample_times, detectable_N_ions, exp_data_fname)

    # Longitudinal energy evolutions
    f = plot(xlabel="Time (s)", ylabel=L"Longitudinal ion energy $E_\parallel$ (eV/q)", 
             legend=false)
    plot!(smoothed_sample_times, [E_par[i,:] for i in range(1,N_ions)], linestyle=:dash)
    plot!(smoothed_sample_times, [detectable_E_par[i,:] for i in range(1,N_ions)]) # TODO use same colors as for dashed lines
    plot!(smoothed_sample_times, detectable_mean_E_par, linewidth=3, linecolor="red")
    plot!(smoothed_sample_times, mean_E_par, linewidth=3, linecolor="black")
    if !isnothing(exp_data_fname)
        scatter!(times_exp, mean_E_par_exp , yerror=err_mean_E_par_exp, 
                 markershape=:circle, markercolor=:black, label="exp. data")
    end 
    display(f)

    # Mean longitudinal energy evolution
    f = plot(xlabel="Time (s)", ylabel=L"Mean longitudinal ion energy $\langle E_\parallel\rangle$ (eV/q)", 
             legend=true)
    plot!(smoothed_sample_times, detectable_mean_E_par, 
          ribbon=detectable_err_mean_E_par, linewidth=1, linecolor="red", label="Detectable ions")
    plot!(smoothed_sample_times, mean_E_par, 
          ribbon=err_mean_E_par, linewidth=1, linecolor="black", label="All ions")
    if !isnothing(exp_data_fname)
        scatter!(times_exp, mean_E_par_exp , yerror=err_mean_E_par_exp, 
                 markershape=:circle, markercolor=:black, label="exp. data")
    end 
    display(f)

    # Sample standard deviation of longitudinal energy evolutions
    f = plot(xlabel="Time (s)", ylabel=L"Std. dev. of  $E_\parallel$ (eV/q)", legend=true)
    plot!(smoothed_sample_times, detectable_std_E_par, linecolor="red", label="Detectable ions")
    plot!(smoothed_sample_times, std_E_par, linecolor="black", label="All ions")
    
    if !isnothing(exp_data_fname)
        scatter!(times_exp, std_E_par_exp, label="exp. data",
                 markershape=:circle, markercolor=:black)
    end 
    display(f)

    # Total energy evolutions
    f = plot(xlabel="Time (s)", ylabel="Total ion energy (eV/q)",  legend=false)
    E_tot = []
    for i in range(1,N_ions)
        E_tot_i = [mean(get_E_tot.(i,it-n_smooth_half:it+n_smooth_half)) for it in range(1+n_smooth_half,length(sample_times)-n_smooth_half)]  
        if ramp_correction # correct ion energies for endcap voltage ramp
            E_tot_i = apply_ramp_correction(E_tot_i, V_nest_eff, species=species)
        end
        plot!(smoothed_sample_times, E_tot_i)
        push!(E_tot, E_tot_i)
    end 
    E_tot = transpose(stack(E_tot))
    mean_E_tot = get_mean(E_tot) 
    std_E_tot = get_std(E_tot)
    err_mean_E_tot = get_err_of_mean(E_tot) 
    detectable_E_tot = nanmask(E_tot, @. !detectable)
    detectable_mean_E_tot = get_mean(detectable_E_tot)
    detectable_std_E_tot = get_std(detectable_E_tot)
    detectable_err_mean_E_tot = get_err_of_mean(detectable_E_tot)
    plot!(smoothed_sample_times, detectable_mean_E_tot, linewidth=3, linecolor="red")
    plot!(smoothed_sample_times, mean_E_tot, linewidth=3, linecolor="black")
    display(f)

    # Mean total energy evolution
    f = plot(xlabel="Time (s)", ylabel="Mean ion energy (eV/q)", legend=true)
    plot!(smoothed_sample_times, detectable_mean_E_tot, 
          ribbon=detectable_err_mean_E_tot, linewidth=1, linecolor="red", label="Detectable ions")
    plot!(smoothed_sample_times, mean_E_tot, 
          ribbon=err_mean_E_tot, linewidth=1, linecolor="black", label="All ions")
    display(f)

    # Sample standard deviation of total energy evolutions
    f = plot(xlabel="Time (s)", ylabel=L"Std. dev. of  $E$ (eV/q)", legend=true)
    plot!(smoothed_sample_times, detectable_std_E_tot, linecolor="red", label="Detectable ions")
    plot!(smoothed_sample_times, std_E_tot, linecolor="black", label="All ions")
    display(f)

    # 3D scatter of final ion positions
    it = length(sample_times) - n_smooth_half
    X = @views position_hists[:,it,1] 
    Y = @views position_hists[:,it,2] 
    Z = @views position_hists[:,it,3]  
    detectable_X = @views nanmask(position_hists[:,1+n_smooth_half:end-n_smooth_half,1] , @. !detectable)[:,it - n_smooth_half] 
    detectable_Y = @views nanmask(position_hists[:,1+n_smooth_half:end-n_smooth_half,2] , @. !detectable)[:,it - n_smooth_half] 
    detectable_Z = @views nanmask(position_hists[:,1+n_smooth_half:end-n_smooth_half,3] , @. !detectable)[:,it - n_smooth_half]                    

    f = scatter(ylabel="x (m)", zlabel="y (m)", legend=true, title="Final ion distribution",)
    scatter!(Z, X, Y, xlabel="z (m)", label="All ions") 
    scatter!(detectable_Z, detectable_X, detectable_Y, markercolor="red", label="Detectable ions")
    display(f)

    # ZR scatter of final ion positions
    f = scatter(xlabel="z (m)", ylabel="r (m)", xlim=(-0.15,0.15), legend=true, title="Final ion distribution")
    scatter!(Z, sqrt.(X.^2 + Y.^2), label="All ions")
    scatter!(detectable_Z, sqrt.(detectable_X.^2 + detectable_Y.^2), markercolor="red", label="Detectable ions")
    display(f)

    # Collect axial and radial displacement metrics
    radial_offsets = []
    axial_offsets = []
    for i in range(1, N_ions)
        R = sqrt.(@views position_hists[i,:,1].^2 .+ @views position_hists[i,:,2].^2) 
        Z = @views position_hists[i,:,3] 
        push!(radial_offsets, R)
        push!(axial_offsets, Z)
    end
    radial_offsets = transpose(stack(radial_offsets))
    axial_offsets = transpose(stack(axial_offsets))
    detectable_radial_offsets = nanmask(radial_offsets[:,1+n_smooth_half:end-n_smooth_half], @. !detectable)
    detectable_axial_offsets = nanmask(axial_offsets[:,1+n_smooth_half:end-n_smooth_half], @. !detectable)

    # RMS radial displacements
    RMS_radial_offsets = get_RMSD(radial_offsets) 
    detectable_RMS_radial_offsets = get_RMSD(detectable_radial_offsets) 
    f = plot(legend=true, xlabel="t (s)", ylabel="RMS transverse displacement (m)")
    plot!(smoothed_sample_times, detectable_RMS_radial_offsets, label="Detectable ions only")
    plot!(sample_times, RMS_radial_offsets, label="All ions")
    display(f)

    mean_axial_offsets = get_mean(axial_offsets) 
    err_mean_axial_offsets = get_err_of_mean(axial_offsets) 
    detectable_mean_axial_offsets = get_mean(detectable_axial_offsets)
    detectable_err_mean_axial_offsets = get_err_of_mean(detectable_axial_offsets)

    # Mean axial positions
    f = plot(legend=true, xlabel="t (s)", ylabel="Mean longitudinal position (m)")
    plot!(smoothed_sample_times, detectable_mean_axial_offsets, ribbon=detectable_err_mean_axial_offsets, 
          label="Detectable ions only")
    plot!(sample_times, mean_axial_offsets, ribbon=err_mean_axial_offsets, label="All ions")
    display(f)
    
    # RMS axial positions
    RMS_axial_offsets = get_RMSD(axial_offsets) 
    detectable_RMS_axial_offsets = get_RMSD(detectable_axial_offsets)
    f = plot(legend=true, xlabel="t (s)", ylabel="RMS longitudinal position (m)")
    plot!(smoothed_sample_times, detectable_RMS_axial_offsets, label="Detectable ions only")
    plot!(sample_times, RMS_axial_offsets, label="All ions")
    display(f)

    close(fid)
end                                                                        