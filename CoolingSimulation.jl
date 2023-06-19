#!/usr/bin/julia
using LinearAlgebra
using StaticArrays
using Statistics
using Random
import Random: default_rng
import Distributions: Normal, truncated
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
import SpecialFunctions: erf
import StatsBase: sample, ProbabilityWeights
include("ParticlePushers.jl")
include("IonNeutralCollisions.jl")

##### Define Orbit integration function
function integrate_orbit_with_friction(times, r, u_last_half; q=q, m=m, B=B, 
                                       n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, r_b=0.001,
                                       neutral_masses=[], neutral_pressures_mbar=[], alphas=[], 
                                       CX_fractions=[], T_n=300., dt=1e-09, sample_every=1, 
                                       velocity_diffusion=true, rng=default_rng())
    """Trace orbit of a single ion"""
    if !(length(neutral_masses) == length(neutral_pressures_mbar) == length(alphas) == length(CX_fractions))
        throw("Lengths of `neutral_masses`, `neutral_pressures_mbar`, `alphas` and `CX_fractions` do not match.")
    end
    t_end = times[end]
    N_samples = Int64(round(t_end/(sample_every*dt))) + 1
    sample_times = zeros(Float64, N_samples) #Vector{Float64}([])
    mass_hist = zeros(Float64, N_samples)
    charge_hist = zeros(Float64, N_samples)
    positions = zeros(Float64, N_samples, 3) #Vector{Vector{Float64}}([])
    velocities = zeros(Float64, N_samples, 3) #Vector{Vector{Float64}}([])
    coll_counts =  [Dict{String, Integer}("glanzing" => 0, "Langevin" => 0, "CX" => 0) for _ in neutral_masses]
    E = MVector{3,Float64}(undef) # SizedArray{Tuple{3}}([0.,0.,0.])
    r = @SVector [r[1], r[2], r[3]]
    u_last_half = @SVector [u_last_half[1], u_last_half[2], u_last_half[3]]
    r_next = @SVector [0., 0., 0.]
    u_next_half = @SVector [0., 0., 0.]
    norm_dist = Normal(0, sqrt(dt))
    dW = @MVector zeros(3)
    model_MC_collisions = any(neutral_pressures_mbar .> 0)
    MFPs = Vector{Float64}(similar(neutral_masses))
    v_effs = Vector{Float64}(similar(neutral_masses))
    coll_probs = Vector{Float64}(similar(neutral_masses))

    # Load potential data
    V_sitp = get_V_sitp(r_b)

    function inside_plasma(r, n_b) # TODO: derive plasma bounds from Warp PIC data 
        return Bool(-0.03 < r[3] < 0.05 && n_b > 0.)
    end 

    for t in times 
        # Get E-field and step
        update_E!(E, r, V_sitp) #update_E!(E, r) # E = E_itp(r)
        if inside_plasma(r, n_b) 
            r_next, u_next_half = Boris_push_with_friction(r, u_last_half, E, B, dt, q=q, m=m, dW, norm_dist,
                                                           n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                           velocity_diffusion=velocity_diffusion, rng=rng)
        else
            r_next, u_next_half = Boris_push(r, u_last_half, E, B, dt, q=q, m=m)
        end

        # Sample time-centred particle data
        if mod(t, sample_every*dt - dt^2)  < dt # subtract dt^2 to prevent rounding issues
            i = Int64(round(t/(sample_every*dt))) + 1
            sample_times[i] = t 
            charge_hist[i] = q
            mass_hist[i] = m
            positions[i,:] = r 
            velocities[i,:] = (u_last_half + u_next_half)/2 
        end

        # Model Monte Carlo ion-neutral collision 
        if model_MC_collisions
            update_MFPs!(MFPs, u_next_half, q, m, neutral_masses, alphas, neutral_pressures_mbar, T_n) 
            update_v_effs!(v_effs, u_next_half, neutral_masses, T_n)
            update_coll_probs!(coll_probs, MFPs, v_effs, dt) 
            if rand(rng, Float64) <= sum(coll_probs) 
                i_target = sample(rng, 1:length(neutral_masses), ProbabilityWeights(coll_probs/sum(coll_probs))) # randomly select neutral collision partner
                u_next_half, q, m, coll_type = ion_neutral_collision(u_next_half, q, m, m_n=neutral_masses[i_target], 
                                                                     CX_frac=CX_fractions[i_target],
                                                                     alpha=alphas[i_target], T_n=T_n, rng=rng)
                coll_counts[i_target][coll_type] += 1
            end
        end

        # Update particle data for next time step
        r, u_last_half = r_next, u_next_half 
    end 

    return sample_times, positions, velocities, charge_hist, mass_hist, coll_counts
end


##### Define orbit integrator with adaptive full-orbit/guiding centre particle pusher 
function get_gyroradius(v_perp; q=q, m=m, B=B)
    return norm(v_perp)/abs(get_ω_c(q=q, m=m, B=B))
end

function get_ω_c(;q=e.val, m=m_u.val, B=1.)
    return q/m*norm(B)
end 

function integrate_orbit_with_adaptive_GCA_pusher(t_end, r, u_last_half; q=q, m=m, B=B, 
                                                  n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, 
                                                  neutral_masses=[], neutral_pressures_mbar=[], alphas=[], 
                                                  CX_fractions=[], T_n=300., dt_FO=1e-09, dt_GC=1e-08, saveat=1e-07, 
                                                  velocity_diffusion=true, rng=default_rng())
    """"
    Only applicable for B-field aligned with z-axis 

    s vector defined in opposite direction as in Rode2023

    UNDER DEVELOPMENT

    #TODO: add rng seeding 
    """
    t = 0.0
    dt = dt_FO
    local R, μ, v_par_last_half, gyrophase, v_perp_norm, s 
    R = [NaN, NaN, NaN]

    sample_times = Vector{Float64}([])
    positions = Vector{Vector{Float64}}([])
    velocities = Vector{Vector{Float64}}([])
    push_types = Vector{String}([])
    function inside_plasma(r, n_b)
        if -0.03 < r[3] < 0.05 && n_b > 0. 
            return true
        else 
            return false 
        end 
    end 

    while t < t_end 
        if inside_plasma(r, n_b) # inside plasma #TODO: Add exact description of plasma boundary
            if dt == dt_GC # switch from GCA to FO
                b = B/norm(B)
                v_perp_norm = sqrt(2*μ*norm(B)/m)
                gyroradius = get_gyroradius(v_perp_norm, q=q, m=m, B=B)
                s = -gyroradius*[cos(gyrophase),sin(gyrophase),0]
                r = R + s
                E = E_itp(r)
                v_ExB = cross(E,B)/norm(B)^2
                u_last_half = v_perp_norm*cross(s,B)/norm(cross(s,B)) + v_ExB + v_par_last_half*b
                println(dot(E,s))
            end
            E = E_itp(r)
            r_next, u_next_half = Boris_push_with_friction(r, u_last_half, E, B, dt_FO, q=q, m=m,  
                                                           n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                           velocity_diffusion=velocity_diffusion) 

            # Record data 
            #TODO: Test
            if mod(t, saveat) < dt_FO
                push!(sample_times, t)
                push!(positions, r)
                push!(velocities, (u_last_half + u_next_half)/2) # (u_last_half*dt + u_next_half*dt_FO)/(dt + dt_FO)) 
                push!(push_types, "FO")
            end
            r = r_next 
            u_last_half = u_next_half 
            dt = dt_FO # update for FO-GCA switch checks
        else
            if dt == dt_FO # switch from FO to GCA 
                b = B/norm(B)
                vxB = cross(u_last_half,B)
                E = E_itp(r)
                v_ExB = cross(E,B)/norm(B)^2
                v_par_last_half = dot(u_last_half,b)
                v_perp_norm = norm(u_last_half - v_par_last_half*b - v_ExB)    # sqrt(2*μ*B/m)
                μ = m*v_perp_norm^2/(2*norm(B))
                gyroradius = get_gyroradius(v_perp_norm, q=q, m=m, B=B)
                s = -gyroradius*vxB/norm(vxB)
                R = r - s 
                if t > 0.0 && v_par_last_half > 0 
                    gyrophase = atan(s[2]/s[1]) + pi
                else 
                    gyrophase = atan(s[2]/s[1])
                end
                println(dot(E,s))
            end
            R_next, μ_next, v_par_next_half, gyrophase_next = GC_push_with_gyrophase(R, μ, v_par_last_half, gyrophase, B, dt_GC, q=q, m=m)

            # Record data
            #TODO: Test
            if mod(t, saveat) < dt_GC
                b = B/norm(B)
                E = E_itp(R)
                v_ExB = cross(E,B)/norm(B)^2
                u_last_half = v_perp_norm*cross(s,B)/norm(cross(s,B)) + v_ExB + v_par_last_half*b
                v_perp_norm_next = sqrt(2*μ_next*norm(B)/m)
                E = E_itp(R_next)
                v_ExB_next = cross(E,B)/norm(B)^2
                v_ExB_next_half = (v_ExB + v_ExB_next)/2
                u_next_half = v_perp_norm_next*cross(s,B)/norm(cross(s,B)) + v_ExB_next_half + v_par_next_half*b
                gyroradius = get_gyroradius(v_perp_norm, q=q, m=m, B=B)
                s = -gyroradius*[cos(gyrophase),sin(gyrophase),0]
                r = R + s
                push!(sample_times, t)
                push!(positions, r)
                push!(velocities, (u_last_half + u_next_half)/2) #(u_last_half*dt + u_next_half*dt_GC)/(dt + dt_GC)) 
                push!(push_types, "GCA")
            end
            R = R_next 
            μ = μ_next
            v_par_last_half = v_par_next_half
            gyrophase = gyrophase_next
            r = R # update for inside plasma check
            dt = dt_GC # update for FO-GCA switch checks
        end
        t += dt
    end

    return sample_times, positions, velocities, push_types
end

function vel_from_E_per_q(E_kin, q, m)
    return sqrt(2*q*E_kin/m)
end 


#using Base.Threads
using Distributed
using SharedArrays
using HDF5
using Dates
using ProgressBars
function integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; μ_z0=-0.125, σ_z0=0.005, σ_xy0=0.001, 
                              q=e.val, m=23*m_u.val, N_ions=100, B=[0.,0.,7.], 
                              n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, r_b=0.001,
                              neutral_masses=[], neutral_pressures_mbar=[], alphas=[], 
                              CX_fractions=[], T_n=300.,t_end=3.7, dt=1e-08, sample_every=100, 
                              velocity_diffusion=true, seed=nothing, fname="ion_orbits")
    now = Dates.now()
    datetime = Dates.format(now, "yyyy-mm-dd_HHMM") # save start time for run info 
    println("\n##### STARTING ION ORBIT TRACING #####\n")
    println("number of cores = ", nprocs())
    println("number of workers = ", nworkers())
    println("number of ions = ", N_ions)
    
    ### Load potential map
    if n_b == 0 && r_b > 0
        println("\nUsing vacuum potential (i.e. r_b=0) as n_b == 0.\n")
        r_b = 0 # ensure vacuum potential is used for plasma-off runs
    end 
    V_sitp = get_V_sitp(r_b)

    ### Randomly initialize ion energies and radial positions
    # TODO consider initial conditions
    if isnothing(seed)
        rng = default_rng()
    else
        rng = MersenneTwister(seed)
    end
    x0 = rand(rng, Normal(0.0, σ_xy0), N_ions)
    y0 = rand(rng, Normal(0.0, σ_xy0), N_ions)
    # r0 = rand(rng, Normal(0.0, σ_r0), N_ions)
    # ϕ0 = rand(rng, N_ions)*pi
    # x0 = [ r0[pid]*cos(ϕ0[pid]) for pid in range(1,N_ions)]
    # y0 = [ r0[pid]*sin(ϕ0[pid]) for pid in range(1,N_ions)]
    z0 = rand(rng, Normal(μ_z0, σ_z0), N_ions) # TODO: Set μ_pos0 to capture well centre
    pos0 = [[x0[pid], y0[pid], z0[pid]] for pid in range(1,N_ions)]
    E0_par = [rand(rng, truncated(Normal(μ_E0_par, σ_E0_par); lower=V_itp(pos0[pid], V_sitp=V_sitp))) for pid in range(1,N_ions)] #E0_par = rand(Normal(μ_E0_par, σ_E0_par))
    v0_par = vel_from_E_per_q.(E0_par .- V_itp.(pos0, V_sitp=V_sitp), q, m)
    E0_perp = rand(rng, truncated(Normal(0., σ_E0_perp); lower=0.), N_ions) # TODO: check distributions
    ζ0 = rand(rng, N_ions)*2*pi
    vx0 = vel_from_E_per_q.(E0_perp, q, m).*cos.(ζ0)
    vy0 = vel_from_E_per_q.(E0_perp, q, m).*sin.(ζ0)

    ### Loop over ions
    times = range(0.0, step=dt, stop=t_end)
    N_samples = Int64(round(t_end/(sample_every*dt))) + 1
    rngs = MersenneTwister.(rand(rng, (0:100000000000000000), N_ions))
    processed = SharedArray{Float64}(N_ions)
    sample_time_hists = SharedArray{Float64}(N_ions,N_samples)
    position_hists = SharedArray{Float64}(N_ions,N_samples,3) 
    velocity_hists = SharedArray{Float64}(N_ions,N_samples,3) 
    charge_hists = SharedArray{Float64}(N_ions,N_samples) 
    mass_hists = SharedArray{Float64}(N_ions,N_samples)
    coll_type_ids = Dict("glanzing" => 1, "Langevin" => 2, "CX" => 3)
    coll_counts = SharedArray{Int64}(N_ions,length(neutral_masses), length(coll_type_ids))
    @sync @distributed for i in range(1, N_ions)
        u_last_half = [vx0[i], vy0[i], v0_par[i]]
        sample_times, positions, velocities, charge_hist, mass_hist, coll_count_dicts = integrate_orbit_with_friction(
                                                                            times, pos0[i], u_last_half; q=q, m=m, B=B, 
                                                                            n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, r_b=r_b, 
                                                                            neutral_masses=neutral_masses, 
                                                                            neutral_pressures_mbar=neutral_pressures_mbar, 
                                                                            alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                                                            dt=dt, sample_every=sample_every, 
                                                                            velocity_diffusion=velocity_diffusion, rng=rngs[i])
        sample_time_hists[i,:] = sample_times 
        position_hists[i,:,:] = positions 
        velocity_hists[i,:,:] = velocities
        charge_hists[i,:] = charge_hist
        mass_hists[i,:] = mass_hist
        for k in range(1,length(neutral_masses))
            for (coll_type, count) in coll_count_dicts[k]
                coll_counts[i,k,coll_type_ids[coll_type]] = count
            end
        end
        processed[i] = true
        println(string(Int64(sum(processed))) * "/" * string(N_ions) * " ion orbits traced.")
    end 
    println("\n Particle tracing completed. \n")

    # Write run results to HDF5 file 
    if !isnothing(fname)
        fid = h5open(fname * ".h5", "w")

        # Write general run info to attributes 
        create_group(fid, "RunInfo")
        info = fid["RunInfo"]
        info["filepath"] = fname * ".h5"
        info["datetime"] = datetime
        info["t_end"] = t_end
        info["dt"] = dt
        info["sample_every"] = sample_every
        info["N_ions"] = N_ions
        info["μ_E0_par"] = μ_E0_par
        info["σ_E0_par"] = σ_E0_par
        info["σ_E0_perp"] = σ_E0_perp
        info["μ_z0"] = μ_z0
        info["σ_xy0"] = σ_xy0
        info["q"] = q
        info["m"] = m
        info["B"] = B
        info["n_b"] = n_b
        info["T_b"] = T_b
        info["q_b"] = q_b
        info["m_b"] = m_b
        info["r_b"] = r_b
        info["neutral_masses"] = neutral_masses
        info["neutral_pressures_mbar"] = neutral_pressures_mbar
        info["alphas"] = alphas
        info["CX_fractions"] = CX_fractions
        info["T_n"] = T_n
        info["coll_types"] = [k for (k,_) in coll_type_ids]
        info["velocity_diffusion"] = velocity_diffusion
        if isnothing(seed) 
            info["seed"] = NaN
        else
            info["seed"] = seed
        end

        # Write ion orbit data 
        create_group(fid, "IonOrbits")
        orbs = fid["IonOrbits"]
        orbs["sample_time_hists"] = sample_time_hists
        orbs["position_hists"] = position_hists 
        orbs["velocity_hists"] = velocity_hists
        orbs["charge_hists"] = charge_hists
        orbs["mass_hists"] = mass_hists
        orbs["coll_counts"] = coll_counts
        close(fid)
        println("Data written to " * fname * ".h5")
    end
end