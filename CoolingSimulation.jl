using LinearAlgebra
using Plots
#using CSV
#using DataFrames
#using Interpolations
using Statistics
using Random
using StaticArrays
import Distributions: Normal, truncated
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
import SpecialFunctions: erf
import StatsBase: sample, ProbabilityWeights
include("ParticlePushers.jl")
include("IonNeutralCollisions.jl")
struct RunResults
    start_Date::Float64 
    start_time::Float64
    run_time_s::Float64
    times::Vector{Float64}
    positions::Vector{Float64}
    velocities::Vector{Float64}
    T_e::Vector{Float64}
end

##### Define Orbit integration function
function integrate_orbit_with_friction(times, r, u_last_half; q=q, m=m, B=B, 
                                       n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, 
                                       neutral_masses=[], neutral_pressures_mbar=[], alphas=[], 
                                       CX_fractions=[], T_n=300.,
                                       dt=1e-10, sample_every=1, velocity_diffusion=true)
    t_end = times[end]
    N_samples = Int64(round(t_end/(sample_every*dt))) + 1
    sample_times = zeros(Float64, N_samples) #Vector{Float64}([])
    positions = zeros(Float64, N_samples, 3) #Vector{Vector{Float64}}([])
    velocities = zeros(Float64, N_samples, 3) #Vector{Vector{Float64}}([])
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
    coll_counts = zeros(length(neutral_masses))
    #coll_type = @SVector
    #TODO: Add collision type counter
    function inside_plasma()
        return Bool(-0.03 < r[3] < 0.05 && n_b > 0.)
    end 
    for t in times 
        # Get E-field and step
        update_E!(E, r) # E = E_itp(r)
        if inside_plasma() # TODO: derive plasma bounds from Warp PIC data 
            r_next, u_next_half = Boris_push_with_friction(r, u_last_half, E, B, dt, q=q, m=m, dW, norm_dist,
                                                           n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                           velocity_diffusion=velocity_diffusion)
        else
            r_next, u_next_half = Boris_push(r, u_last_half, E, B, dt, q=q, m=m)
        end

        # Sample time-centred particle data
        if mod(t, sample_every*dt) < dt
            i = Int64(round(t/(sample_every*dt))) + 1
            sample_times[i] = t #push!(sample_times, t)
            positions[i,:] = r #push!(positions, r)
            velocities[i,:] = (u_last_half + u_next_half)/2 #push!(velocities, (u_last_half + u_next_half)/2) 
        end

        # Model Monte Carlo ion-neutral collision 
        if model_MC_collisions
            update_MFPs!(MFPs, u_next_half, q, m, neutral_masses, alphas, neutral_pressures_mbar, T_n) #MFPs = [get_mean_free_path(u_next_half, q, m, m_n=neutral_masses[i], alpha=alphas[i], p_n_mbar=neutral_pressures_mbar[i],T_n=T_n) for i in range(1,length(neutral_masses))]
            update_v_effs!(v_effs, u_next_half, neutral_masses, T_n) #v_effs = [get_eff_collision_speed(u_next_half, m_n=m_n, T_n=T_n) for m_n in neutral_masses]
            update_coll_probs!(coll_probs, MFPs, v_effs, dt) ##coll_probs = 1. .- exp.(-v_effs./MFPs.*dt)
            if rand(Float64) <= sum(coll_probs) 
                i_coll = sample(1:length(neutral_masses), ProbabilityWeights(coll_probs/sum(coll_probs))) # randomly select neutral collision partner
                u_next_half, coll_type = ion_neutral_collision(u_next_half, q, m, m_n=neutral_masses[i_coll], 
                                                               CX_frac=CX_fractions[i_coll],
                                                               alpha=alphas[i_coll], T_n=T_n)
                coll_counts[i_coll] += 1
            end
        end

        # Update particle data for next time step
        r, u_last_half = r_next, u_next_half 
    end 

    return sample_times, positions, velocities
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
                                                  velocity_diffusion=true)
    """"
    Only applicable for B-field aligned with z-axis 

    s vector defined in opposite direction as in Rode2023
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

struct IonOrbit
    sample_times::Vector{Float64}
    positions::Vector{Vector{Float64}}
    velocities::Vector{Vector{Float64}}
    T_b::Vector{Float64}
end

#using Base.Threads
using Distributed
using SharedArrays
using HDF5
function integrate_ion_orbits(μ_E0_par, σ_E0_par, σ_E0_perp; 
                              μ_z0=-0.125, σ_z0=0.005, σ_xy0=0.001, q=e.val, m=23*m_u.val, N_ions=100, B=B, 
                              n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, 
                              neutral_masses=[], neutral_pressures_mbar=[], alphas=[], 
                              CX_fractions=[], T_n=300.,
                              t_end=3.7, dt=1e-08, sample_every=100, velocity_diffusion=true, fname=nothing)

    ### Randomly initialize ion energies and radial positions
    # TODO add seed and truncate E0_par normal distribution
    x0 = rand(Normal(0.0, σ_xy0), N_ions)
    y0 = rand(Normal(0.0, σ_xy0), N_ions)
    # r0 = rand(Normal(0.0, σ_r0), N_ions)
    # ϕ0 = rand(N_ions)*pi
    # x0 = [ r0[pid]*cos(ϕ0[pid]) for pid in range(1,N_ions)]
    # y0 = [ r0[pid]*sin(ϕ0[pid]) for pid in range(1,N_ions)]
    z0 = rand(Normal(μ_z0, σ_z0), N_ions) # TODO: Set μ_pos0 to capture well centre
    pos0 = [[x0[pid], y0[pid], z0[pid]] for pid in range(1,N_ions)]
    E0_par = [rand(truncated(Normal(μ_E0_par, σ_E0_par); lower=V_itp(pos0[pid]))) for pid in range(1,N_ions)] #E0_par = rand(Normal(μ_E0_par, σ_E0_par))
    v0_par = vel_from_E_per_q.(E0_par .- V_itp.(pos0), q, m)
    E0_perp = rand(truncated(Normal(0., σ_E0_perp); lower=0.), N_ions) # TODO: check distributions
    ζ0 = rand(N_ions)*2*pi
    vx0 = vel_from_E_per_q.(E0_perp, q, m).*cos.(ζ0)
    vy0 = vel_from_E_per_q.(E0_perp, q, m).*sin.(ζ0)

    ### Loop over ions
    times = range(0.0, step=dt, stop=t_end)
    ion_orbits = []
    N_samples = Int64(round(t_end/(sample_every*dt))) + 1
    all_sample_times = SharedArray{Float64}(N_ions,N_samples)
    all_positions = SharedArray{Float64}(N_ions,N_samples,3) #SharedArray{Float64}(N_ions,N_samples,3)
    all_velocities = SharedArray{Float64}(N_ions,N_samples,3) #SharedArray{Float64}(N_ions,N_samples,3)
    @sync @distributed for pid in range(1, N_ions)
        u_last_half = [vx0[pid], vy0[pid], v0_par[pid]]
        sample_times, positions, velocities = integrate_orbit_with_friction(times, pos0[pid], u_last_half; q=q, m=m, B=B, 
                                                                            n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                                            neutral_masses=neutral_masses, 
                                                                            neutral_pressures_mbar=neutral_pressures_mbar, 
                                                                            alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                                                            dt=dt, sample_every=sample_every, 
                                                                            velocity_diffusion=velocity_diffusion)
        #push!(ion_orbits, IonOrbit(sample_times, positions, velocities, [T_b]))
        all_sample_times[pid,:] = sample_times 
        all_positions[pid,:,:] = positions 
        all_velocities[pid,:,:] = velocities
    end 
    println("Particle tracing completed.")

    # Write run results to HDF5 file 
    #fname = "@__DIR__/OutputFiles/test_run_plasma_on"
    if !isnothing(fname)
        fid = h5open(fname * ".h5", "w")
        create_group(fid, "IonOrbits")
        orbs = fid["IonOrbits"]
        orbs["sample_time_hists"] = all_sample_times
        orbs["position_hists"] = all_positions 
        orbs["velocity_hists"] = all_velocities
        close(fid)
        println("Data written to " * fname * ".h5")
    end

    ### Plot results 
    # Total energy evolutions
    # f1 = plot(xlabel="Time (s)", ylabel="Total ion energy (eV/q)",  legend=false)
    # total_energies = []
    # for orbit in ion_orbits
    #     E_tot = 0.5*m*norm.(orbit.velocities).^2/q + V_itp.(orbit.positions)
    #     plot!(orbit.sample_times, E_tot)
    #     push!(total_energies, E_tot)
    # end 
    # mean_energies = [mean(getindex.(total_energies, i)) for i in range(1,length(ion_orbits[1].sample_times))]
    # err_mean_energies = [std(getindex.(total_energies, i))/sqrt(N_ions) for i in range(1,length(ion_orbits[1].sample_times))] 
    # plot!(ion_orbits[1].sample_times, mean_energies, linewidth=3, linecolor="black")
    # display(f1)

    # # Mean energy evolution
    # f2 = plot(xlabel="Time (s)", ylabel="Mean ion energy (eV/q)", legend=false)
    # plot!(ion_orbits[1].sample_times, mean_energies, 
    #       ribbon=err_mean_energies, linewidth=3, linecolor="black")
    # display(f2)

    # # Total longitudinal energy evolutions 
    # f3 = plot(xlabel="Time (s)", ylabel="Longitudinal ion energy (eV/q)",  legend=false)
    # par_energies = []
    # for orbit in ion_orbits
    #     E_par = 0.5*m*getindex.(orbit.velocities,3).^2/q + V_itp.(orbit.positions)
    #     plot!(orbit.sample_times, E_par)
    #     push!(par_energies, E_par)
    # end 
    # mean_par_energies = [mean(getindex.(par_energies, i)) for i in range(1,length(ion_orbits[1].sample_times))]
    # err_mean_par_energies = [std(getindex.(par_energies, i))/sqrt(N_ions) for i in range(1,length(ion_orbits[1].sample_times))] 
    # plot!(ion_orbits[1].sample_times, mean_par_energies, linewidth=3, linecolor="black")
    # display(f3)
    
    # # Mean longitudinal energy evolution
    # f4 = plot(xlabel="Time (s)", ylabel="Mean longitudinal ion energy (eV/q)", legend=false)
    # plot!(ion_orbits[1].sample_times, mean_par_energies, 
    #       ribbon=err_mean_par_energies, linewidth=3, linecolor="black")
    # display(f4)

    # # 3D scatter of final ion positions
    # it = length(ion_orbits[1].sample_times)
    # X = [orbit.positions[it][1] for orbit in ion_orbits]
    # Y = [orbit.positions[it][2] for orbit in ion_orbits]
    # Z = [orbit.positions[it][3] for orbit in ion_orbits]

    # f = scatter(Z, X, Y, xlabel="z (m)", ylabel="x (m)", zlabel="y (m)", 
    #             legend=false, title="Final ion distribution")
    # display(f)

    # # ZR scatter of final ion positions
    # f = scatter(Z, sqrt.(X.^2 + Y.^2), xlabel="z (m)", ylabel="r (m)", xlim=(-0.15,0.15), 
    #             legend=false, title="Final ion distribution")
    # display(f)

    # # Collect axial and radial displacement metrics
    # radial_offsets = []
    # axial_offsets = []
    # for orbit in ion_orbits
    #     R = sqrt.(getindex.(orbit.positions, 1).^2 .+ getindex.(orbit.positions, 2).^2)
    #     Z = getindex.(orbit.positions, 3)
    #     push!(radial_offsets, R)
    #     push!(axial_offsets, Z)
    # end
    # mean_radial_offsets = [mean(getindex.(radial_offsets, i)) for i in range(1,length(ion_orbits[1].sample_times))]
    # err_mean_radial_offsets = [std(getindex.(radial_offsets, i))/sqrt(N_ions) for i in range(1,length(ion_orbits[1].sample_times))]
    # RMS_radial_offsets = [sqrt(mean((getindex.(radial_offsets, i)).^2)) for i in range(1,length(ion_orbits[1].sample_times))]
    # mean_axial_offsets = [mean(getindex.(axial_offsets, i)) for i in range(1,length(ion_orbits[1].sample_times))]
    # err_mean_axial_offsets = [std(getindex.(axial_offsets, i))/sqrt(N_ions) for i in range(1,length(ion_orbits[1].sample_times))]
    # RMS_axial_offsets = [sqrt(mean((getindex.(axial_offsets, i)).^2)) for i in range(1,length(ion_orbits[1].sample_times))]
    
    # # Radial displacements
    # f = plot(legend=false, xlabel="t (s)", ylabel="Radial displacement (m)")
    # for R in radial_offsets
    #     plot!(ion_orbits[1].sample_times, R)
    # end
    # plot!(ion_orbits[1].sample_times, mean_radial_offsets, linewidth=3, 
    #       linecolor="black",)
    # display(f)

    # # Mean radial displacements
    # f = plot(ion_orbits[1].sample_times, mean_radial_offsets, legend=false, 
    #         ribbon=err_mean_radial_offsets, xlabel="t (s)", ylabel="Mean radial displacement (m)")
    # display(f)

    # # RMS radial displacements
    # f = plot(ion_orbits[1].sample_times, RMS_radial_offsets, xlabel="t (s)", 
    #         legend=false, ylabel="RMS radial displacement (m)")
    # display(f)
    
    # # Axial positions 
    # f = plot(legend=false, xlabel="t (s)", ylabel="Axial position (m)")
    # for Z in axial_offsets
    #     plot!(ion_orbits[1].sample_times, Z)
    # end
    # plot!(ion_orbits[1].sample_times, mean_axial_offsets, linecolor="black", 
    #       linewidth=3)
    # display(f)

    # # Mean axial positions
    # f = plot(ion_orbits[1].sample_times, mean_axial_offsets, ribbon=err_mean_axial_offsets, 
    #         legend=false, xlabel="t (s)", ylabel="Mean axial position (m)")
    # display(f)
    
    # # RMS axial positions
    # f = plot(ion_orbits[1].sample_times, RMS_axial_offsets, ribbon=err_mean_axial_offsets, 
    #         legend=false, xlabel="t (s)", ylabel="RMS axial position (m)")
    # display(f)
                
    #return ion_orbits
end