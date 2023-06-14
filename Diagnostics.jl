using Plots
using HDF5
using LaTeXStrings
using LinearAlgebra
import Base.@kwdef
default(fmt = :png) # prevent slowdown from dense figures
include("ParticlePushers.jl")

##### Define particle trajectory diagnostics
function plot_trajectory_data(times, positions, velocities; q=e.val, m=m_u.val, r_b=0.011, n_smooth_E=0)
    plot(times, 10*getindex.(positions,1), xlabel="t (s)", ylabel="x, y, z (m)", label="x*10")
    plot!(times, 10*getindex.(positions,2), label="y*10")
    display(plot!(times, getindex.(positions,3), label="z"))

    display(plot(getindex.(positions,1), getindex.(positions,2), xlabel="x", ylabel="y", label=""))

    display(plot(getindex.(positions,1), getindex.(positions,3), xlabel="x", ylabel="z", label=""))

    display(plot(times, getindex.(velocities,2), xlabel="t", ylabel="vy", label=""))

    display(plot(times, getindex.(velocities,3), xlabel="t", ylabel="vz", label=""))
    
    V_sitp = get_V_sitp(r_b)
    E_kins = [0.5*m*norm(vel)^2 for vel in velocities]/q
    E_pots = V_itp.(positions, V_sitp)
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
    datetime::String 
    t_end::Float64
    dt::Float64 
    sample_every::Int64 
    N_ions::Int64
    μ_E0_par::Float64
    σ_E0_par::Float64
    σ_E0_perp::Float64
    μ_z0::Float64
    σ_xy0::Float64
    q::Float64
    m::Float64
    B::Vector{Float64}
    n_b::Float64
    T_b::Float64
    q_b::Float64
    m_b::Float64
    r_b::Float64
    neutral_pressures_mbar::Vector{Float64}
    alphas::Vector{Float64}
    CX_fractions::Vector{Float64}
    T_n::Float64
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
                        datetime = read(info["datetime"]), 
                        t_end = read(info["t_end"]),
                        dt = read(info["dt"]),
                        sample_every = read(info["sample_every"]),
                        N_ions= read(info["N_ions"]),
                        μ_E0_par = read(info["μ_E0_par"]),
                        σ_E0_par = read(info["σ_E0_par"]),
                        σ_E0_perp = read(info["σ_E0_perp"]),
                        μ_z0 = read(info["μ_z0"]),
                        σ_xy0 = read(info["σ_xy0"]),
                        q = read(info["q"]),
                        m = read(info["m"]),
                        B = read(info["B"]),
                        n_b = read(info["n_b"]),
                        T_b = read(info["T_b"]),
                        q_b = read(info["q_b"]),
                        m_b = read(info["m_b"]),
                        r_b = read(info["r_b"]),
                        neutral_pressures_mbar = read(info["neutral_pressures_mbar"]),
                        alphas = read(info["alphas"]),
                        CX_fractions = read(info["CX_fractions"]),
                        T_n = read(info["T_n"]),
                        velocity_diffusion = read(info["velocity_diffusion"]),
                        seed = read(info["seed"]),
                      )
    close(fid)

    return run_info
end

function print_RunInfo()
    nothing
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

function plot_run_results(fname; rel_path="/Tests/OutputFiles/")
    """Plot diagnostics for ion orbit data stored in HDF5 file"""
    run_info = get_run_info(fname; rel_path=rel_path)

    path = string(@__DIR__) * rel_path * fname 
    fid = h5open(path, "r")
    orbs = fid["IonOrbits"] 
    sample_times = read(orbs["sample_time_hists"])[1,:]
    position_hists = read(orbs["position_hists"])
    velocity_hists = read(orbs["velocity_hists"])
    N_ions = run_info.N_ions
    q = run_info.q
    m = run_info.m
    V_sitp = get_V_sitp(run_info.r_b) # load potential map

    # Total energy evolutions
    f = plot(xlabel="Time (s)", ylabel="Total ion energy (eV/q)",  legend=false)
    total_energies = []
    for i in range(1,N_ions)
        E_tot = [0.5*m*norm(@views Vector(velocity_hists[i,it,:]))^2/q + @views  V_itp(Vector(position_hists[i,it,:]), V_sitp) for it in range(1,length(sample_times))]  
        plot!(sample_times, E_tot)
        push!(total_energies, E_tot)
    end 
    mean_energies = [mean(getindex.(total_energies, i)) for i in range(1,length(sample_times))]
    std_energies = [std(getindex.(total_energies, i)) for i in range(1,length(sample_times))]    
    err_mean_energies = std_energies/sqrt(N_ions) # TODO: use detectable ion number
    plot!(sample_times, mean_energies, linewidth=3, linecolor="black")
    display(f)

    # Mean energy evolution
    f = plot(xlabel="Time (s)", ylabel="Mean ion energy (eV/q)", legend=false)
    plot!(sample_times, mean_energies, 
          ribbon=err_mean_energies, linewidth=3, linecolor="black")
    display(f)

    # Sample standard deviation of total energy evolutions
    f = plot(xlabel="Time (s)", ylabel=L"Std. dev. of $E$ (eV/q)", legend=false)
    plot!(sample_times, std_energies)
    display(f)

    # Longitudinal energy evolutions 
    f = plot(xlabel="Time (s)", ylabel=L"Longitudinal ion energy $E_\parallel$ (eV/q)",  legend=false)
    par_energies = []
    for i in range(1,N_ions)
        #E_par = 0.5*m*getindex.(velocities,3).^2/q + V_itp.(positions)
        E_par = [0.5*m*norm(@views Vector(velocity_hists[i,:,3][it,:]))^2/q + @views V_itp(Vector(position_hists[i,:,:][it,:]), V_sitp) for it in range(1,length(sample_times))]  
        plot!(sample_times, E_par)
        push!(par_energies, E_par)
    end 
    mean_par_energies = [mean(getindex.(par_energies, it)) for it in range(1,length(sample_times))]
    std_par_energies = [std(getindex.(par_energies, it)) for it in range(1,length(sample_times))]                  
    err_mean_par_energies = std_par_energies/sqrt(N_ions) # TODO: use detectable ion number
    plot!(sample_times, mean_par_energies, linewidth=3, linecolor="black")
    display(f)
  
    # Mean longitudinal energy evolution
    f = plot(xlabel="Time (s)", ylabel=L"Mean longitudinal ion energy $\langle E_\parallel\rangle$ (eV/q)", legend=false)
    plot!(sample_times, mean_par_energies, 
          ribbon=err_mean_par_energies, linewidth=3, linecolor="black")
    display(f)

    # Sample standard deviation of longitudinal energy evolutions
    f = plot(xlabel="Time (s)", ylabel=L"Std. dev. of  $E_\parallel$ (eV/q)", legend=false)
    plot!(sample_times, std_par_energies)
    display(f)

    # 3D scatter of final ion positions
    it = length(sample_times)
    X = [@views position_hists[i,it,1] for i in range(1,N_ions)] 
    Y = [@views position_hists[i,it,2] for i in range(1,N_ions)]  
    Z = [@views position_hists[i,it,3] for i in range(1,N_ions)]                       

    f = scatter(Z, X, Y, xlabel="z (m)", ylabel="x (m)", zlabel="y (m)", 
                legend=false, title="Final ion distribution")
    display(f)

    # ZR scatter of final ion positions
    f = scatter(Z, sqrt.(X.^2 + Y.^2), xlabel="z (m)", ylabel="r (m)", xlim=(-0.15,0.15), 
                legend=false, title="Final ion distribution")
    display(f)

    # Collect axial and radial displacement metrics
    radial_offsets = []
    axial_offsets = []
    for i in range(1, N_ions)
        R = sqrt.(@views position_hists[i,:,1].^2 .+ @views position_hists[i,:,2].^2) # sqrt.(getindex.(orbit.positions, 1).^2 .+ getindex.(orbit.positions, 2).^2)
        Z = @views position_hists[i,:,3] #getindex.(orbit.positions, 3)
        push!(radial_offsets, R)
        push!(axial_offsets, Z)
    end
    mean_radial_offsets = [mean(getindex.(radial_offsets, it)) for it in range(1,length(sample_times))]
    err_mean_radial_offsets = [std(getindex.(radial_offsets, it))/sqrt(N_ions) for it in range(1,length(sample_times))]
    RMS_radial_offsets = [sqrt(mean((getindex.(radial_offsets, it)).^2)) for it in range(1,length(sample_times))]
    mean_axial_offsets = [mean(getindex.(axial_offsets, it)) for it in range(1,length(sample_times))]
    err_mean_axial_offsets = [std(getindex.(axial_offsets, it))/sqrt(N_ions) for it in range(1,length(sample_times))]
    RMS_axial_offsets = [sqrt(mean((getindex.(axial_offsets, it)).^2)) for it in range(1,length(sample_times))]

    # Radial displacements
    f = plot(legend=false, xlabel="t (s)", ylabel="Transverse displacement (m)")
    for R in radial_offsets
        plot!(sample_times, R)
    end
    plot!(sample_times, mean_radial_offsets, linewidth=3, 
          linecolor="black",)
    display(f)

    # Mean radial displacements
    f = plot(sample_times, mean_radial_offsets, legend=false, 
            ribbon=err_mean_radial_offsets, xlabel="t (s)", ylabel="Mean transverse displacement (m)")
    display(f)

    # RMS radial displacements
    f = plot(sample_times, RMS_radial_offsets, xlabel="t (s)", 
            legend=false, ylabel="RMS transverse displacement (m)")
    display(f)

    # Axial positions 
    f = plot(legend=false, xlabel="t (s)", ylabel="Longitudinal position (m)")
    for Z in axial_offsets
        plot!(sample_times, Z)
    end
    plot!(sample_times, mean_axial_offsets, linecolor="black", 
          linewidth=3)
    display(f)

    # Mean axial positions
    f = plot(sample_times, mean_axial_offsets, ribbon=err_mean_axial_offsets, 
            legend=false, xlabel="t (s)", ylabel="Mean longitudinal position (m)")
    display(f)

    # RMS axial positions
    f = plot(sample_times, RMS_axial_offsets, ribbon=err_mean_axial_offsets, 
            legend=false, xlabel="t (s)", ylabel="RMS longitudinal position (m)")
    display(f)

    close(fid)
end                                                                        