using Plots

##### Define particle trajectory diagnostics
function plot_trajectory_data(times, positions, velocities; q=e.val, m=m_u.val, n_smooth_E=0)
    plot(times, 10*getindex.(positions,1), xlabel="t (s)", ylabel="x, y, z (m)", label="x*10")
    plot!(times, 10*getindex.(positions,2), label="y*10")
    display(plot!(times, getindex.(positions,3), label="z"))

    display(plot(getindex.(positions,1), getindex.(positions,2), xlabel="x", ylabel="y", label=""))

    display(plot(getindex.(positions,1), getindex.(positions,3), xlabel="x", ylabel="z", label=""))

    display(plot(times, getindex.(velocities,2), xlabel="t", ylabel="vy", label=""))

    display(plot(times, getindex.(velocities,3), xlabel="t", ylabel="vz", label=""))
    
    E_kins = [0.5*m*norm(vel)^2 for vel in velocities]/q
    E_pots = V_itp.(positions)
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