#!/usr/bin/julia
using LinearAlgebra
using Plots
using CSV
using DataFrames
using Interpolations
using Statistics
using Random
using StaticArrays
import Distributions: Normal
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, h, μ_B
import SpecialFunctions: erf

##### Define Boris pushers without friction or diffusion
function gamma_from_u(u)
    return sqrt(1 + sum((u./c_0.val).^2))
end

""" Boris particle pusher based on advanced implementation from Zenitani et al.

    See Zenitani et al., Physics of Plasmas 25, 112110 (2018)
    
    For a Python implementation see https://stanczakdominik.github.io/posts/on-the-recent-on-the-boris-solver-in-particle-in-cell-simulations-paper/
"""
function Boris_push(r, u_last_half, E, B, dt; q=e.val, m=m_u.val) 
    eps_dt = 0.5*q/m*E*dt 
    u_m = u_last_half + eps_dt # Eqn. 3
    
    norm_B = norm(B)
    theta = q/m*dt/gamma_from_u(u_m)*norm_B # Eqn. 6
    b = B/norm_B
    
    u_m_parallel = dot(u_m, b)*b # Eqn. 11
    u_p = u_m_parallel + (u_m - u_m_parallel)*cos(theta) + cross(u_m, b)*sin(theta) # Eqn. 12
    u_next_half = u_p + eps_dt # Eqn. 5
    r_next = r + u_next_half/gamma_from_u(u_next_half)*dt # Eqn. 1
    
    return r_next, u_next_half
end

""" Integrate ion orbit in a analytically defined electric field"""
function integrate_orbit_global_E(times; r=r,u_last_half=u_last_half, q=q, m=m, E=E, B=B, dt=1e-10, sample_every=1)
    sample_times = []
    positions = []
    velocities = []
    for t in times 
        # Step
        r_next, u_next_half = Boris_push(r, u_last_half, E, B, dt, q=q, m=m)
        # Sample time-centred particle data
        if mod(t, sample_every*dt) < dt
            push!(sample_times, t)
            push!(positions, r)
            push!(velocities, (u_last_half + u_next_half)/2) 
        end
        # Update particle data for next time step
        r, u_last_half = r_next, u_next_half 
    end 
    return sample_times, positions, velocities
end

""" Integrate ion orbit inside an ideal Penning trap"""
function integrate_PT_orbit(times; r=r,u_last_half=u_last_half, q=q, m=m, B=B, dt=1e-10, sample_every=1)
    sample_times = []
    positions = []
    velocities = []
    for t in times 
        # Get E-field and step
        E = E_trap(r)
        r_next, u_next_half = Boris_push(r, u_last_half, E, B, dt, q=q, m=m)
        # Sample time-centred particle data
        if mod(t, sample_every*dt) < dt
            push!(sample_times, t)
            push!(positions, r)
            push!(velocities, (u_last_half + u_next_half)/2) 
        end
        # Update particle data for next time step
        r, u_last_half = r_next, u_next_half 
    end 
    return sample_times, positions, velocities
end

##### Define potential and E-field interpolation functions 
""" Load 2D potential map from CSV file"""
function load_PIC_potentials(fname; path="./")
    csv_file = CSV.File(path * fname)
    return DataFrame(csv_file)
end 

""" Define scaled cubic BSpline interpolation based on 2D potential map"""
function define_potential_interpolation(df)
    dropcol(M::AbstractMatrix, j) = M[:, deleteat!(collect(axes(M, 2)), j)]
    ZR_potentials = dropcol(Matrix(df), 1)
    #display(plot(ZR_potentials))
    rp = [parse(Float64, n) for n in names(df)[2:end]]
    zp = Matrix(df)[:,1]
    rp_range = range(extrema(rp)..., length=size(rp)[1])
    zp_range = range(extrema(zp)..., length=size(zp)[1])
        
    #ZR_potentials = transpose([phi_trap([ri,0,zi]) for ri in rp, zi in zp]) # quadrupole field as for PT example for debugging
    itp = interpolate(ZR_potentials, BSpline(Cubic(Flat(OnGrid()))))
        
    return scale(itp, zp_range, rp_range)
end 

""" Create scaled interpolation of potential grid for a plasma with radius `r_b`"""
function get_V_sitp(r_b)
    path = string(@__DIR__)*"/PotentialMaps/Runs with 4.2E07 electrons, 100um r grid/Plasma radius "*string(r_b*1e03)*"mm/"
    fname = "final_RZ_potential.txt" 
    df = load_PIC_potentials(fname, path=path)
    return define_potential_interpolation(df)
end

""" Get interpolated potential at arbitrary position in simulation volume [eV]"""
function V_itp(pos; V_sitp=nothing)::Float64 
    r = norm(pos[1:2]) 
    return V_sitp(pos[3], r)
end

""" Get interpolated on-axis potential [eV]"""
function V_itp_on_axis(z, V_sitp)
    return [V_itp([0.,0.,zi], V_sitp=V_sitp) for zi in z]
end

##### Function for getting electron density data
""" Load 2D electron density map from CSV file"""
function load_electron_density_map(fname; path="./")
    csv_file = CSV.File(path * fname)
    return DataFrame(csv_file)
end 

""" Create scaled interpolation of electron density grid for a plasma of radius `r_b`"""
function get_n_e_sitp(r_b)
    if r_b == 0.0 
        return nothing 
    end 
    path = string(@__DIR__)*"/PotentialMaps/Runs with 4.2E07 electrons, 100um r grid/Plasma radius "*string(r_b*1e03)*"mm/"
    fname = "electron_density_"*string(r_b*1e03)*"mm_plasma_radius.txt" 
    df = load_PIC_potentials(fname, path=path)
    return define_potential_interpolation(df)
end

""" Get interpolated potential at arbitrary position in simulation volume [1/m^3]"""
function n_e_itp(pos; n_e_sitp=nothing)::Float64 
    if n_e_sitp == nothing
        return 0.0 
    end 
    r = norm(pos[1:2]) 
    return n_e_sitp(pos[3], r)
end

""" Update interpolated plasma electron number density [1/m^3]"""
function update_n_e!(n_e::Vector{Float64}, pos, n_e_sitp)
    n_e[1] = n_e_itp(pos, n_e_sitp=n_e_sitp) 
end

##### Functions for grabbing electric field strength
""" Get azimuthal angle of position vector in cylindrical coordinate system"""
function get_azimuthal_angle(pos::SVector{3,Float64})
    if pos[1] > 0 
        phi = atan(pos[2]/pos[1])
    elseif pos[1] < 0 && pos[2] >= 0 
        phi = atan(pos[2]/pos[1]) + pi 
    elseif pos[1] < 0 && pos[2] < 0 
        phi = atan(pos[2]/pos[1]) - pi
    elseif pos[1]==0 && pos[2]!=0 
        phi = pi/2*sign(pos[2])
    else 
        phi = NaN
    end
    return phi
end

""" Get interpolated electric field vector [V/m]"""
function E_itp(pos::SVector{3,Float64})
    r = norm(pos[1:2])
    if r != 0.0 
        phi = get_azimuthal_angle(pos)
    else 
        phi = 0.0
    end
    dVdz::Float64, dVdr::Float64 = ∇V(r, pos[3]) #gradient(sitp, pos[3], r)
    return -1.0*[dVdr*cos(phi), dVdr*sin(phi), dVdz]
end 

""" Get interpolated electric field vector [V/m]"""
function E_itp!(E, pos) #TODO: Remove if adaptice GC solver is updated or removed 
    r = norm(pos[1:2])
    if r != 0.0 
        phi = get_azimuthal_angle(pos)
    else 
        phi = 0.0
    end
    gradV = update_gradV!(gradV, r, pos[3]) #dVdz::Float64, dVdr::Float64 = ∇V(r, pos[3])
    E .= -1.0*[gradV[2]*cos(phi), gradV[2]*sin(phi), gradV[1]]
end 

""" Calculate radial displacement from trap axis [m]"""
function radial_offset(pos::SVector{3, Float64})::Float64
    return sqrt(pos[1]^2 + pos[2]^2)
end 

""" Get interpolated electric field vector [V/m]"""
function update_E_fixed!(E, pos)
    r = radial_offset(pos)
    if r != 0.0 
        phi = get_azimuthal_angle(pos)
    else 
        phi = 0.0
    end
    dVdz::Float64, dVdr::Float64 = ∇V(r, pos[3])
    E .= -1.0.*[dVdr*cos(phi), dVdr*sin(phi), dVdz]
end 

""" Update interpolated electric field vector [V/m]"""
function update_E!(E, pos, V_sitp)
    r = radial_offset(pos)
    if r != 0.0 
        phi = get_azimuthal_angle(pos)
    else 
        phi = 0.0
    end
    dVdz::Float64, dVdr::Float64 = gradient(V_sitp, pos[3], r)::SVector{2,Float64}  
    E .= -1.0.*[dVdr*cos(phi), dVdr*sin(phi), dVdz]
end 

#### Define functions for friction and diffusion from ion-electron collisions 
""" Calculate Chandrasekhar function"""
function G(x)
    return 0.5*(erf(x) - 2*x/sqrt(pi)*exp(-x^2))/x^2
end

""" Coulomb logarithm for electron-ion scattering from NRL Plasma Formulary 2016"""
function Coulomb_log(n_b; T_b=300., q=e.val)
    return 23. - log(sqrt(n_b*1e-06)*(q/e.val)/(k_B.val*T_b/e.val)^1.5)
end    

""" Calculate characteristic collision frequency 

    ... as defined in CGS units iKunz, Lecture Notes on Irreversible Processes in Plasmas (2021)
"""
function get_ν_0(v; n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val)
    v_b = sqrt(2*k_B.val*T_b/m_b)
    if m_b == m_e.val
        Coul_log = Coulomb_log(n_b, T_b=T_b, q=q)
    else
        throw("Ion-ion collision Coulomb logarithm not implemented yet.") # TODO: Add Coulomb log for ion-ion scattering
    end
    return q^2*q_b^2*n_b*Coul_log/(4*pi*ε_0.val^2*m^2*v_b^3)
end 

""" Calculate slowing down frequency for ions collinding on Maxwellian background species
    
    See Ichimaru, Basic Principles of Plasma Physics - A Statistical Approach (1973)
"""
function get_ν_s(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    v_b = sqrt(2*k_B.val*T_b/m_b)
    return 2*(1 + m/m_b)*v_b/v*G(v/v_b) * get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
end

""" Calculate parallel diffusion coefficient for ions collinding on Maxwellian background species
    
    See Ichimaru, Basic Principles of Plasma Physics - A Statistical Approach (1973)
"""
function get_D_par(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    v_b = sqrt(2*k_B.val*T_b/m_b)
    return 2*v_b^3/v*G(v/v_b) * get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
end

""" Calculate parallel diffusion frequency for ions collinding on Maxwellian background species
    
    See Kunz, Lecture Notes on Irreversible Processes in Plasmas (2021)
"""
function get_ν_par(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    v_b = sqrt(2*k_B.val*T_b/m_b)
    return 2*(v_b/v)^3*G(v/v_b) * get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
end

""" Calculate transverse diffusion coefficient for ions collinding on Maxwellian background species
    
    See Ichimaru, Basic Principles of Plasma Physics - A Statistical Approach (1973)
"""
function get_D_perp(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    v_b = sqrt(2*k_B.val*T_b/m_b)
    return v_b^3/v*(erf(v/v_b) - G(v/v_b)) * get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
end

""" Calculate transverse diffusion frequency for ions collinding on Maxwellian background species
    
    See Kunz, Lecture Notes on Irreversible Processes in Plasmas (2021)
"""
function get_ν_perp(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    v_b = sqrt(2*k_B.val*T_b/m_b)
    return 2*(v_b/v)^3*(erf(v/v_b) - G(v/v_b)) * get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
end

""" Calculate slowing down and diffusion frequencies for ions collinding on Maxwellian background species
    
    See Kunz, Lecture Notes on Irreversible Processes in Plasmas (2021)
"""
function get_all_νs(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    ν0 = get_ν_0(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
    v_b = sqrt(2*k_B.val*T_b/m_b)
    ν_slowing = 2*(1 + m/m_b)*v_b/v*G(v/v_b) * ν0
    ν_par_diffusion = 2*(v_b/v)^3*G(v/v_b) * ν0 
    ν_perp_diffusion = 2*(v_b/v)^3*(erf(v/v_b) - G(v/v_b)) * ν0
    return ν_slowing, ν_par_diffusion, ν_perp_diffusion
end

""" Calculate energy loss frequency for ions collinding on Maxwellian background species
    
    See Kunz, Lecture Notes on Irreversible Processes in Plasmas (2021)
"""
function get_ν_E(v; n_b=1e11*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, q=e.val, m=m_u.val) 
    ν_slowing, ν_par_diffusion, ν_perp_diffusion = get_all_νs(v, n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
    return 2*ν_slowing - ν_par_diffusion - ν_perp_diffusion
end

##### Boris particle pusher with friction and Euler-Maryurama velocity diffusion
const I33 = Matrix{Int}(I, 3, 3) # define identity matrix
""" Boris particle pusher based on advanced implementation from Zentani et al. 2018 

    Extended to include friction and optionally velocity diffusion

    See Zenitani et al., Physics of Plasmas 25, 112110 (2018)
"""
function Boris_push_with_friction(r, u, E, B, dt, dW, norm_dist; q=e.val, m=m_u.val, 
    n_b=1e08*1e06, T_b=300., q_b=-e.val, m_b=m_e.val, velocity_diffusion=true, rng=default_rng()) 
    eps_dt = 0.5*q/m*E*dt 
    ## TODO: Add gamma factors to friction terms (for fully relativistic implementation)
    u_m = u + eps_dt - u*get_ν_s(norm(u), n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)*dt # Eqn. 3 plus friction

    norm_B = norm(B)
    θ = q/m*dt/gamma_from_u(u_m)*norm_B # Eqn. 6
    b = B/norm_B

    u_m_parallel = dot(u_m, b)*b # Eqn. 11
    u_p = u_m_parallel + (u_m - u_m_parallel)*cos(θ) + cross(u_m, b)*sin(θ) # Eqn. 12

    u = u_p + eps_dt - u_p*get_ν_s(norm(u_p), n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)*dt # Eqn. 5 plus friction
    if velocity_diffusion && n_b > 0.0
        # Add Euler Maryuama step
        D_par = get_D_par(norm(u_p), n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
        D_perp = get_D_perp(norm(u_p), n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, q=q, m=m)
        uu_normed = u_m*transpose(u_m)/norm(u_m)^2 
        rand!(rng, norm_dist, dW) 

        u += ( sqrt(D_par)*uu_normed + sqrt(D_perp)*(I33 - uu_normed) )*dW
    end
    r += u/gamma_from_u(u)*dt # Eqn. 1

    return r, u 
end

##### Guiding centre particle pusher
""" Leapfrog guiding center pusher - inspired by Bacchini et al. 2020
    
    Assumes classical motion in uniform magnetic field along z-axis   
    
    See Bacchini et al., 2020 ApJS 251 10
"""
function GC_push(R, v_par_last_half, E, B, dt; q=e.val, m=m_u.val) 
    norm_B = norm(B)
    b = B/norm_B
    v_par_next_half = v_par_last_half + q/m*dot(E,b)*dt 
    v_ExB = cross(E,B)/norm_B^2
    R_next_guess = R + (v_ExB + v_par_next_half*b)*dt
    E_next_guess = E_itp(R_next_guess) 
    R_next = R + (v_ExB + cross(E_next_guess,B)/norm_B^2)/2*dt + v_par_next_half*b*dt 
    
    return R_next, v_par_next_half
end

##### Guiding centre particle pusher with gyrophase tracking
""" Calculate cyclotron frequency [Hz]"""
function get_ω_c(;q=e.val, m=m_u.val, B=1.)
    return q/m*norm(B)
end 

""" Define radial electric (angular) oscillation frequency for guiding centre calculations
    
    See Joseph et al., Physics of Plasmas 28, 042102 (2021)
"""
function get_ω_E(pos; q=e.val, m=m_u.val)    
    r = norm(pos[1:2])
    dEdr = hessian(sitp, pos[3], r)[2,2]
    E_r = -gradient(sitp, pos[3], r)[2]
    ∇E_perp = E_r/r + dEdr 
    return sqrt(ComplexF64(-q*∇E_perp/m))
end

""" Leapfrog guiding center pusher including gyrophase tracking - inspired by Bacchini et al. 2020 
    
    Assumes classical motion in uniform magnetic field along z-axis. 

    See Bacchini et al., 2020 ApJS 251 10
    
    #TODO: Add friction and diffusion
"""
function GC_push_with_gyrophase(R, μ, v_par_last_half, gyrophase, B, dt; q=e.val, m=m_u.val, iterations=1)
    norm_B = norm(B)
    b = B/norm_B
    E = E_itp(R) 
    v_par_next_half = v_par_last_half + q/m*dot(E,b)*dt 
    v_ExB = cross(E,B)/norm_B^2
    E_next = E
    R_next = NaN
    for i in range(1,iterations)
        v_ExB_next = cross(E_next,B)/norm(B)^2
        v_ExB_next_half = (v_ExB + v_ExB_next)/2
        R_next = R + (v_ExB_next_half + v_par_next_half*b)*dt
        E_next = E_itp(R_next) 
    end
    ω_c = get_ω_c(q=q, m=m, B=B)
    ω_E = get_ω_E(R; q=q, m=m)
    ω_p = ω_c/2 + sqrt(ω_c^2/4 - Float64(ω_E^2)/2)
    gyrophase_next = gyrophase - sign(q)*ω_p*dt 
    μ_next = μ

    return R_next, μ_next, v_par_next_half, gyrophase_next
end
