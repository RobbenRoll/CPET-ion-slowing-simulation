
# Check for collision event and determine collision partner
using Random
import PhysicalConstants.CODATA2018: c_0, ε_0, m_e, e, m_u, k_B, ħ
import Distributions: Normal
import ArbNumerics: ArbFloat, elliptic_k

# Polarizabilities from https://cccbdb.nist.gov/pollistx.asp
alpha_H2 = 0.787*1e-30 # [4*pi*eps0*m**3]
alpha_H2O = 1.501*1e-30 # [4*pi*eps0*m**3]
alpha_N2 = 1.71*1e-30 # [4*pi*eps0*m**3]
alpha_CO = 1.953*1e-30 # [4*pi*eps0*m**3]
alpha_CO2 = 2.507*1e-30 # [4*pi*eps0*m**3]

function rotation(α,β)
    """3x3 matrix for rotation by Euler angle β around the z-axis and α around the y-axis"""
    return [[cos(α)*cos(β), -sin(α), sin(α)*cos(β)] [cos(α)*sin(β), cos(α), sin(α)*sin(β)] [-sin(α), 0., cos(α)]] 
end 

function get_b_crit(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300)
    """Critical impact parameter for Langevin collision [m]
    
    """
    v_rel = sqrt(norm(v_i)^2 + 3*k_B.val*T_n/m_n) # TODO: Verify
    m_red = m_i*m_n/(m_i + m_n)
    E_coll = 0.5*m_red*v_rel^2 # COM coll. energy [J]
    C4 = alpha*q^2/(4*π*ε_0.val)
    return (2*C4/E_coll)^0.25
end

function R4_scat_angle(b, v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300)
    """Elastic scattering angle [rad] in COM for attractive 1/r**4 scattering potential
        
    Only for b > b_crit!

    v_i : ion velocity in lab frame [eV]
    """
    bb = b/get_b_crit(v_i, q, m_i; m_n=m_n, alpha=alpha, T_n=T_n)
    arg = ArbFloat(2*bb^4 - 2*bb^2*sqrt(bb^4 -1.) -1., digits=30)
    return π - 2*bb*sqrt(2*(bb^2 - sqrt(bb^4 - 1.)))*Float64(elliptic_k(arg))
end

function Langevin_cross_section(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300.)
    """Langevin collision cross section (m^2)"""
    return π*get_b_crit(v_i, q, m_i, m_n=m_n, alpha=alpha, T_n=T_n)^2
end

function get_eff_collision_speed(v_i; m_n=2*m_u.val, T_n=300.)
    return sqrt(norm(v_i)^2 + 3*k_B.val*T_n/m_n) 
end

function get_SC_cross_section(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300.)
    """Semiclassical cross section for elastic scattering and charge exchange [m^2]

    See Mahdian2021
    """
    v_rel = get_eff_collision_speed(v_i, m_n=m_n, T_n=T_n)
    m_red = m_i*m_n/(m_i + m_n)
    E_coll = 0.5*m_red*v_rel^2 # COM coll. energy [J]
    C4 = alpha*q^2/(4*π*ε_0.val)
    return π*(m_red*C4^2/ħ.val^2)^(1/3)*(1 + π^2/16)*E_coll^(-1/3)
end

function get_b_max(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300.)
    """Largest relevant impact parameter for semiclassical scattering [m]"""
    Xsec = get_SC_cross_section(v_i, q, m_i, m_n=m_n, alpha=alpha, T_n=T_n)
    return sqrt(Xsec/π)
end

function rand_b(b_max)
    """Draw random impact parameter [m]"""
    u = rand(Float64)
    return b_max*sqrt(u)
end

function ion_neutral_collision(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, T_n=300., CX_frac=0.)
    """Simulate Monte Carlo ion-neutral collision
    
    CX_fraction : Float64   
        Fraction of glanzing collisions that results in charge exchange rather than elastic scattering.
    """
    ### Generate random neutral velocity vector in lab frame (sampled from Maxwellian)
    v_n =  rand(Normal(0, sqrt(k_B.val*T_n/m_n)), 3) # sample velocity from Maxwellian distribution

    ### Transform velocity vectors into centre-of-momentum (COM) frame (see ShortMSc2018 for equations)
    v_COM = (m_i*v_i .+ m_n*v_n)./(m_i + m_n) # velocity of COM in lab frame
    w_i = v_i - v_COM # ion velocity in COM frame

    # Calculate maximal impact parameter

    ### Draw random impact parameter
    b_crit = get_b_crit(v_i, q, m_i, m_n=m_n, alpha=alpha, T_n=T_n)
    b_max = get_b_max(v_i, q, m_i, m_n=m_n, alpha=alpha, T_n=T_n)
    b = rand_b(b_max)

    ### Select the collision type (elastic or Langevin)
    if b < b_crit 
        θ = π*rand(Float64) # polar scattering angle in COM 
        coll_type = "Langevin"
    else
        if rand(Float64) <= CX_frac
            coll_type = "CX"
        else
            coll_type = "glanzing"
        end
        θ = R4_scat_angle(b, v_i, q, m_i; m_n=m_n, alpha=alpha, T_n=T_n)
    end
    ### Determine COM ion velocity after collision
    #θ =     # polar scattering angle in COM 
    ϕ = 2*π*rand(Float64) # azimuthal scattering angle in COM frame
    w_i_final = norm(w_i)*[cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)] # ion velocity in COM frame after collision 
    if coll_type == "CX" 
        # Handle CX (set final ion to final atom velocity) # TODO: VERIFY
        w_i_final *= -1 # exchange velocity of ion for atom in COM frame
    end

    ### Transform final ion velocity from COM back into the lab frame (see ShortMSc2018 for equations)
    α = acos(w_i[1]/norm(w_i)) 
    β = atan(w_i[2]/w_i[1]) 
    v_i_final = rotation(α, β)*w_i_final + v_COM # rotate final ion velocity vector and boost into lab frame

    return v_i_final, coll_type
end

function get_mean_free_path(v_i, q, m_i; m_n=2*m_u.val, alpha=alpha_H2, p_n_mbar=1e-08, T_n=300.)
    """Get mean free path for semiclassical ion-neutral collision [m]"""
    Xsec = get_SC_cross_section(v_i, q, m_i; m_n=m_n, alpha=alpha, T_n=T_n) 
    n_n = p_n_mbar*1e02/(k_B.val*T_n) # neutral density [1/m^3]
    return 1/(n_n*Xsec)
end