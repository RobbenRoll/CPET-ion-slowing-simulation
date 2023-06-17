using Profile
using ProfileView

function profile_test(n)
    for i = 1:n
        A = randn(100,100,20)
        m = maximum(A)
        Am = mapslices(sum, A; dims=2)
        B = A[:,:,5]
        Bsort = mapslices(sort, B; dims=1)
        b = rand(100)
        C = B.*b
    end
end

# # compilation
# ProfileView.@profview profile_test(1)
# # pure runtime
# ProfileView.@profview profile_test(10)


include("../CoolingSimulation.jl")
# Global parameters
const B = [0.,0.,7.]
# Define ion initial conditions
q = e.val
m = 23*m_u.val
N_ions = 100
μ_z0 = -0.125
σ_z0 = 0.005
σ_xy0 = 0.001
μ_E0_par, σ_E0_par = 80., 1.
σ_E0_perp = 0.5

# Define plasma parameters
n_b = 1e08*1e06
T_b = 300.
q_b = -e.val 
m_b = m_e.val

# Define residual gas parameters
neutral_masses = [28*m_u.val]
neutral_pressures_mbar = [5e-09]
alphas = [alpha_N2]
CX_fractions = [0.0] 
T_n = 300. 

# Define run parameters
t_end = 370.0e-06
dt = 1e-08
sample_every = 1000
velocity_diffusion = true
seed = 96732607

r = [0.001, 0, 0]
u_last_half = [0.01, 0.01, 100]
times = 0:dt:t_end
# compilation
integrate_orbit_with_friction(times, r, u_last_half; q=q, m=m, B=B, 
                                                    n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                    neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                                    alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                                    dt=dt, sample_every=sample_every, velocity_diffusion=velocity_diffusion, seed=seed)
Profile.clear_malloc_data()
# pure runtime
#ProfileView.@profview integrate_orbit_with_friction(times, r, u_last_half; q=q, m=m, B=B, 
integrate_orbit_with_friction(times, r, u_last_half; q=q, m=m, B=B, 
                                                    n_b=n_b, T_b=T_b, q_b=q_b, m_b=m_b, 
                                                    neutral_masses=neutral_masses, neutral_pressures_mbar=neutral_pressures_mbar, 
                                                    alphas=alphas, CX_fractions=CX_fractions, T_n=T_n, 
                                                    dt=dt, sample_every=sample_every, velocity_diffusion=velocity_diffusion=seed)

