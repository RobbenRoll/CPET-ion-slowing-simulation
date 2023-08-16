using Surrogates
using HDF5
using LinearAlgebra
using Statistics

 
fname = "/OutputFiles/2023-08-14_optimization_results.h5"
path = string(@__DIR__) * fname 
noise_variance = 1.23^2 #TODO: LOAD FROM optimization_results.h5 file

# Load sample data
fid = h5open(path, "r+")

initial_samples = fid["initial_samples"]
x = read(initial_samples["inputs"])
y = read(initial_samples["combined_loss_vals"])

optimizer_pars = fid["optimizer_pars"]
p = read(optimizer_pars["p"])
theta = read(optimizer_pars["theta"])
lower_bounds = read(optimizer_pars["lower_bounds"])
upper_bounds = read(optimizer_pars["upper_bounds"])

close(fid)
N_dims = length(x[1])

println("Loaded sample data from " * path)
#println("Samples:")
#println(x)
#println("Loss values:")
#println(y)


# Build surrogate
surrogate = Kriging(x, y, lower_bounds, upper_bounds, p=p, theta=theta, noise_variance=noise_variance)
println(Surrogates.kriging_log_likelihood(x, y, p, theta, noise_variance))
#ys = surrogate.(xs)
#println(ys)
#println(1e-6 * norm(maximum(surrogate.y) - minimum(surrogate.y)))



function log_likelihood(hyper_pars; x=x, y=y, noise_variance=noise_variance)
    p = hyper_pars[1:5]
    theta = hyper_pars[5:10]
    #noise_vairance = hyper_pars[11]
    return Surrogates.kriging_log_likelihood(x, y, p, theta, noise_variance)
end

p = [0.5 for _ in range(1,N_dims)]
theta = [0.5 / max(1e-06 * norm(upper_bounds .- lower_bounds), std(x_i[i] for x_i in x))^p[i] for i in 1:N_dims] # default from Kriging.jl
println(theta)
#hyper_pars = append!(p, theta)
#println(log_likelihood(hyper_pars))

# Optimize kriging hyperparameters
using BlackBoxOptim
p_bounds = [(0.5,1.99) for _ in range(1,N_dims)]
theta_bounds = [(10.0, 100000.0) for i in range(1,N_dims)]
ranges = append!(p_bounds, theta_bounds)
#push!(ranges, (0.1,10.0))
initial_guess = append!(p, theta)
#push!(initial_guess, 1.23^2)
println(initial_guess)
println(log_likelihood(initial_guess))
println(ranges)
res = bboptimize(log_likelihood, initial_guess; SearchRange = ranges, NumDimensions = 2*N_dims,# + 1, 
                 MaxFuncEvals = 1000, TraceInterval = 0.1)
best_hyper_pars = best_candidate(res)
p = best_hyper_pars[1:N_dims]
theta = best_hyper_pars[N_dims+1:2*N_dims]