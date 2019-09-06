module FiniteVolumeRDS
using Interpolations: interpolate, Gridded, Linear
using LinearAlgebra
using Serialization
using Mmap

# export public functions
export StaggeredGrid, surface_area, total_volume, depth, unpack, amount
export FiniteVolumeRDSSolution, DenseFiniteVolumeRDSSolution, ThinnedFiniteVolumeRDSSolution, ThinnedMemoryBoundFiniteVolumeRDSSolution, tuples, views, serialize, deserialize
export FiniteVolumeDiscretization, solve
export ReactionDiffusionSystem, update!
export col

#=======================================
Helper function to cycle through indexes
=======================================#
mod1(i,lim) = (i-1)%lim+1
modidx(i,lim) = convert(Int64, 1+trunc((i-1)/lim))


#=======================================
Solution Object
	Basically contains a matrix with the solution for each point in time and space.
	Provides pre-allocated variables desired during computation
	Provides a view to the solution matrix at ti and ti+1
=======================================#
# Abstract Type
abstract type FiniteVolumeRDSSolution end

# indexing and iteration of solution objects
Base.size(sol::FiniteVolumeRDSSolution) = size(sol.sol)
Base.length(sol::FiniteVolumeRDSSolution) = length(sol.t)
Base.getindex(sol::FiniteVolumeRDSSolution, i::Int64) = (sol.t[i], sol.sol[:,:,i])
Base.getindex(sol::FiniteVolumeRDSSolution, I::Vararg{Union{Int,Colon,UnitRange},2}) = sol.sol[:,I[2],I[1]]
Base.firstindex(sol::FiniteVolumeRDSSolution, i::Int64) = 1
Base.lastindex(sol::FiniteVolumeRDSSolution, i::Int64) = begin
															if i == 1
																size(sol.sol)[3]
															elseif i == 2
																size(sol.sol)[2]
															else
																throw(BoundsError(sol, i))
															end
														end
Base.iterate(sol::FiniteVolumeRDSSolution, state=1) = state > length(sol) ? nothing : (sol[state], state+1)
# convert solution to list of tuples
tuples(sol::FiniteVolumeRDSSolution) = [s for s in sol]
# shift view to solution at ti and ti+1
views(sol::FiniteVolumeRDSSolution, ti) = (view(sol.sol, :, :, ti), view(sol.sol, :, :, ti+1))

#========================
Thinned Solution Object
=========================#
struct ThinnedFiniteVolumeRDSSolution <: FiniteVolumeRDSSolution
	# solution type specific aruments
	thinned::Int64
	# actual solution
	t::Array{Float64}
	sol::Array{Float64,3}
	sol_tmp::Array{Float64,3}
	# intermediate preallocated variables
	f::Array{Float64,2}
	Q::Array{Float64,2}
	dl::Array{Float64,1}
	d::Array{Float64,1}
	du::Array{Float64,1}
	Bdl::Array{Float64,1}
	Bd::Array{Float64,1}
	Bdu::Array{Float64,1}
	solio::IOStream
	path::AbstractString
end
ThinnedFiniteVolumeRDSSolution(thinned::Int64, path::AbstractString) = (t::Array{Float64}, dt::Float64, nz::Int64, nvars::Int64) -> ThinnedFiniteVolumeRDSSolution(thinned, t, dt, nz, nvars, path)
function ThinnedFiniteVolumeRDSSolution(thinned::Int64, t::Array{Float64}, dt::Float64, nz::Int64, nvars::Int64, path::AbstractString)
	t_thinned = collect(t[1]:(dt*thinned):t[end])
	solio = open(path, "w+")
	write(solio, t_thinned[1])
	write(solio, t_thinned[end])
	write(solio, dt*thinned)
	write(solio, nz)
	write(solio, nvars)
	sol = Mmap.mmap(solio, Array{Float64,3}, (nz, nvars, length(t_thinned)))
	sol_tmp = zeros(nz, nvars, thinned-1)


	ThinnedFiniteVolumeRDSSolution(
		thinned,
		t_thinned,								# t
		sol,		# sol
		sol_tmp,
		zeros(nz+1,nvars),				# f
		zeros(nz,nvars),				# Q
		zeros(nz*nvars),				# dl
		zeros(nz*nvars),				# d
		zeros(nz*nvars),				# du
		zeros(nz*nvars),				# Bdl
		zeros(nz*nvars),				# Bd
		zeros(nz*nvars),				# Bdu
		solio,
		path
		)
end
function ThinnedFiniteVolumeRDSSolution(path::AbstractString)
    solio = open(path, "r+")
    tstart = read(solio, Float64)
	tend = read(solio, Float64)
	dt = read(solio, Float64)
	t = tstart:dt:tend
	nz = read(solio, Int)
	nvars = read(solio, Int)
	sol = Mmap.mmap(solio, Array{Float64,3}, (nz,nvars,length(t)))
	close(solio)
	ThinnedFiniteVolumeRDSSolution(
							1.0,
							t,								# t
							sol,		# sol
							zeros(0,0,0),
							zeros(nz+1,nvars),				# f
							zeros(nz,nvars),				# Q
							zeros(nz*nvars),				# dl
							zeros(nz*nvars),				# d
							zeros(nz*nvars),				# du
							zeros(nz*nvars),				# Bdl
							zeros(nz*nvars),				# Bd
							zeros(nz*nvars),				# Bdu
							solio,
							path
							)
end
views(sol::ThinnedFiniteVolumeRDSSolution, ti) = begin
    if mod1(ti, sol.thinned) == 1
    	(view(sol.sol, :, :, modidx(ti, sol.thinned)), view(sol.sol_tmp, :, :, 1))
    elseif mod1(ti, sol.thinned) == sol.thinned
    	(view(sol.sol_tmp, :, :, sol.thinned-1), view(sol.sol, :, :, modidx(ti+1, sol.thinned)))
    else
    	(view(sol.sol_tmp, :, :, mod1(ti, sol.thinned)-1), view(sol.sol_tmp, :, :, mod1(ti+1, sol.thinned)-1))
    end
end

#========================
Thinned memory-bound solution
=========================#
struct ThinnedMemoryBoundFiniteVolumeRDSSolution <: FiniteVolumeRDSSolution
	# solution type specific aruments
	thinned::Int64
	# actual solution
	t::Array{Float64}
	sol::Array{Float64,3}
	sol_tmp::Array{Float64,3}
	# intermediate preallocated variables
	f::Array{Float64,2}
	Q::Array{Float64,2}
	dl::Array{Float64,1}
	d::Array{Float64,1}
	du::Array{Float64,1}
	Bdl::Array{Float64,1}
	Bd::Array{Float64,1}
	Bdu::Array{Float64,1}
end
ThinnedMemoryBoundFiniteVolumeRDSSolution(thinned::Int64) = (t::Array{Float64}, dt::Float64, nz::Int64, nvars::Int64) -> ThinnedMemoryBoundFiniteVolumeRDSSolution(thinned, t, dt, nz, nvars)
function ThinnedMemoryBoundFiniteVolumeRDSSolution(thinned::Int64, t::Array{Float64}, dt::Float64, nz::Int64, nvars::Int64)
	t_thinned = collect(t[1]:(dt*thinned):t[end])
	sol::Array{Float64,3} = zeros(nz, nvars, length(t_thinned))
	sol_tmp = zeros(nz, nvars, thinned-1)


	ThinnedMemoryBoundFiniteVolumeRDSSolution(
		thinned,
		t_thinned,								# t
		sol,		# sol
		sol_tmp,
		zeros(nz+1,nvars),				# f
		zeros(nz,nvars),				# Q
		zeros(nz*nvars),				# dl
		zeros(nz*nvars),				# d
		zeros(nz*nvars),				# du
		zeros(nz*nvars),				# Bdl
		zeros(nz*nvars),				# Bd
		zeros(nz*nvars),				# Bdu
		)
end
views(sol::ThinnedMemoryBoundFiniteVolumeRDSSolution, ti) = begin
    if mod1(ti, sol.thinned) == 1
    	(view(sol.sol, :, :, modidx(ti, sol.thinned)), view(sol.sol_tmp, :, :, 1))
    elseif mod1(ti, sol.thinned) == sol.thinned
    	(view(sol.sol_tmp, :, :, sol.thinned-1), view(sol.sol, :, :, modidx(ti+1, sol.thinned)))
    else
    	(view(sol.sol_tmp, :, :, mod1(ti, sol.thinned)-1), view(sol.sol_tmp, :, :, mod1(ti+1, sol.thinned)-1))
    end
end

#========================
Dense Solution Object
=========================#
struct DenseFiniteVolumeRDSSolution <: FiniteVolumeRDSSolution
	# actual solution
	t::Array{Float64}
	sol::Array{Float64,3}
	# intermediate preallocated variables
	f::Array{Float64,2}
	Q::Array{Float64,2}
	dl::Array{Float64,1}
	d::Array{Float64,1}
	du::Array{Float64,1}
	Bdl::Array{Float64,1}
	Bd::Array{Float64,1}
	Bdu::Array{Float64,1}
	solio::IOStream
	path::AbstractString
end

function DenseFiniteVolumeRDSSolution(t::Array{Float64}, dt::Float64, nz::Int64, nvars::Int64, path::AbstractString)
	solio = open(path, "w+")
	write(solio, t[1])
	write(solio, t[end])
	write(solio, dt)
	write(solio, nz)
	write(solio, nvars)
	sol = Mmap.mmap(solio, Array{Float64,3}, (nz, nvars, length(t)))


	DenseFiniteVolumeRDSSolution(t,								# t
	sol,		# sol
	zeros(nz+1,nvars),				# f
	zeros(nz,nvars),				# Q
	zeros(nz*nvars),				# dl
	zeros(nz*nvars),				# d
	zeros(nz*nvars),				# du
	zeros(nz*nvars),				# Bdl
	zeros(nz*nvars),				# Bd
	zeros(nz*nvars),				# Bdu
	solio,
	path
	)
end

(::Type{DenseFiniteVolumeRDSSolution})(path::AbstractString) =	begin
    solio = open(path, "r+")
    tstart = read(solio, Float64)
	tend = read(solio, Float64)
	dt = read(solio, Float64)
	t = tstart:dt:tend
	nz = read(solio, Int)
	nvars = read(solio, Int)
	sol = Mmap.mmap(solio, Array{Float64,3}, (nz,nvars,length(t)))
	close(solio)
	DenseFiniteVolumeRDSSolution(t,								# t
							sol,		# sol
							zeros(nz+1,nvars),				# f
							zeros(nz,nvars),				# Q
							zeros(nz*nvars),				# dl
							zeros(nz*nvars),				# d
							zeros(nz*nvars),				# du
							zeros(nz*nvars),				# Bdl
							zeros(nz*nvars),				# Bd
							zeros(nz*nvars),				# Bdu
							solio,
							path
							)
end

save(sol::Union{DenseFiniteVolumeRDSSolution, ThinnedFiniteVolumeRDSSolution}) = 	begin
												if sol.solio != nothing
													Mmap.sync!(sol.sol)
													close(sol.solio)
													#finalize(sol.sol)
												end
											end

#========================
Interface to reaction diffusion system
=========================#
abstract type ReactionDiffusionSystem end
update!(::ReactionDiffusionSystem)=error("not implemented")

#========================
Staggered Grid
=========================#
col(a::Array{T,1}) where T <: Any = a[:,:]
#flatten(a::AbstractArray{Float64,2}) = reshape(a, (length(a), 1))
#flatten(a::Array{Array{T,1},1}) where T <: Any = vcat(a...)
#flatten(a::Array{Array{T,2},1}) where T <: Any = hcat(a...)

struct StaggeredGrid
	z_start::Float64
	z_end::Float64
	nz::Int64
	z_faces::Array{Float64,2}
	z_centres::Array{Float64,2}
	h_centres::Array{Float64,2}
	h_faces::Array{Float64,2}
	a_faces::Array{Float64,2}
	a_centres::Array{Float64,2}
	volumes::Array{Float64,2}
	function StaggeredGrid(z_start::Float64, z_end::Float64, nz::Int64, z_areas::Array{Float64,1}, areas::Array{Float64,1})
		z_faces = col(collect(range(z_start, stop=z_end, length=nz+1)))
		z_centres = (z_faces[1:end-1,:]+z_faces[2:end,:])/2.0
		h_centres = z_faces[2:end,:]-z_faces[1:end-1,:]
		h_faces = z_centres[2:end,:]-z_centres[1:end-1,:]
		a_faces = col(interpolate((z_areas,), areas, Gridded(Linear()))(z_faces[:,1]))
		a_centres = (a_faces[1:end-1,:]+a_faces[2:end,:])/2.0
		volumes = a_centres.*h_centres
		new(
			z_start,
			z_end,
			nz,
			z_faces,
			z_centres,
			h_centres,
			h_faces,
			a_faces,
			a_centres,
			volumes
			)
	end
end

surface_area(g::StaggeredGrid) = g.a_faces[1]
depth(g::StaggeredGrid) = g.z_end - g.z_start
total_volume(g::StaggeredGrid) = sum(g.volumes)
unpack(g::StaggeredGrid) = (g.z_start, g.z_end, g.nz,
							g.z_centres, g.z_faces,
							g.h_centres, g.h_faces,
							g.a_centres, g.a_faces)
amount(g::StaggeredGrid, concentrations::AbstractArray) = sum(g.volumes.*concentrations)


#========================
Finite Volume Discretization
=========================#
struct FiniteVolumeDiscretization
	grid::StaggeredGrid
	n_vars::Int64
	nz::Int64
	dt::Float64
	form1::Array{Float64,2}
	form2::Array{Float64,2}
	dim::Int64

	function FiniteVolumeDiscretization(grid::StaggeredGrid, n_vars::Int64, dt::Float64)
		z_0, z_end, nz, z_centres, z_faces, h_centres, h_faces, a_centres, a_faces = unpack(grid)

		# constants used to build the tridiagonals
		form = -4.0*dt*a_faces[2:end-1,:]./(h_centres[2:end,:]+h_centres[1:end-1,:])
		form1 = [0;			# no-flux boundary
				form./(a_faces[3:end,:]+a_faces[2:end-1,:])./h_centres[2:end,:]]
		form2 = [form./(a_faces[2:end-1,:]+a_faces[1:end-2,:])./h_centres[1:end-1,:];
				0]			# no-flux boundary
		form1 = repeat(form1, outer=[n_vars,1])
		form2 = repeat(form2, outer=[n_vars,1])
		dim = nz*n_vars
		new(grid, n_vars, nz, dt, form1, form2, dim)
	end
end

function solve(fvd::FiniteVolumeDiscretization, u0::Array{Float64,2}, tstart::Float64, tend::Float64, rds::ReactionDiffusionSystem; repetitions::Int=1, solutiontype::Union{DataType,Function}=DenseFiniteVolumeRDSSolution)
	# time vector without repetitions
	t = collect(tstart:fvd.dt:tend)
	plainlen=length(t)
	# time vector with repetitions
	tend_rep = t[1]+(t[end]-t[1])*repetitions
	t = collect(tstart:fvd.dt:tend_rep)
	replen=length(t)
	# initialize solution
	sol = solutiontype(t, fvd.dt, fvd.nz, fvd.n_vars)
	diffusivities::Array{Float64,2} = zeros(fvd.nz+1, fvd.n_vars)
	fluxes::Array{Float64,2} = zeros(fvd.nz+1, fvd.n_vars)
	sources::Array{Float64,2} = zeros(fvd.nz, fvd.n_vars)
	# set initial values
	(state_n, state_n1) = views(sol, 1)
	state_n[:] .= u0[:]
	# step forward in time
	irep::Int64=0
	for i in 1:(replen-1)
		# prepare step
		irep=mod1(i,plainlen)
		(state_n, state_n1) = views(sol, i)
		update!(rds, state_n, t[i], i, t[irep], irep, diffusivities, fluxes, sources)
		step!(fvd, sol, i, diffusivities, fluxes, sources)
		# don't allow negative values
		state_n1[state_n1.<0.0].=0.0
	end
	sol
end

function step!(fvd::FiniteVolumeDiscretization, sol::FiniteVolumeRDSSolution, tidx::Int64, diffusivities::Array{Float64,2}, fluxes::Array{Float64,2}, sources::Array{Float64,2}) 
	(state_n, state_n1) = views(sol, tidx)
	sol.f[:,:] .= fluxes.*fvd.grid.a_faces
	sol.Q[:,:] .= @views (sol.f[1:end-1,:].-sol.f[2:end,:])./fvd.grid.volumes

	D1 = view(diffusivities, 1:fvd.nz, :)
	D2 = view(diffusivities, 2:fvd.nz+1, :)
	@inbounds @simd for i in 1:fvd.dim
		sol.dl[i] = 0.5*fvd.form1[i]*D1[i]
		sol.du[i] = 0.5*fvd.form2[i]*D2[i]
		sol.d[i] = 1.0-sol.du[i]-sol.dl[i]
		sol.Bd[i] = 2.0-sol.d[i]
		sol.Bdl[i] = -sol.dl[i]
		sol.Bdu[i] = -sol.du[i]
	end
	tridiag_mul!(sol.Bdl, sol.Bd, sol.Bdu, state_n, state_n1, fvd.dim)
	@inbounds @simd for i in 1:fvd.dim
		state_n1[i] = fvd.dt*(sources[i]+sol.Q[i])+state_n1[i]
	end
	solve_tridiag!(sol.dl, sol.d, sol.du, state_n1, fvd.dim)
	nothing
end

# multiplication of tridiagonal matrix with column vector
function tridiag_mul!(dl,d,du,vec,out,N)
	out[1] = vec[1]*d[1]+vec[2]*du[1]
	@inbounds @simd for i in 2:N-1
		out[i] = vec[i-1]*dl[i]+vec[i]*d[i]+vec[i+1]*du[i]
	end
	out[N] = vec[N-1]*dl[N]+vec[N]*d[N]
end

# Thomas-Algorythm to solve tridiagonal system
function solve_tridiag!(dl,d,du,rhs,N)
	du[1] = du[1]/d[1]
	rhs[1] = rhs[1]/d[1]
	@inbounds for i in 2:N
		du[i] = du[i]/(d[i]-du[i-1]*dl[i])
		rhs[i] = (rhs[i]-rhs[i-1]*dl[i])/(d[i]-du[i-1]*dl[i])
	end
	@inbounds for i in (N-1):-1:1
		rhs[i] = rhs[i]-du[i]*rhs[i+1]
	end
	nothing
end

# Testcase
function generate_problem()
	grid = StaggeredGrid(0.0,4.0,500,collect(range(0.0, stop=4.0, length=500+1)), collect(range(8.0, stop=2.0, length=500+1)))
	fvd = FiniteVolumeDiscretization(grid,6,0.01)
	diffusivities0 = repeat([0.02], outer=[501,6])
	sources0 = repeat([0.0], outer=[500,6])
	sources0[250,:] = repeat([5.0], outer=[6])
	fluxes0 = repeat([0.0], outer=[501,6])
	rds = ReactionDiffusionSystem((u,t,diffusivities) -> diffusivities[:] .= diffusivities0[:],
								  (u,t,fluxes) -> fluxes[:] .= fluxes0[:],
								  (u,t,sources) -> sources[:] .= sources0[:])
	u0 = repeat(col(collect(range(0.0, stop=5.0, length=500))), outer=[1,6])
	return (fvd, u0, rds)
end

end
