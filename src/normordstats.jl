module OrderStatistics

using StatsFuns

using IntervalArithmetic.Interval
import IntervalArithmetic.interval_from_midpoint_radius
import QuadGK.quadgk

export NormOrderStatistic, moment, expectation, expectationproduct

###############################################################################
#
#   Type aware integration
#
###############################################################################

gkord(n) = 50+ceil(Int, sqrt(n))

function integrate(f::Function, a::T, b::T; kwargs...) where T<:Real
    return quadgk(f, a, b; kwargs...)[1]::T
end

function interval_from_midpoint_radius(mid::T, rad::T) where T<:Interval
    return T(mid.lo - rad.lo, mid.hi+rad.hi)
end

function integrate(f::Function, a::T, b::T; kwargs...) where T<:Interval
    res = quadgk(f, a, b; kwargs...)
    return interval_from_midpoint_radius(res[1], res[2])
end

###############################################################################
#
#   NormOrderStatistic Type
#
###############################################################################

function factorials(n)
    facs = Array{BigInt}(n)
    facs[1] = big(1)
    for i in 2:n
        facs[i] = i*facs[i-1]
    end
    return facs
end

mutable struct NormOrderStatistic{T<:Real} <: AbstractVector{T}
    n::Int
    facs::Vector{BigInt}
    E::Vector{T}

    function NormOrderStatistic{T}(n::Int) where T

        OS = new{T}(n, factorials(n))

        OS.E = zeros(T, OS.n)
        # expected values of order statistics
        # Exact Formulas:
        # N. Balakrishnan, A. Clifford Cohen
        # Order Statistics & Inference: Estimation Methods
        # Section 3.9
        if OS.n == 2
            OS.E[1] = -1/sqrt(T(π))
        elseif OS.n == 3
            OS.E[1] = -3/2sqrt(T(π))
        elseif OS.n == 4
            OS.E[1] = -6/sqrt(T(π)) * 1/T(π)*atan(sqrt(T(2)))
            OS.E[2] = -4*3/2sqrt(T(π)) - 3*OS.E[1] # 4*E(3|3) - 3*E(4|4)
        elseif OS.n == 5
            OS.E[1] = -10/sqrt(T(π)) * ( (3/(2T(π)))*atan(sqrt(T(2))) - 1/4 )
            a = -6/sqrt(T(π)) * 1/T(π)*atan(sqrt(T(2)))
            OS.E[2] = 5*a - 4*OS.E[1] # 5*E(4|4) - 4*E(5|5)
        else
            for i in 1:div(OS.n,2)
                OS.E[i] = moment(OS, i, 1)
            end
        end

        if iseven(OS.n)
            OS.E[div(OS.n,2)+1: end] = -reverse(OS.E[1:div(OS.n,2)])
        else
            OS.E[div(OS.n, 2)+2:end] = -reverse(OS.E[1:div(OS.n,2)])
        end
        return OS
    end
end

NormOrderStatistic(n::Int) = NormOrderStatistic{Float64}(n)

Base.size(OS::NormOrderStatistic) = (OS.n,)
Base.IndexStyle(::Type{NormOrderStatistic{T}}) where T = IndexLinear()
Base.eltype(OS::NormOrderStatistic{T}) where T = T

Base.getindex(OS::NormOrderStatistic{T}, i::Int) where T = OS.E[i]

###############################################################################
#
#   Exact expected values of normal order statistics
#
###############################################################################

I(x, i, n) = exp((i-1)*normlogcdf(x) + (n-i)*normlogccdf(x) + normlogpdf(x))

function moment(OS::NormOrderStatistic{T}, i::Int, pow=1, r::T=T(R)) where T
    C = OS.facs[end]
    if i != 1
        C /= OS.facs[i-1]
    end
    if i != OS.n
        C /= OS.facs[end-i]
    end
    return integrate(x -> x^pow * T(C) * I(x, i, OS.n), -r, r; order=gkord(OS.n))
end

expectation(OS::NormOrderStatistic{T}) where T = OS.E

expectation(OS::NormOrderStatistic, i::Int) = OS[i]

function Base.show(io::IO, OS::NormOrderStatistic{T}) where T
    show(io, "Normal Order Statistics ($T-valued) for $(OS.n)-element samples")
end

###############################################################################
#
#   Poor man's caching
#
###############################################################################

const _cache = Dict{Symbol, Dict{Type, Dict}}()

function getval!(f, returnT::Type, args...)
    sf = Symbol(f)

    if !(haskey(_cache, sf))
        _cache[sf] = Dict{Type, Dict}()
    end

    if !(haskey(_cache[sf], returnT))
        _cache[sf][returnT] = Dict{typeof(args), returnT}()
    end

    if !(haskey(_cache[sf][returnT], args))
        _cache[sf][returnT][args] = f(args...)
    end

    return _cache[sf][returnT][args]
end

###############################################################################
#
#   Exact expected products of normal order statistics
#
# after H.J. Godwin
# Some Low Moments of Order Statistics
# Ann. Math. Statist.
# Volume 20, Number 2 (1949), 279-285.
# doi:10.1214/aoms/1177730036
#
# Radius of integration taken after
# Rudolph S. Parrish
# Computing variances and covariances of normal order statistics
# Communications in Statistics - Simulation and Computation
# Volume 21, 1992 - Issue 1
# doi:10.1080/03610919208813009
#
###############################################################################

const R = 12 # Note: normcdf(-12.0) < 1.8e-33

function α(i::Int, j::Int, r::T=T(R)) where T
    res = integrate(x -> x*exp(i*normlogcdf(x) + j*normlogccdf(x)),
        -r, r; order=gkord(i+j))
    return res
end

function β(i::Int, j::Int, r::T=T(R)) where T
    res = integrate(x -> x^2*exp(i*normlogcdf(x) + j*normlogccdf(x) + normlogpdf(x)),
        -r, r; order=gkord(i+j))
    return res
end

function integrand(x::T, j::Int, r::T) where T
    return integrate(y -> normcdf(y)^j, -r, -x; order=gkord(j))
end

function ψ(i::Int, j::Int, r::T=T(R)) where T
    @time res = integrate(x -> exp(i*normlogcdf(x) + log(integrand(x, j, r))),
        -r,  r; order=gkord(i+j))
    return res
end

function γ(i, j, r::T=T(R)) where T
    res = (
            getval!(α, T, (i,j,r)...) +
          i*getval!(β, T, (i-1,j,r)...) -
            getval!(ψ, T, (i,j,r)...)
          ) / (i*j)
    return ifelse(res > zero(T), res, eps(T))
end

function K(OS::NormOrderStatistic, i::T, j::T) where T<:Integer
    C = OS.facs[end]
    if i != 1
        C /= OS.facs[i-1]
    end
    if j != OS.n
        C /= OS.facs[end-j]
    end
    return C
end

function expectationproduct(OS::NormOrderStatistic{T}, i::Int, j::Int) where T
    if i == j
        return moment(OS, i, 2)
    elseif i > j
        return expectationproduct(OS, j, i)
    elseif i+j > OS.n+1
        return expectationproduct(OS, OS.n-j+1, OS.n-i+1)

    else
        S = zero(T)
        for r in 0:j-i-1
            a = (r>0 ? OS.facs[r] : 1)
            for s in 0:j-i-1-r
                b = (s>0 ? OS.facs[s] : 1)
                c = (j-i-1-r-s>0 ? OS.facs[j-i-1-r-s] : 1)
                C = T(inv(a*b*c))
                S += (-one(T))^(r+s) * C * γ(i+r, OS.n-j+s+1, T(R))
            end
        end
        return S*T(K(OS, i, j))
    end
end

function expectationproduct(OS::NormOrderStatistic)
    return [expectationproduct(OS, i, j) for i in 1:OS.n, j in 1:OS.n]
end

###############################################################################
#
#   Exact variances and covariances
#
###############################################################################

function Base.var(OS::NormOrderStatistic, i::Int)
    return expectationproduct(OS,i,i) - expectation(OS,i)^2
end

function Base.cov(OS::NormOrderStatistic{T}, i::Int, j::Int) where T
    return expectationproduct(OS,i,j) - expectation(OS,i)*expectation(OS,j)
end

function Base.cov(OS::NormOrderStatistic{T}) where T
    V = Array{T}(OS.n, OS.n)
    for j in 1:OS.n
        for i in j:OS.n
            V[i,j] = cov(OS, i, j)
        end
    end
    return Symmetric(V, :L)
end

end # of module OrderStatistics
