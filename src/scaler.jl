export MinMaxScaler, ShapeScaler, ImageScaler, fit_transform, fit, transform, inverse_transform

"""
```julia
scaler = MinMaxScaler()
x = [1 2 3 4; -1 -2 -3 -4]
y = fit_transform(scaler, x)
inverse_transform(scaler, y) == x
```
"""
@with_kw mutable struct MinMaxScaler
    sample_dim::Int = 2
    𝑚ax::Array{Float32} = []
    𝑚in::Array{Float32} = []
end

function fit_transform(scaler::MinMaxScaler, x, shape = (); dim = 2, reshape = false)
    fit(scaler, x; dim = dim)
    transform(scaler, x)
end

function fit(scaler::MinMaxScaler, x, shape = (); dim = 2, reshape = false)
    scaler.sample_dim, scaler.𝑚ax, scaler.𝑚in = dim, maximum(x, 2), minimum(x, 2)
    return scaler
end

transform(scaler::MinMaxScaler, x, shape = (); reshape = false) = (Array{Float32}(x) .- scaler.𝑚in) ./ (scaler.𝑚ax .- scaler.𝑚in .+ 1.0f-20) .- 0.5f0

inverse_transform(scaler::MinMaxScaler, x, shape = (); reshape = false) = (Array{Float32}(x) .+ 0.5f0) .* (scaler.𝑚ax .- scaler.𝑚in + 1.0f-20) .+ scaler.𝑚in

"""
```julia
scaler = ShapeScaler()
x = ([1 2 3 4; -1 -2 -3 -4; 1 2 3 4; 2 2 3 4], )
fit_transform(scaler, x, ((2, 2),))
fit_transform(scaler, x, ((2, 2),); reshape = false)
fit(scaler, x, ((2, 2),))
transform(scaler, x)
```
"""

@with_kw mutable struct ShapeScaler
    scalers::Vector{MinMaxScaler} = Vector{MinMaxScaler}()
    shapes::Tuple = ()
end

function fit_transform(scaler::ShapeScaler, x, shapes; reshape = false)
    fit(scaler, x, shapes)
    transform(scaler, x; reshape = reshape)
end

function fit(scaler::ShapeScaler, x, shapes)
    scaler.shapes = shapes
    for xi in x  push!(scaler.scalers, fit(MinMaxScaler(), xi)) end
end

function transform(scaler::ShapeScaler, x; reshape = false)
    ntuple(length(x)) do i
        xi = transform(scaler.scalers[i], x[i])
        reshape ? Base.reshape(xi, (scaler.shapes[i]..., ccount(xi))) : xi
    end
end


"""
```julia
scaler = ImageScaler()
x = [1 2 3 4; -1 -2 -3 -4; 1 2 3 4; 2 2 3 4]
fit_transform(scaler, x, (2, 2))
fit_transform(scaler, x, (2, 2); reshape = false)
fit(scaler, x, (2, 2))
transform(scaler, x)
```
"""
@with_kw mutable struct ImageScaler
    𝑚ax::Float32 = 1
    shape::NTuple = ()
end

function fit_transform(scaler::ImageScaler, x, shape; reshape = false)
    fit(scaler, x, shape)
    transform(scaler, x; reshape = reshape)
end

fit(scaler::ImageScaler, x, shape) = (scaler.shape = shape; scaler.𝑚ax = maximum(abs, x))

function transform(scaler::ImageScaler, x; reshape = false)
    xs = Array{Float32}(x) / scaler.𝑚ax
    reshape ? Base.reshape(xs, (scaler.shape..., ccount(xs))) : xs
end
