export capitalize
capitalize = uppercase

@static if VERSION < v"1.0"
Base.find(s::String, c::Union{Char, Vector{Char}, String, Regex}, start = 1) = search(s, c, start)
end
