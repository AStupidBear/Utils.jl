export capitalize
capitalize = uppercase

Base.find(s::String, c::Union{Char, Vector{Char}, String, Regex}, start = 1) = search(s, c, start)
