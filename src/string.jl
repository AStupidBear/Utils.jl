export str2float, float2str
str2float(str) = reinterpret(Float32, resize!(unsafe_wrap(Vector{UInt8}, str), 4))[1]
float2str(flt) = strip(String(reinterpret(UInt8, Float32[flt])), '\0')
