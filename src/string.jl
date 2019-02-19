export str2float, float2str
str2float(str) = reinterpret(Float32, resize!(unsafe_wrap(Vector{UInt8}, str), 4))[1]
float2str(flt) = modf(flt)[1] == 0 && flt < 7f5 ? string(Int(flt)) : 
                strip(String(reinterpret(UInt8, Float32[flt])), '\0')
