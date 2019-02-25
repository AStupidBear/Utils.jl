export str2float, float2str, double2str, str2double
str2float(str) = reinterpret(Float32, resize!(unsafe_wrap(Vector{UInt8}, str), 4))[1]
float2str(flt) = modf(flt)[1] == 0 && flt < 7f5 ? string(Int(flt)) : 
                strip(String(reinterpret(UInt8, Float32[flt])), '\0')
str2double(str) = reinterpret(Float64, resize!(unsafe_wrap(Vector{UInt8}, str), 8))[1]
double2str(flt) = modf(flt)[1] == 0 && flt < 7f5 ? string(Int(flt)) : 
                strip(String(reinterpret(UInt8, Float64[flt])), '\0')
