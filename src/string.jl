export str2float, float2str
str2float(str) = reinterpret(Float32, resize!(convert(Vector{UInt8}, str), 4))[1]
float2str(int) = strip(String(reinterpret(UInt8, [int])), '\0')
