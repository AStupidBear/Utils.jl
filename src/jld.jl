using JLD

export jldwrite
function jldwrite(filename, name::String, data)
    fid = jldopen(filename, "r+")
    try
        write(fid, name, data)
    finally
        close(fid)
    end
end

export jldread
function jldread(filename, name::String)
    local dat
    fid = jldopen(filename, "r")
    try
        obj = fid[name]
        dat = read(obj)
    finally
        close(fid)
    end
    dat
end