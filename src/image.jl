function imconvert(ext1, ext2)
    for (root, dirs, files) in walkdir(pwd())
        for file in files
            name, ext = splitext(joinpath(root, file))
            if ext == ext1
                name1 = name * ext1
                name2 = name * ext2
                run(`imconvert $name1 $name2`)
            end
        end
    end
end

export mat2img, img2mat, vec2rgb

function mat2img(A)
    @eval using Images
    colorview(RGB, permutedims(A, (3, 1, 2)))
end

function img2mat(A)
    @eval using Images
    @> A channelview permutedims((2, 3, 1))
end

vec2rgb(x) = (W = Int(√(length(x) ÷ 3)); reshape(x, W, W, 3))
