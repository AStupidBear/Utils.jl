export LabelEncoder, OneHotEncoder, fit_transform, fit, transform, inverse_transform

@with_kw mutable struct LabelEncoder
    unique_label::Array = []
end

Base.length(encoder::LabelEncoder) = length(encoder.unique_label)

function fit_transform(encoder::LabelEncoder, label)
    fit(encoder, label)
    transform(encoder, label)
end

function fit(encoder::LabelEncoder, label)
    encoder.unique_label = unique(label)
    sort!(encoder.unique_label)
end

function transform(encoder::LabelEncoder, label)
    [findfirst(encoder.unique_label, l) - 1 for l in label]
end

function inverse_transform(encoder::LabelEncoder, index)
    [encoder.unique_label[Int(i + 1)] for i in index]
end

export confusmat
function confusmat(ul, y, ypred)
    encoder = LabelEncoder(ul)
    ypred_int = transform(encoder, ypred) + 1
    y_int = transform(encoder, y) + 1
    R = Int[countnz((y_int .== i) .& (ypred_int .== j)) for i in 1:length(ul), j in 1:length(ul)]
    mat = Any["gt/pred" ul'; Any[ul R]]
end

export mat2acc
mat2acc(mat) = sum(diag(mat)) / sum(mat)

"""
```
label = [1, 3, 2]' #["small", "medium", "large"]'
encoder = OneHotEncoder()
vector = fit_transform(encoder, label)
inverse_transform(encoder, vector) == label
```
"""
@with_kw mutable struct OneHotEncoder
    unique_label::Array = []
end

function fit_transform(encoder::OneHotEncoder, label)
    fit(encoder, label)
    transform(encoder, label)
end

function fit(encoder::OneHotEncoder, label)
    encoder.unique_label = unique(label)
    sort!(encoder.unique_label)
end

function transform(encoder::OneHotEncoder, label)
    vector = zeros(length(encoder.unique_label), length(label))
    for j in 1:size(vector, 2)
        i = findfirst(encoder.unique_label, label[j])
        vector[i, j] = 1
    end
    vector
end

function inverse_transform(encoder::OneHotEncoder, vector)
    label = Array{eltype(encoder.unique_label)}(1, size(vector, 2))
    for j = 1:size(vector, 2)
        i = argmax(vector[:, j])
        label[1, j] = encoder.unique_label[i]
    end
    label
end
