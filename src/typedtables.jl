using TypedTables

import TypedTables: @Table, @Row

macro Table(exprs...)
    N = length(exprs)
    names = Vector{Any}(N)
    values = Vector{Any}(N)
    for i = 1:N
        expr = exprs[i]
        if isa(expr, Symbol)
            names[i] = expr
            values[i] = esc(expr)
        elseif isa(expr.args[1],Symbol)
            names[i] = (expr.args[1])
            values[i] = esc(expr.args[2])
        elseif isa(expr.args[1],Expr)
            if expr.args[1].head != :(::) || length(expr.args[1].args) != 2
                error("A Expecting expression like @Table(name1::Type1 = value1, name2::Type2 = value2) or @Table(name1 = value1, name2 = value2)")
            end
            names[i] = (expr.args[1].args[1])
            values[i] = esc(Expr(:call, :convert, expr.args[1].args[2], expr.args[2]))
        else
            error("A Expecting expression like @Table(name1::Type1 = value1, name2::Type2 = value2) or @Table(name1 = value1, name2 = value2)")
        end
    end
    tabletype = TypedTables.Table{(names...)}
    return Expr(:call, tabletype, Expr(:tuple, values...))
end

macro Row(exprs...)
    N = length(exprs)
    names = Vector{Any}(N)
    values = Vector{Any}(N)
    for i = 1:N
        expr = exprs[i]
        if isa(expr, Symbol)
            names[i] = expr
            values[i] = esc(expr)
        elseif isa(expr.args[1],Symbol)
            names[i] = (expr.args[1])
            values[i] = esc(expr.args[2])
        elseif isa(expr.args[1],Expr)
            if expr.args[1].head != :(::) || length(expr.args[1].args) != 2
                error("A Expecting expression like @Row(name1::Type1 = value1, name2::Type2 = value2) or @Row(name1 = value1, name2 = value2)")
            end
            names[i] = (expr.args[1].args[1])
            values[i] = esc(Expr(:call, :convert, expr.args[1].args[2], expr.args[2]))
        else
            error("A Expecting expression like @Row(name1::Type1 = value1, name2::Type2 = value2) or @Row(name1 = value1, name2 = value2)")
        end
    end
    rowtype = TypedTables.Row{(names...)}
    return Expr(:call, rowtype, Expr(:tuple, values...))
end

DataFrames.DataFrame{Names, StorageTypes}(tbl::TypedTables.Table{Names, StorageTypes}) = DataFrame(collect(tbl.data), collect(Names))

TypedTables.Table(df::DataFrame) = Table(DataFrames.columns(df), names(df))

function TypedTables.Table(column_eltypes::Vector{DataType}, names::Vector{Symbol})
    column_types =Tuple{[Vector{typ} for typ in column_eltypes]...}
    Table{tuple(names...), column_types}()
end

function TypedTables.Table(columns::Vector, names::Vector{Symbol})
    Table{tuple(names...)}(tuple(columns...))
end

function Base.push!(tbl::TypedTables.Table, row)
    for (col, val) in zip(tbl.data, row)
        push!(col, val)
    end
end

Base.Matrix(tbl::TypedTables.Table) = hcat(tbl.data...)

Base.writedlm(f::AbstractString, tbl::TypedTables.Table, delim = ','; opts...) = writedlm(f, Matrix(tbl), delim; opts...)
