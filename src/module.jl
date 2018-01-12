export import_module
"""
```julia
module foo
bar(x) = x
end
import_module(foo, Main)
```
"""
function import_module(m1, m2 = Main)
    for name in names(m1, true)
        if string(name)[1] != '#'
            ex = :($name = getfield($m1, Symbol($(string(name)))))
            eval(m2, ex)
        end
    end
end
