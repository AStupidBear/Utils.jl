"""
```julia
name = "/tmp/tmp1.txt"
attach = ["/tmp/tmp1.txt","/tmp/tmp2.txt"]
mail(name)
mail(name, attach)
```
"""
function mail(name)
    spawn(pipeline(`cat $name`,`mail -s "Computation Results" luyaocns@gmail.com`))
end

function mail(name, attach)
    spawn(pipeline(`cat $name`,`mail -s "Computation Results" --attach=$attach luyaocns@gmail.com`))
end
