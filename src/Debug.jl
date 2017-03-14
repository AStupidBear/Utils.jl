module Debug

import Atom, Juno, Gallium

function bp(file, lines...)
  for line in lines
    Juno.breakpoint(file, line)
  end
end

function clearbp()
  for (~, set) in Gallium.bps_at_location
    for (bp, ~) in set.dict
      Gallium.disable(bp)
    end
  end
end

function clear()
  fail = true
  while fail
    try
      clearbp()
      fail = false
    end
  end
end

export @debug
"""
    Debug.@debug inaloop(2) 4 5
    Debug.@debug inaloop(2) "test.jl" 4 5
"""
macro debug(fun, lines...)
  ex = quote
    Debug.bp(@__FILE__(), $lines...)
    $fun
    Debug.clear()
  end
  @show ex
  esc(ex)
end
macro debug(fun, file::AbstractString, lines...)
  ex = quote
    Debug.bp($file, $lines...)
    $fun
    Debug.clear()
  end
  @show ex
  esc(ex)
end
macro debug(file::AbstractString, lines...)
  ex = :(Debug.bp($file, $lines...))
  @show ex
  esc(ex)
end

foo() = nothing
Juno.breakpoint(foo, )

end
