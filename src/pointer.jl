function offsetof(type_, member::Symbol)
  for (i, item) in enumerate(fieldnames(type_))
    if item == member
      return fieldoffset(type_, i)
    end
    #print(typeof(i))
  end
  # what to do when symbol not in type_?
  throw("$type_ has no member named $member")
end

function GetStructType(type_, member::Symbol)
  for (i, item) in enumerate(fieldnames(type_))
    if item == member
      return fieldtype(type_, i)
    end
    #print(typeof(i))
  end
  # what to do when symbol not in type_?
  throw("$type_ has no member named $member")
end

function Base.getindex(ptr::Ptr{T}, s::Symbol) where {T}
  address = UInt(ptr)
  if address == 0
    throw("Base.getindex(Ptr::{$T}) would dereference a NULL pointer")
  end
  offset = offsetof(T, s)
  fieldtype = GetStructType(T, s)
  fieldptr = Ptr{fieldtype}(address + offset)
  #log("Symbol $s $ptrtype address=$address offset=$offset fieldtype=$fieldtype ptr=$ptr fieldptr=$fieldptr\n")
  #return 123
  return unsafe_load(fieldptr)
end

function Base.setindex!(ptr::Ptr{T}, value, s::Symbol) where {T}
  address = UInt(ptr)
  if address == 0
    throw("Base.setindex!(Ptr) would write to a NULL pointer")
  end
  offset = offsetof(T, s)
  fieldtype = GetStructType(T, s)
  fieldptr = Ptr{fieldtype}(address + offset)
  #log("Symbol $s $ptrtype address=$address offset=$offset fieldtype=$fieldtype ptr=$ptr fieldptr=$fieldptr\n")
  unsafe_store!(fieldptr, value)
  return value
end