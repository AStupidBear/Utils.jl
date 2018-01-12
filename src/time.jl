export date, hour, minute, second, msecond
date() = Dates.format(now(), "yyyy-mm-dd")
hour() = Dates.format(now(), "yyyy-mm-dd_HH")
minute() = Dates.format(now(), "yyyy-mm-dd_HH-MM")
second() = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
msecond() = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-s")

export timename
timename(fn) = joinpath(tempdir(), fn * "_" * Dates.format(now(), "yyyy-mm-dd_HH-MM-SS-s"))
