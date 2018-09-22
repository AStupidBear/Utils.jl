using Compat.Dates: format, unix2datetime, datetime2unix, Date, Time

export unix2date, unix2time, unix2datetime, datetime2unix, unix2intstr, unix2int

unix2date(t) = Date(unix2datetime(t))
unix2time(t) = Time(unix2datetime(t))
unix2intstr(t) = format(unix2datetime(t), "yyyymmdd")
unix2int(t) = parse(Int, unix2intstr(t))