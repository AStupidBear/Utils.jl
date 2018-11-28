using Compat.Dates: format, unix2datetime, datetime2unix, Date, Time

export unix2datetime, datetime2unix
export unix2date, unix2time
export unix2str, unix2str8, unix2str6, str2unix
export unix2int, int2unix

unix2date(t) = Date(unix2datetime(t))
unix2time(t) = Time(unix2datetime(t))

unix2str8(t) = format(unix2datetime(t), "yyyymmdd")
unix2str6(t) = format(unix2datetime(t), "yymmdd")
const unix2str = unix2str8
str2unix(str) = datetime2unix(DateTime(str, "yyyymmdd"))

unix2int(t) = parse(Int, unix2str8(t))
int2unix(i) = str2unix(string(i))
