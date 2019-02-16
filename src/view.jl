export viewdf
function viewdf(df)
    fn = tempname() * ".csv"
    writetable(fn, df)
    Sys.iswindows() && spawn(`csvfileview $fn`)
end

export viewmat
function viewmat(x)
    fn = tempname() * ".csv"
    writecsv(fn, vcat(rowvec(1:size(x, 2)), x))
    Sys.iswindows() && spawn(`csvfileview $fn`)
end
