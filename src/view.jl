export viewdf
function viewdf(df)
    fn = tempname() * ".csv"
    writetable(fn, df)
    is_windows() && spawn(`csvfileview $fn`)
end

export viewmat
function viewmat(x)
    fn = tempname() * ".csv"
    writecsv(fn, vcat(rowvec(1:size(x, 2)), x))
    is_windows() && spawn(`csvfileview $fn`)
end
