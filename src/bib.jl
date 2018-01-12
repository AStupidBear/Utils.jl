"""
```
file = "A unified approach to building and controlling spiking attractor networks"
name2bib(file)
clipboard(join([file, "12"], "\n"))
name2bib()
```
"""
function name2bib(file::AbstractString; issn = "", volume= "", issue = "", pages = "")
    try
        filep = replace(file, " ", "+")
        url = "http://libgen.io/scimag/index.php?s=$filep&journalid=$issn&&v=$volume&i=$issue&p=$pages&redirect=1"
        content = url |> download |> readstring

        pat = r"<a href=\"(.*)\"  title=.*>Libgen"
        url = match(pat, content).captures[1] |> string
        content = url |> download |> readstring

        pat = r"<textarea.*>(.*)</textarea>"s
        bib = match(pat, content).captures[1] |> string
    catch
        file
    end
end

function name2bib()
    files = strip.(split(clipboard(), "\n"))
    filter!(x->!isempty(x), files)
    fails = []; succs = []
    for file in files
        bib = bibtex(file)
        if bib != file
            push!(succs, bib)
        else
            push!(fails, file)
        end
    end
    bibs = join([succs; fails],"\n")
    println(bibs)
    clipboard(bibs)
    bibs
end


"""
```julia
file = "A unified approach to building and controlling spiking attractor networks"
libgen(file)
clipboard(join([file,"12"],"\n"))
libgen()
```
"""
function libgen(file::AbstractString; issn = "", volume = "", issue = "", pages = "")
    try
        filep = replace(file, " ", "+")
        url = "http://libgen.io/scimag/index.php?s=$filep&journalid=$issn&&v=$volume&i=$issue&p=$pages&redirect=1"
        content = url |> download |> readstring

        pat = r"<a href=\"(.*)\"  title=.*>Libgen"
        url = match(pat, content).captures[1] |> string
        content = url |> download |> readstring

        pat = r"<a href='(.*)'><h2>DOWNLOAD</h2>"
        url = match(pat, content).captures[1] |> string
    catch
        file
    end
end

function libgen()
    files = strip.(split(clipboard(), "\n"))
    filter!(x->!isempty(x), files)
    fails = []; succs = []
    for file in files
        url = libgen(file)
        if url != file
            push!(succs, url)
        else
            push!(fails, file)
        end
    end
    urls = join([succs; fails],"\n")
    println(urls)
    clipboard(urls)
    urls
end

"""doi2cit("10.1126/science.169.3946.635")"""
doi2cit(doi) = readstring(`curl -LH "Accept: text/x-bibliography; style=apa" https://doi.org/$doi -k`)

"""doi2bib("10.1126/science.169.3946.635")"""
doi2bib(doi) = readstring(`curl -LH "Accept: application/x-bibtex" https://doi.org/$doi -k`)
