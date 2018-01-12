"""
```
aria2c("https://www.youtube.com")
aria2c("https://www.youtube.com"; proxy = "127.0.0.1:1080")
```
"""
function aria2c(url, fn; proxy = "")
    dir, base  = splitdir(fn)
    run(`aria2c --max-connection-per-server=8 --all-proxy=$proxy -d $dir -o $base $url`)
    fn
end
aria2c(url; o...) = aria2c(url, tempname(); o...)

export psdownload
function psdownload(url, to = tempname())
    run("powershell (new-object system.net.webClient).downloadFile(\"$url\", \"$to\")")
    return to
end

export parseweb
function parseweb(url; relative = false, parent = false)
    opts = `--continue --recursive --convert-links --html-extension
    --page-requisites --no-check-certificate`
    relative == true && (opts = `$opts --relative`)
    parent == false && (opts = `$opts --no-parent`)
    run(`wget $opts $url`)
end
