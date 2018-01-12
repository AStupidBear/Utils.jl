function paper(fn)
    download("https://raw.githubusercontent.com/ihrke/markdown-paper/master/templates/elsarticle-template-1-num.latex", "elsarticle-template-1-num.latex")
    run(`pandoc $fn.md
    -s -S -o $fn.pdf
    --filter=pandoc-crossref
    --filter=pandoc-citeproc
    --template=elsarticle-template-1-num.latex
    --bibliography=references.bib`)
    rm("elsarticle-template-1-num.latex")
end


const STYLEPATH = "https://raw.githubusercontent.com/tompollard/phd_thesis_markdown/master/style"
# const STYLEPATH = "https://raw.githubusercontent.com/AStupidBear/phd_thesis_markdown/master/style"

function thesis(fn, fmt = "pdf"; title = "This is the title of the thesis", name = "Yao Lu")
    if fmt == "pdf"
        download("$STYLEPATH/template.tex", "template.tex")
        download("$STYLEPATH/preamble.tex", "preamble.tex")
        run(`pandoc $(glob("*.md"))
        -o $fn.pdf
        --filter=pandoc-crossref
        --filter=pandoc-citeproc
        --include-in-header=preamble.tex
        --template=template.tex
        --bibliography=references.bib
        --csl=$STYLEPATH/ref_format.csl
        --highlight-style=pygments
        --variable=fontsize:12pt
        --variable=papersize:a4paper
        --variable=documentclass:report
        --number-sections
        --latex-engine=xelatex`)
        rm("template.tex")
        rm("preamble.tex")
    elseif fmt == "html"
        download("$STYLEPATH/template.html", "template.html")
        download("$STYLEPATH/style.css", "style.css")
        @>(readstring("template.html"),
        replace("This is the title of the thesis", title),
        replace("Firstname Surname", name)) |>
        x -> write("template.html", x)
        run(`pandoc $(glob("*.md"))
        -o $fn.html
        --standalone
        --filter=pandoc-crossref
        --filter=pandoc-citeproc
        --template=template.html
        --bibliography=references.bib
        --csl=$STYLEPATH/ref_format.csl
        --include-in-header=style.css
        --toc
        --number-sections
        --mathjax`)
        rm("template.html")
        rm("style.css")
    end
end

"""
markdown-preview-enhanced

# Examples

```{julia output:"html", id:"hehe"}
using Plots; plot(rand(10)) |> mpe
```
"""
function mpe(p, fmt::Symbol = :svg)
    p.attr[:html_output_format] = fmt
    show(STDOUT, MIME("text/html"), p)
end
