export git, jgit

jgit() = git(pwd(), ".jl")

function git(path = pwd(), suffix = "")
    folder = splitdir(path)[end]
    cd(path)
    run(`git config --global user.name "Yao Lu"`)
    run(`git config --global user.email "luyaocns@gmail.com"`)
    run(`git init`)
    try run(`git remote add $folder git@github.com:AStupidBear/$folder$suffix.git`) end
    try run(`git pull $folder master`) end
    run(`git add .`)
    try run(`git commit -m $(now())`) end
    run(`git push $folder master`)
end
