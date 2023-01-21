# 在VSCode中使用Git时遇到的问题

## 1. 在最初的 master 这个 branch 中，没法创建新的 branch
- 查看 git output 后发现：
    - > git checkout -q -b sweep --no-track HEAD
    - > [info] fatal: 'HEAD' is not a commit and a branch 'sweep' cannot be created from it
- 搜索 Stack Overflow 之后：
    - git remote -v 显示为空，即 master branch 之下没有remote
    - 猜测：原因可能是： master 这个 branch 是本地创建的文件，而不是由 git clone 克隆到本地的 repo 出发拉取的 branch，因此可能 remote 不存在？

## 2. 在 git clone 得到的 FFA_test 这个文件夹下，从其最初的 main 这个 branch 出发，可以拉取出 fix 这个新的 branch， 但没法把 fix 给 push 回到 github 上的 repo，显示没有权限
- git remove -v 显示有两个 remote, 分别是 origin 和 upstream
- 报错如下：
    - > git push -u origin fix
    - > remote: Permission to Raidriar-Dai/FFA_test.git denied to ianpundar.
    - > fatal: unable to access 'https://github.com/Raidriar-Dai/FFA_test.git/': The requested URL returned error: 403
    - 经查：ianpundar 似乎是我们组内另一位 彭雨昂 同学的 github账号；但我现在登录的是我的 github 账号，之前也从未与那位同学有接触，不知道怎么能把账号切换到自己的。