# GitHub upload checklist

Target repository: `https://github.com/khk0606/ADM-MoE-Teacher.git`

Before the first push, run from the clean repository directory:

```bash
git status --short
git ls-files | sort
find . -type f -size +50M -not -path './.git/*'
git grep -nE '/home/kang|/Users/kanghyunkyu|DESKTOP-KANG|kang-gpu-server' -- .
git grep -nE 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}' -- .
```

The two `git grep` commands should print nothing. The large-file check should also print nothing.

Review the staged file list, then commit and push:

```bash
git diff --cached --stat
git commit -m "Initial source release"
git push -u origin main
```

Never use `git add -f` to override `.gitignore` for datasets, checkpoints, experiment outputs, motion TXT files, or Unity exports.
