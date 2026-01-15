Always follow all the rules defined in this file, and recursively every file linked by this one.

# Project specific rules

Project specific rules are defined in the [../rules/all.md](../rules/all.md) file.

# Cline specific rules

## General rules

1. If the user is asking for a "Quick task", then tests, documentation and git commits are not needed from Cline, unless specifically asked in the quick task description. However other rules still apply.
2. If the user is not asking for a "Quick task", then all rules must be followed.

## When starting a task

## When working on a task

1. When modified files can be committed in a meaningful commit, create a new git commit in the current branch (**never in another branch**) and push the branch to the github remote. This can be done using the command: `git add <file1> <file2> ... <fileN>; git commit -m"<Meaningful git commit comment>"; git push github`.
2. For big files, replace_in_file does not work properly. Always check that the file is containing the edits you expect. Use write_in_file when you see that there are no edits in the proposed changes.
3. Unit tests can't be run using `bundle exec rspec` directly. Use `.\wsl_tests.cmd` instead.

## Before completing a task

1. **Always make sure that tests are all running without failures**, and fix any failure if they don't succeed (get back to working on the task).
2. Always make sure that all your modifications are committed and pushed on the github remote in the current branch.
