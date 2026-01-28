ALWAYS start by telling the user "WORKFLOW: I WILL REBASE MY BRANCH".

1. Use `git rebase main` to bring your branch on top of the main branch. **NEVER use `git merge`**.
2. Fix any conflict you see during the rebase, and continue the rebase using `git rebase --continue` until all your commits have been rebased properly.
3. If you don't know how to solve a conflict, ask the user.
4. ALWAYS push your rebased branch to github using the --force-with-lease option: `git push github --force-with-lease`.
