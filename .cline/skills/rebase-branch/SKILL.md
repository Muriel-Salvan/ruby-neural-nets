---
name: rebase-branch
description: Provide all rules and steps to rebase the current branch on main
---

# Rebase branch

When rebasing the current branch on main, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL REBASE MY BRANCH ON MAIN".

## 1. Rebase the branch

- ALWAYS use `git rebase main` to bring your branch on top of the main branch. **NEVER use `git merge`**.
- Fix any conflict you see during the rebase, and continue the rebase using `git rebase --continue` until all your commits have been rebased properly.
- If you don't know how to solve a conflict, ask the USER.

## 2. Push the rebased branch

- ALWAYS push your rebased branch to github using the --force-with-lease option: `git push github --force-with-lease`.
