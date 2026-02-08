---
name: wrap-up-task
description: Provide all rules and steps to perform before attempting task completion
---

# Wrap-up task

When wrapping up a task before attempting completion, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL WRAP-UP MY TASK".

## 1. Check that there is no regression

- ALWAYS make sure that tests are all running without failures.
- ALWAYS fix all the failures that you see in the tests output. In this case don't mark the task as completed, and get back to working on the task.

## 2. Check that documentation is up-to-date

- Update the README file if needed.

## 3. Make sure all modifications are pushed

- ALWAYS commit all your changes in the current branch.
- ALWAYS check that the git history of your branch is semi-linear (you should not have any merge commit between your branch and main), and that it is based on the latest main branch. If that is not the case, rebase the branch on main
