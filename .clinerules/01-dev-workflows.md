# Development workflow rules

- ALWAYS start by informing the user that you have read those rules, saying "RULES: I HAVE READ THE DEVELOPMENT WORKFLOW RULES".

## Task

- ALWAYS use the Skill `wrap-up-task` just before you complete the task.

## Skills

- ALWAYS use the Skill `commit-changes` to commit changes. NEVER use `git commit` directly.
- ALWAYS use the Skill `create-pull-request` to create a Pull Request on Github. NEVER use the `gh` CLI to create a Pull Request.
- ALWAYS use the Skill `rebase-branch` to rebase the current branch on main. NEVER use `git merge`.
- ALWAYS use the Skill `running-tests` to run tests for this project. NEVER run tests using other commands.
- ALWAYS use the Skill `updating-readme` to update the README file in the scope of your task.

## Git

- ALWAYS stay in the branch that is checked out. NEVER checkout another git branch than the one already checked out.
- When working on a task, identify changes that would make sense functionnally to be grouped in 1 commit, and commit them.
- Use the Github CLI tool (`gh`) to get information about Github Pull Requests, comments and workflows, but NEVER use this tool to post new information or modify information.

## Testing

- ALWAYS run all the tests scenarios after any code or test modification that you commit, and also before attempting completion on a task. If any test is failing, identify the root cause and fix it.
