# Development workflow rules

- ALWAYS start by informing the user that you have read those rules, saying "RULES: I HAVE READ THE DEVELOPMENT WORKFLOW RULES".

## Task

- ALWAYS use the Skill `wrap-up-task` just before you complete the task.

## Git

- ALWAYS stay in the branch that is checked out. NEVER checkout another git branch than the one already checked out.
- When working on a task, identify commits that make sense functionnally and ALWAYS use the Skill `commit-changes` to commit them.
- NEVER use `git commit` directly.
- Use the Github CLI tool (`gh`) to get information about Github Pull Requests, comments and workflows, but NEVER use this tool to post new information or modify information.

## Testing

- ALWAYS use the Skill `running-tests` to run tests for this project. NEVER run tests using other commands.
- ALWAYS run all the tests scenarios after any code or test modification that you commit, and also before attempting completion on a task. If any test is failing, identify the root cause and fix it.
