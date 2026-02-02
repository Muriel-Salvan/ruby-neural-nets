# Development workflow rules

- ALWAYS start by informing the user that you have read those rules, saying "RULES: I HAVE READ THE DEVELOPMENT WORKFLOW RULES".

## Git

- ALWAYS stay in the branch that is checked out. NEVER checkout another git branch than the one already checked out.

## Testing

- ALWAYS use the Skill `running-tests` to run tests for this project. NEVER run tests using other commands.
- ALWAYS run all the tests scenarios after any code or test modification that you commit, and also before attempting completion on a task. If any test is failing, identify the root cause and fix it.
