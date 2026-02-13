# Development workflow rules

- ALWAYS start by informing the user that you have read those rules, saying "RULES: I have read the development workflow rules".

## Testing

- ALWAYS use `skill: running-cli-in-wsl-portable` when you want to run any test. Tests should ALWAYS be run using a WSL Portable environment.
- ⚠️ **CRITICAL REMINDER**: Before running ANY test, you MUST use `skill: running-cli-in-wsl-portable`. 
- Direct test execution is prohibited.

## CLI

- ALWAYS run CLI commands from the workspace's root directory.
- NEVER run commands from another directory.

## Git

- ALWAYS stay in the branch that is checked out.
- NEVER checkout another git branch than the one already checked out.
