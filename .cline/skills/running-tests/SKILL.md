---
name: running-tests
description: Run tests. Use when any test (single ones or all) needs to be run.
---

# Running tests

When running any test scenario, follow this workflow:

## 1. Find out which tests need to be run

- Identify the names of the test scenarios that need to be run. By default all tests are run.

## 2. Run tests using WSL only

- ALWAYS use the `.\tools\wsl\bash.cmd bundle exec rspec` CLI to run tests. NEVER use another command.
- Add all necessary CLI arguments that you need to select some test scenarios, using RSpec CLI arguments.
- For example, running all tests named "tracks progress and reports correct cost and accuracy" is done by using the CLI `.\tools\wsl\bash.cmd bundle exec rspec -e "tracks progress and reports correct cost and accuracy"`.

## 3. Check the output of the tests run

- ALWAYS check for the output of the tests run. Successful tests run should output "0 failures" at the end. If this string is not found then it means tests are failing.
