---
name: running-tests
description: Provide rules on how to run tests
---

# running-tests

ALWAYS start by telling the user "SKILL: I WILL RUN TESTS".

1. ALWAYS run tests using the WSL-wrapped RSpec CLI command: `./tools/wsl/bash.cmd bundle exec rspec`
2. If you want to select specific tests or change RSpec run, you can use any normal rspec CLI argument, like this: `./tools/wsl/bash.cmd bundle exec rspec -e "tracks progress and reports correct cost and accuracy"`
3. ALWAYS check for the output of the tests run. Successful tests run should output "0 failures" at the end. If this string is not found then it means tests are failing.

## Usage

Use this skill every time you need to run tests.

## Steps

1. Run tests from WSL using `./tools/wsl/bash.cmd bundle exec rspec`
2. Analyze the output to understand if tests are failing or not.
