---
name: create-pull-request
description: Give all steps to create a Pull Request for the current git branch on Github
---

# Create Pull Request

When creating a Pull Request changes, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL CREATE A PULL REQUEST".

## 1. Devise a good description for the Pull Request

- Find a meaningful description for all the changes that you have in the current branch, and for the task you want to achieve in this branch. ALWAYS add a section in the description that contains the exact initial prompt of the USER for this task.
- Ask the USER if this PR relates to or closes some Github issues. If the user gives a list of issues IDs that relate or close this Pull Request, make sure to add this information in the Pull Request description, in a dedicated section with mentions like "Closes #<issue_id>" or "Relates to #<issue_id>".

## 2. Create the Pull Request between the current branch and main

- ALWAYS create a Pull Request using the command `bundle exec ruby ./tools/github/create_pr <title> <description>`. Use a meaningful title and the description you devised previously for this Pull Request.
- ALWAYS replace new lines with \n in your PR description. This ensures the CLI argument does not have real new lines, like this: `bundle exec ruby ./tools/github/create_pr "Pull Request Title" "Line 1\nLine 2\nLine 3"`.
- NEVER use the `gh` CLI to create Pull Requests.
