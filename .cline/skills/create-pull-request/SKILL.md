---
name: create-pull-request
description: Give all steps to create a Pull Request for the current git branch on Github
---

# Create Pull Request

When creating a Pull Request changes, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL CREATE A PULL REQUEST".

## 1. Create a temporary file with a good description for the Pull Request

- Find a meaningful description for all the changes that you have in the current branch, and for the task you want to achieve in this branch. ALWAYS add a section in the description that contains the exact initial prompt of the USER for this task.
- ALWAYS ask the USER if this PR relates to or closes some Github issues. If the user gives a list of issues IDs that relate or close this Pull Request, make sure to add this information in the Pull Request description, in a dedicated section with mentions like "Closes #<issue_id>" or "Relates to #<issue_id>".
- Write the devised Pull Request description in a temporary file (later referenced as {{pr_description_file}}).

## 2. Create the Pull Request between the current branch and main

- ALWAYS create a Pull Request using the command `bundle exec ruby ./tools/github/create_pr {{title}} {{pr_description_file}}`. Use a meaningful title for this Pull Request.
- NEVER use the `gh` CLI to create Pull Requests.
- Delete the temporary description file {{pr_description_file}} once the Pull Request has been created.
