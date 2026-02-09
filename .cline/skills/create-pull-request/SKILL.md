---
name: create-pull-request
description: Give all steps to create a Pull Request for the current git branch on Github
---

# Create Pull Request

When creating a Pull Request changes, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL CREATE A PULL REQUEST".

## 1. Devise the list of Github issues linked to this Pull Request

- ALWAYS ask the USER which Github issues are closed by or related to this Pull Request, even if you know of some of those issues already. There could be more Github issues that you are not aware of.
- Use any information from the previous USER prompts to know which additional issues are closed by or related to this Pull Request.

## 2. Create a temporary file with a good description for the Pull Request

- ALWAYS devise a meaningful Pull Request description for all the changes that you have in the current branch, and for the task you want to achieve in this branch.
- ALWAYS add a section in the Pull Request description that lists all Github issues closed by or related to this Pull Request (devised in step 1), with mentions like "Closes #{{issue_id}}" or "Relates to #{{issue_id}}".
- ALWAYS add a section in the Pull Request description that contains the exact initial prompt of the USER for this task, and all USER inputs or precisions that you have received from the USER while implementing the task.
- Write the devised Pull Request description in a temporary file (later referenced as {{pr_description_file}}).

## 3. Create the Pull Request between the current branch and main

- ALWAYS create a Pull Request using the command `bundle exec ruby ./tools/github/create_pr {{title}} {{pr_description_file}}`. Use a meaningful title for this Pull Request.
- NEVER use the `gh` CLI to create Pull Requests.
- Delete the temporary description file {{pr_description_file}} once the Pull Request has been created.
