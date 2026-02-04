<task name="Create Pull Request">

<task_objective>
Create a Pull Request on Github for the current branch.
</task_objective>

<detailed_sequence_steps>

- ALWAYS start by informing the user that you are running this workflow, saying "WORKFLOW: I WILL CREATE A PR".

## 1. Devise a good description for the Pull Request

1. Find a meaningful description for all the changes that you have in the current branch, and the task you want to achieve in this branch. ALWAYS add a section in the description that contains the exact initial prompt of the user for this task.

2. Ask the USER if this PR relates to or closes some Github issues. If the user gives a list of issues that relate or close this Pull Request, make sure to add this information in the Pull Request description, in a dedicated section with mentions like "Closes #<issue_id>" or "Relates to #<issue_id>".

## 2. Create the Pull Request between the current branch and main

1. Create a Pull Request using the command `bundle exec ruby ./tools/github/create_pr <title> <description>`. Use a meaningful title and the description you devised previously for this Pull Request. **New lines in the description should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/github/create_pr "Pull Request Title" "Line 1\nLine 2\nLine 3"`**.

</detailed_sequence_steps>

</task>
