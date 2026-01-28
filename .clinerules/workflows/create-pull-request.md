ALWAYS start by telling the user "WORKFLOW: I WILL CREATE A PR".

1. Find a meaningful description for all the changes that you have in the current branch, and the task you want to achieve in this branch. ALWAYS add a section in the description that contains the exact initial prompt of the user for this task.
2. Create a Pull Request using the command `bundle exec ruby ./tools/github/create_pr <title> <description>`. Use a meaningful title and description for this Pull Request. **New lines in the description should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/github/create_pr "Pull Request Title" "Line 1\nLine 2\nLine 3"`**.
