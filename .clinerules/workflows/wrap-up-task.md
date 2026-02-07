<task name="Wrap-up task">

<task_objective>
Perform all needed checks before attempting a task's completion.
</task_objective>

- ALWAYS start by informing the user that you are running this workflow, saying "WORKFLOW: I WILL WRAP UP MY TASK".

## 1. Check that there is no regression

1. ALWAYS make sure that tests are all running without failures. Use the running-tests skill to do so.
2. ALWAYS fix all the failures that you see in the tests output. In this case don't mark the task as completed, and get back to working on the task.

## 2. Check that documentation is up-to-date

1. ALWAYS use the Skill updating-readme to check what should be changed in the README in the scope of your task.

## 3. Make sure all modifications are pushed

1. ALWAYS make sure that all your modifications are committed and pushed on the github remote in the current branch. Use the workflow /commit-changes.md if some modifications are still not pushed.
2. ALWAYS check that the git history of your branch is semi-linear (you should not have any merge commit between your branch and main), and that it is based on the latest main branch. If that is not the case use the workflow /rebase-branch.md.

</detailed_sequence_steps>

</task>
