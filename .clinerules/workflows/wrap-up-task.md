1. **ALWAYS make sure that tests are all running without failures**. Use the running-tests skill to do so.
2. ALWAYS fix all the failures that you see in the tests output. In this case don't mark the task as completed, and get back to working on the task.
3. Check what should be changed in the README in the scope of your task. Check the updating-readme skill to do so.
4. ALWAYS make sure that all your modifications are committed and pushed on the github remote in the current branch. Use the workflow /commit-changes.md if some modifications are still not pushed.
5. ALWAYS check that the git history of your branch is semi-linear (you should not have any merge commit between your branch and main), and that it is based on the latest main branch. If that is not the case use the workflow /rebase-branch.md.
