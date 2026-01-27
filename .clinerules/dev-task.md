# Task rules, applicable to any development

1. **NEVER checkout another git branch than the one already checked out for you**.
2. ALWAYS run the workflow /wrap-up-task.md just before you complete the task.
3. When working on a task, identify commits that make sense functionnally. When you decide a commit is needed, then use the workflow /commit-changes.md.
4. ALWAYS run all tests after any code or test modification that you commit. If any test is failing, identify the root cause and fix it. Check the running-tests skill for that purpose.
5. Any code or test modification should trigger a verification of the README.md file content, and make sure that all sections still have up-to-date content regarding the change. Check the updating-readme skill for that purpose.
6. Use the Github CLI tool (`gh`) to get information about Github Pull Requests, comments and workflows, but NEVER use this tool to post new information or modify information.
