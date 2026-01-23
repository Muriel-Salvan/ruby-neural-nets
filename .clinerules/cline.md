Always follow all the rules defined in this file, and recursively every file linked by this one.

# Project specific rules

Project specific rules are defined in the [../rules/all.md](../rules/all.md) file.

# Cline specific rules

## General rules

1. If the user is asking for a "Quick task", then tests, documentation and git commits are not needed from Cline, unless specifically asked in the quick task description. However other rules still apply.
2. If the user is not asking for a "Quick task", then all rules must be followed.
3. RSpec tests should only be run in WSL, using the `.\tools\rspec_wsl.cmd` command, accepting any rspec argument if needed. Don't use `bundle exec rspec` directly.
4. If the user is asking you to address Pull Request comments, use the command line `bundle exec ruby .\tools\check_unresolved_pr_comments` to get those comments, and address them (see section "When addressing Pull Requests comments" below).
5. You can use the Github CLI tool (`gh`) to handle Github Pull Requests, comments and workflows.

## When working on a task

1. When modified files can be committed in a meaningful commit, create a new git commit in the current branch (**never in another branch**) and push the branch to the github remote. This can be done using the command: `git add <file1> <file2> ... <fileN>; git commit -m"<Meaningful git commit comment>"; git push github`.
2. When you commit code, **always add a line to the commit message at the very end that says "Co-authored by: Cline (<model_name>)" with the model name given by the command `bundle exec ruby .\tools\cline_model`.**
3. For big files, replace_in_file does not work properly. Always check that the file is containing the edits you expect. Use write_in_file when you see that there are no edits in the proposed changes.

## Before completing a task

1. **Always make sure that tests are all running without failures**, and fix any failure if they don't succeed (get back to working on the task).
2. Always make sure that all your modifications are committed and pushed on the github remote in the current branch.
3. The first time you push this branch on Github, create a Pull Request of this branch compared to the main branch, and add a meaningful title and description for this Pull Request. The description should also include a section with the exact initial prompt of the user for this task.

## When addressing Pull Requests comments

1. Every unresolved comment made by a user that starts with the string "/cline" should be addressed by you. Those comments are all returned by the command `bundle exec ruby .\tools\check_unresolved_pr_comments` (among other comments as well that can be used to understand the context).
2. A comment can invite you to perform another code change, if so get back to working on the task with all associated rules.
3. You should reply to every comment that starts with "/cline" and has no reply from you yet:
  - You can reply to a comment using the command line `bundle exec ruby .\tools\reply_to_comment <pr_number> <comment_id> <your_comment_body_reply>`. The Pull Request number and original comment database ID are found from the output of `bundle exec ruby .\tools\check_unresolved_pr_comments`.
  - **Always prefix your reply body with the string "[Cline (<model_name>)] " with the model name given by the command `bundle exec ruby .\tools\cline_model`.**
  - If you added new commits because of that comment, explain what improvements you made in your reply body.
  - If the user was asking a question in his comment, give an answer to his question in your reply body.
  - If you think the user comment did not need any code change, explain the reason why you think so in your reply body.
