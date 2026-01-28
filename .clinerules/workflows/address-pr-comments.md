ALWAYS start by telling the user "WORKFLOW: I WILL ADDRESS PR COMMENTS".

1. Use the command line `bundle exec ruby ./tools/github/check_unresolved_pr_comments` to know about all the unresolved PR comments.
2. Every unresolved comment that starts with the string "/cline" and that has no direct reply from you yet should be addressed by you.
3. A comment can invite you to perform a code change. If that is the case, you should work on the task again to implement what is asked by the user.
4. ALWAYS think about the reply body you will post to the comment you are addressing:
  - If you added new commits because of that comment, explain what improvements you made in your reply body.
  - If the user was asking a question in his comment, give an answer to his question in your reply body.
  - If you think the user comment did not need any code change, explain the reason why you think so in your reply body.
5. ALWAYS reply to a comment using the command line `bundle exec ruby ./tools/github/reply_to_comment <pr_number> <comment_id> <your_comment_body_reply>`. The Pull Request number and original comment database ID are found from the output of step 1 (`bundle exec ruby ./tools/github/check_unresolved_pr_comments`). **New lines in the reply comment should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/github/reply_to_comment 3 12345 "Line 1\nLine 2\nLine 3"`**
