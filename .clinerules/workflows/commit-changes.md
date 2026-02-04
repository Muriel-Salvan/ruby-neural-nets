<task name="Commit changes">

<task_objective>
Create a meaningful commit of current changes, push it to Github and create a Pull Request if needed.
</task_objective>

<detailed_sequence_steps>

- ALWAYS start by informing the user that you are running this workflow, saying "WORKFLOW: I WILL COMMIT CHANGES".

## 1. Create a meaningful commit

1. Identify all the files that make sense to commit altogether as part of 1 commit.

2. Add those identified files using `git add <file1> <file2> ... <fileN>`.

3. Create a git commit using `bundle exec ruby ./tools/git/commit "<Meaningful git commit comment>"`. NEVER use `git commit`. **New lines in the comment should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/git/commit "Line 1\nLine 2\nLine 3"`**.

## 2. Push this commit on Github

1. Push the commit on Github using `git push github`.

## 3. Make sure a Pull Request is created for the current branch

1. Check on the Github project if there is already a Pull Request created for the current branch.

2. If there is no Pull Request, create a Pull Request using workflow /create-pull-request.md.

</detailed_sequence_steps>

</task>
