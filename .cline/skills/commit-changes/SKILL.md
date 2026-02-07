---
name: commit-changes
description: Give all steps to commit changes and push them on Github
---

# Commit changes

When committing changes, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL COMMIT CHANGES".

## 1. Create a meaningful commit

- Identify all the files that make sense to commit altogether as part of 1 commit.
- Add those identified files using `git add <file1> <file2> ... <fileN>`.
- ALWAYS create a git commit using `bundle exec ruby ./tools/git/commit "<Meaningful git commit comment>"`. **New lines in the comment should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/git/commit "Line 1\nLine 2\nLine 3"`**.
- NEVER use `git commit` directly.

## 2. Push this commit on Github

- Push the commit on Github using `git push github`.

## 3. Make sure a Pull Request is created for the current branch

- Check on the Github project if there is already a Pull Request created for the current branch.
- If there isn't any Pull Request for the current branch, ALWAYS create a Pull Request using the Skill `create-pull-request`.
