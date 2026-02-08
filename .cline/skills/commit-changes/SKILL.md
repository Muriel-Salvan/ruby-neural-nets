---
name: commit-changes
description: Give all steps to commit changes and push them on Github
---

# Commit changes

When committing changes, follow this workflow:

- ALWAYS start by informing the user that you are running this skill, saying "SKILL: I WILL COMMIT CHANGES".

## 1. Stage all the files that should be part of the commit

- Identify all the files that make sense to commit altogether as part of 1 commit.
- Add those identified files using `git add <file1> <file2> ... <fileN>`.

## 2. Create a temporary file with the commit description

- Devise a meaningful commit comment, and write it in a temporary file (later referenced as {{description_file}}).

## 3. Create the commit

- ALWAYS create a git commit using `bundle exec ruby ./tools/git/commit {{description_file}}`.
- NEVER use `git commit` directly.
- Delete the temporary description file {{description_file}} once the git commit has been done.

## 4. Push this commit on Github

- Push the commit on Github using `git push github`.

## 5. Make sure a Pull Request is created for the current branch

- Check on the Github project if there is already a Pull Request created for the current branch.
- If there isn't any Pull Request for the current branch, create one.
