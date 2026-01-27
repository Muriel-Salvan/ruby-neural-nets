ALWAYS start by telling the user "I WILL COMMIT CHANGES".

1. Identify all the files that make sense to commit altogether as part of 1 commit.
2. Add those identified files using `git add <file1> <file2> ... <fileN>`.
3. Create a git commit using `bundle exec ruby ./tools/git/commit "<Meaningful git commit comment>"`. NEVER use `git commit`. **New lines in the comment should be given using \n and not real new lines, like this: `bundle exec ruby ./tools/git/commit "Line 1\nLine 2\nLine 3"`**.
4. Push the commit on Github using `git push github`.
5. If it is the first time you push a change to Github for this branch, create a Pull Request using workflow /create-pull-request.md.
