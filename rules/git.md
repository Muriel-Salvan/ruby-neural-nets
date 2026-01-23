# Git workflows rules

1. The git history of any branch (feature or main) should always be semi-linear. That means the only merge commits in the history are the ones of feature branches on main directly.
2. Feature branches should always be rebased on the main branch. If a feature branch lags behind, then it should be brought up to date **using `git rebase`, never use `git merge`**.
