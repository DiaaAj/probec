# Releasing

probec publishes to PyPI from GitHub Actions (`.github/workflows/publish.yml`) through PyPI's trusted
publishing, so no API token lives in the repository.

**One-time setup (PyPI account owner):** on pypi.org, under Publishing, add a pending trusted publisher
for project `probec`: owner `DiaaAj`, repository `probec`, workflow `publish.yml`, environment `pypi`.
In the GitHub repository settings, create an environment named `pypi` (optionally with required
reviewers).

**Each release:**
1. Set `version` in `pyproject.toml` and add the changes to `PROGRESS.md`; merge to `main` with CI green.
2. Create a GitHub release with tag `v<version>` (for example `v0.2.0`). The workflow builds the sdist
   and wheel, checks that the tag matches the version, runs `twine check` and publishes.

Check locally first: `python -m build && twine check dist/*`.
