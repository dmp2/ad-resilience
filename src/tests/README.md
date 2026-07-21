Run the project-extension tests from the package root after installing dependencies:

```bash
PYTHONPATH=src pytest -q tests/test_project_extensions.py
```

A separate end-to-end test should be run on Dalet with the pinned xmodmap repository,
one legacy particle case, and one known registration output. These unit tests do not
validate PyKeOps kernels or the numerical optimizer.
