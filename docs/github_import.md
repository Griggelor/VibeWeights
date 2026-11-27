# Public repository import and CI

Repository: https://github.com/Griggelor/VibeWeights · main · Apache-2.0.

The initial overview and LICENSE already exist. Import the reviewed v0.7.0 public snapshot as a normal fast-forward commit on top of them. Preserve LICENSE. The Windows helper included outside the source tree clones the repository, checks the expected initial commit, verifies the package manifest, copies the source, stages and commits it, then pushes without force. Authentication is handled by Git / Git Credential Manager. Do not copy tokens or local data into the repository.

The workflow runs CPU tests on Windows and Linux with Python 3.12 and 3.13, weight regeneration and smoke tests. Check actual run results in Actions before tagging a release. GitHub-hosted CI is not evidence of Strix Halo GPU support. `Data/`, `Models/`, databases and environments must remain untracked. No third-party model weights are included. GitHub Pages is a separate future deployment.

The public snapshot omits the private handoff context and sanitizes local paths in historical logs. The original v0.7.0 archive remains the source of the unmodified historical evidence. See [PUBLIC_IMPORT.md](PUBLIC_IMPORT.md) for the scope of changes.
