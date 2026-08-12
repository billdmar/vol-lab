# ENV.md — machine & toolchain (vol-lab)

Captured 2026-08-07T16:50:50Z on the dev machine. CI runs on ubuntu-latest with the same pins.

## Dev machine
- Hardware: Apple M4 (arm64)
- OS: macOS 26.5 (Darwin 25.5.0)
- Xcode CLT: /Applications/Xcode.app/Contents/Developer
- Homebrew: Homebrew 6.0.15

## Toolchain
- Python (project venv): Python 3.12.13
- git: git version 2.50.1 (Apple Git-155)
- gh: gh version 2.93.0 (2026-05-27)

## Pinned Python dependencies (project venv == CI)
Runtime (in `[project.dependencies]`) — lean; the Deribit client uses the stdlib `urllib`:
```
matplotlib         3.11.1
numpy              2.5.1
scipy              1.18.0
```
Dev tooling (in `[project.optional-dependencies].dev`):
```
coverage           7.15.4
hypothesis         6.165.2
mypy               2.3.0
pytest             9.1.1
ruff               0.16.1
```

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # project + dev extras, from the pinned pyproject
ruff check . && mypy && coverage run -m pytest && coverage report
```
