# Release Checklist

This checklist is for tagging a public-ready version of `llm-memory`.

## Scope

Use this before:

- publishing the repository publicly
- cutting a tagged release
- announcing a meaningful milestone version

## Pre-Release

- [ ] confirm the target version number
- [ ] review `pyproject.toml` version and any user-visible release notes
- [ ] confirm the roadmap items included in the release
- [ ] check `git status` is clean except for intended release changes

## Quality Gate

- [ ] run `pytest -q`
- [ ] run `pytest tests/test_golden_path.py -q`
- [ ] confirm Docker configuration is valid with `docker compose config`
- [ ] verify the quickstart still matches the shipped runtime

## Product Surface

- [ ] confirm `README.md` describes the active runtime and tool surface
- [ ] confirm `QUICKSTART.md` is aligned with the verified golden path
- [ ] confirm `DOCKER_GUIDE.md` is aligned with the active container setup
- [ ] confirm `CONTRIBUTING.md` still matches the branch and PR workflow
- [ ] confirm `LICENSE` is present and correct

## Runtime Checks

- [ ] verify stdio startup: `python -m src.mcp_server.server`
- [ ] verify HTTP startup: `python -m src.mcp_server.http_server`
- [ ] verify `GET /health` responds with status `ok`
- [ ] verify local import/export paths still resolve under `MEMORY_IMPORT_EXPORT_BASE_DIR`

## Release Artifacts

- [ ] update version if required
- [ ] prepare a short release summary
- [ ] create the release commit if needed
- [ ] create and push the git tag
- [ ] create the GitHub release entry

## Post-Release

- [ ] verify the tagged commit matches `main`
- [ ] verify the public repository landing page links to the right docs
- [ ] record the released version and date in project notes if needed
