# Agent Notes

- Keep `CHANGELOG.md` focused on user-facing changes: public APIs, documented
  behavior, data/package availability, compatibility, and docs that help users.
  Avoid internal-only maintenance details unless they materially affect users.
- Keep Python bindings in the separate `python/` package so docs.rs remains
  focused on the Rust API. Prefer small binding macros for repetitive wrappers,
  but keep Python-facing classes and conversions explicit.
- Use `uv` for Python environment and tool execution where practical. Keep
  `maturin` as the build backend and wheel builder, but prefer invoking it via
  `uv run` or CI-installed `uv` unless a purpose-built action is a better fit.
