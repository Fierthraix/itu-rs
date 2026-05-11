# itu-rs Python bindings

Python bindings for the Rust `itu-rs` crate.

Install from PyPI:

```sh
uv add itu-rs
# or: pip install itu-rs
```

Import from Python:

```python
import itu_rs

attenuation = itu_rs.gas_attenuation_default(
    45.4215, -75.6972, 12.0, 30.0, 0.1, 1.2
)
print(attenuation)
```

The Python wheels embed the ITU-R model data through the Rust crate's `data`
feature, so no separate data directory is required for normal wheel installs.

For local development, use `uv`:

```sh
uv run --project python --group dev maturin develop --manifest-path python/Cargo.toml
uv run --project python --group dev pytest python/tests
```
