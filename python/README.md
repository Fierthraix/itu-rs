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

The bindings expose the same implemented Rust APIs, including the direct
[P.1144](https://www.itu.int/rec/R-REC-P.1144), [P.618](https://www.itu.int/rec/R-REC-P.618), and
[P.676](https://www.itu.int/rec/R-REC-P.676) helper functions.

The Python package downloads and caches the ITU-R model data automatically on
first import. Set `ITU_RS_DATA_DIR` to use an existing local data directory.

For local development, use `uv`:

```sh
uv run --project python --group dev maturin develop --manifest-path python/Cargo.toml
uv run --project python --group dev pytest python/tests
```
