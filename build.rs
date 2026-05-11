#[cfg(feature = "data")]
use std::env;
#[cfg(feature = "data")]
use std::fs;
#[cfg(feature = "data")]
use std::io::{Read, Write};
#[cfg(feature = "data")]
use std::path::{Path, PathBuf};

#[cfg(feature = "data")]
const DEFAULT_DATA_URL: &str =
    "https://github.com/Fierthraix/itu-rs/releases/download/itu-rs-data-v1/itu-rs-data-v1.zip";
#[cfg(feature = "data")]
const DEFAULT_DATA_SHA256: &str =
    "20436b31774260950032328e18630dbfe3d13a2d679fd1de5edad54bf153a294";

fn main() {
    println!("cargo:rerun-if-env-changed=ITU_RS_DATA_ARCHIVE");
    println!("cargo:rerun-if-env-changed=ITU_RS_DATA_CACHE");
    println!("cargo:rerun-if-env-changed=ITU_RS_DATA_SHA256");
    println!("cargo:rerun-if-env-changed=ITU_RS_DATA_URL");

    #[cfg(feature = "data")]
    feature_data_main();
}

#[cfg(feature = "data")]
fn feature_data_main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let generated = out_dir.join("bundled_data.rs");

    if env::var_os("DOCS_RS").is_some() {
        write_empty_bundled_data(&generated);
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let local_data = manifest_dir.join("data");
    let data_dir = if local_data.join("1511/v2_lat.npz").exists() {
        local_data
    } else {
        let extracted = out_dir.join("itu-rs-data");
        if extracted.join("data/1511/v2_lat.npz").exists() {
            extracted.join("data")
        } else {
            fs::create_dir_all(&extracted).expect("failed creating data extraction directory");
            let archive = data_archive_path(&out_dir);
            extract_archive(&archive, &extracted);
            extracted.join("data")
        }
    };

    write_bundled_data(&generated, &data_dir);
}

#[cfg(feature = "data")]
fn data_archive_path(out_dir: &Path) -> PathBuf {
    if let Some(path) = env::var_os("ITU_RS_DATA_ARCHIVE") {
        let path = PathBuf::from(path);
        verify_archive(&path);
        return path;
    }

    let cache_dir = env::var_os("ITU_RS_DATA_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join("data-cache"));
    fs::create_dir_all(&cache_dir).expect("failed creating ITU-R data cache directory");

    let archive = cache_dir.join("itu-rs-data.zip");
    if archive.exists() {
        verify_archive(&archive);
        return archive;
    }

    let url = env::var("ITU_RS_DATA_URL").unwrap_or_else(|_| DEFAULT_DATA_URL.to_string());
    let bytes = download_archive(&url);
    let mut file =
        fs::File::create(&archive).expect("failed creating downloaded ITU-R data archive");
    file.write_all(&bytes)
        .expect("failed writing downloaded ITU-R data archive");
    verify_archive(&archive);
    archive
}

#[cfg(feature = "data")]
fn download_archive(url: &str) -> Vec<u8> {
    let response = ureq::get(url)
        .call()
        .unwrap_or_else(|err| panic!("failed downloading ITU-R data archive from {url}: {err}"));
    let mut reader = response.into_body().into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| panic!("failed reading ITU-R data archive response: {err}"));
    bytes
}

#[cfg(feature = "data")]
fn verify_archive(path: &Path) {
    let expected = env::var("ITU_RS_DATA_SHA256")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_DATA_SHA256.to_string());
    if expected.trim().is_empty() {
        return;
    }

    use sha2::{Digest, Sha256};
    let bytes = fs::read(path).unwrap_or_else(|err| {
        panic!(
            "failed reading ITU-R data archive {}: {err}",
            path.display()
        )
    });
    let digest = Sha256::digest(&bytes);
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        actual,
        expected.to_ascii_lowercase(),
        "SHA256 mismatch for ITU-R data archive {}",
        path.display()
    );
}

#[cfg(feature = "data")]
fn extract_archive(archive: &Path, target: &Path) {
    let file = fs::File::open(archive).unwrap_or_else(|err| {
        panic!(
            "failed opening ITU-R data archive {}: {err}",
            archive.display()
        )
    });
    let mut zip = zip::ZipArchive::new(file).unwrap_or_else(|err| {
        panic!(
            "failed reading ITU-R data archive {}: {err}",
            archive.display()
        )
    });

    for idx in 0..zip.len() {
        let mut entry = zip
            .by_index(idx)
            .unwrap_or_else(|err| panic!("failed reading ITU-R data archive entry {idx}: {err}"));
        if entry.is_dir() {
            continue;
        }

        let Some(rel_path) = data_relative_path(entry.name()) else {
            continue;
        };
        let out_path = target.join("data").join(rel_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("failed creating {}: {err}", parent.display()));
        }
        let mut out_file = fs::File::create(&out_path)
            .unwrap_or_else(|err| panic!("failed creating {}: {err}", out_path.display()));
        std::io::copy(&mut entry, &mut out_file)
            .unwrap_or_else(|err| panic!("failed extracting {}: {err}", out_path.display()));
    }

    if !target.join("data/1511/v2_lat.npz").exists() {
        panic!(
            "ITU-R data archive {} did not contain data/1511/v2_lat.npz",
            archive.display()
        );
    }
}

#[cfg(feature = "data")]
fn data_relative_path(name: &str) -> Option<&str> {
    if let Some(path) = name.strip_prefix("data/") {
        return Some(path);
    }
    name.split_once("/data/").map(|(_, path)| path)
}

#[cfg(feature = "data")]
fn write_bundled_data(generated: &Path, data_dir: &Path) {
    let mut files = Vec::new();
    collect_data_files(data_dir, data_dir, &mut files);
    files.sort();

    let mut output = String::from("pub fn get(rel_path: &str) -> Option<&'static [u8]> {\n");
    output.push_str("    match rel_path {\n");
    for rel_path in files {
        let full_path = data_dir.join(&rel_path);
        output.push_str("        ");
        output.push_str(&format!(
            "{rel_path:?} => Some(include_bytes!({:?})),\n",
            full_path.display().to_string()
        ));
    }
    output.push_str("        _ => None,\n");
    output.push_str("    }\n");
    output.push_str("}\n");

    fs::write(generated, output)
        .unwrap_or_else(|err| panic!("failed writing {}: {err}", generated.display()));
}

#[cfg(feature = "data")]
fn write_empty_bundled_data(generated: &Path) {
    fs::write(
        generated,
        "pub fn get(_rel_path: &str) -> Option<&'static [u8]> { None }\n",
    )
    .unwrap_or_else(|err| panic!("failed writing {}: {err}", generated.display()));
}

#[cfg(feature = "data")]
fn collect_data_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed reading data directory {}: {err}", dir.display()))
    {
        let entry =
            entry.unwrap_or_else(|err| panic!("failed reading data directory entry: {err}"));
        let path = entry.path();
        if path.is_dir() {
            collect_data_files(root, &path, out);
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .expect("data file is under data root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
}
