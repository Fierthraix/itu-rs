use itu_rs::{atmospheric_attenuation_slant_path, SlantPathOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let attenuation = atmospheric_attenuation_slant_path(
        45.4215,
        -75.6972,
        12.0,
        30.0,
        0.1,
        1.2,
        SlantPathOptions::default(),
    )?;

    println!("{:.6} dB", attenuation.total_db);
    Ok(())
}
