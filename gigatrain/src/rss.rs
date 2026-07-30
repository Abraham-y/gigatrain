//! Stage-boundary RSS probe, enabled with GIGATRAIN_STATS=1.
//!
//! Shells out to `ps` rather than taking a libc dependency; it only runs a
//! handful of times per training run, gated behind the env var.

pub fn enabled() -> bool {
    std::env::var_os("GIGATRAIN_STATS").is_some()
}

/// Current resident set size in bytes, or None if unavailable.
pub fn current() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    // ps reports RSS in kilobytes.
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

pub fn report(stage: &str) {
    if !enabled() {
        return;
    }
    if let Some(bytes) = current() {
        eprintln!("rss after {stage}: {} MB", bytes / (1 << 20));
    }
}
