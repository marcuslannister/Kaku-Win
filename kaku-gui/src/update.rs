use anyhow::anyhow;
use config::proxy::{apply_to_command, detect_system_proxy};
use config::{configuration, wezterm_version};
use serde::*;
use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use wezterm_toast_notification::*;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Release {
    pub url: String,
    pub body: String,
    pub html_url: String,
    pub tag_name: String,
    pub assets: Vec<Asset>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Asset {
    pub name: String,
    pub size: usize,
    pub url: String,
    pub browser_download_url: String,
}

fn curl_get_release_json(url: &str, proxy: &Option<String>) -> anyhow::Result<Release> {
    use std::process::Command;

    let mut cmd = Command::new("/usr/bin/curl");
    cmd.arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg("--connect-timeout")
        .arg("15")
        .arg("--user-agent")
        .arg(format!("kaku/{}", wezterm_version()))
        .arg(url);
    apply_to_command(&mut cmd, proxy);

    let out = cmd.output().map_err(|e| anyhow!("curl failed: {}", e))?;
    if !out.status.success() {
        anyhow::bail!(
            "curl request failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    serde_json::from_slice(&out.stdout).map_err(|e| anyhow!("failed to parse release JSON: {}", e))
}

pub fn get_latest_release_info() -> anyhow::Result<Release> {
    let proxy = detect_system_proxy();
    curl_get_release_json(
        "https://api.github.com/repos/tw93/Kaku/releases/latest",
        &proxy,
    )
    .or_else(|_| get_latest_tag_via_redirect(&proxy))
}

fn get_latest_tag_via_redirect(proxy: &Option<String>) -> anyhow::Result<Release> {
    use std::process::Command;

    let mut cmd = Command::new("/usr/bin/curl");
    cmd.arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg("--connect-timeout")
        .arg("10")
        .arg("--write-out")
        .arg("%{url_effective}")
        .arg("--output")
        .arg("/dev/null")
        .arg("https://github.com/tw93/Kaku/releases/latest");
    apply_to_command(&mut cmd, proxy);

    let output = cmd.output().map_err(|e| anyhow!("curl failed: {}", e))?;

    if !output.status.success() {
        anyhow::bail!("curl returned non-zero status");
    }

    let effective_url = String::from_utf8_lossy(&output.stdout);
    let tag = effective_url
        .trim()
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("failed to extract tag from URL"))?;

    Ok(Release {
        url: String::new(),
        body: String::new(),
        html_url: "https://github.com/tw93/Kaku/releases/latest".to_string(),
        tag_name: tag.to_string(),
        assets: vec![],
    })
}

#[allow(unused)]
pub fn get_nightly_release_info() -> anyhow::Result<Release> {
    let proxy = detect_system_proxy();
    curl_get_release_json(
        "https://api.github.com/repos/wezterm/wezterm/releases/tags/nightly",
        &proxy,
    )
}

fn is_newer(latest: &str, current: &str) -> bool {
    let latest = latest.trim_start_matches('v');
    let current = current.trim_start_matches('v');

    // If latest is a WezTerm-style date version (e.g. 20240203-...) and current is SemVer (e.g. 0.1.0),
    // treat the date version as older/different system.
    if latest.starts_with("20") && latest.contains('-') && !current.starts_with("20") {
        return false;
    }

    match compare_versions(latest, current) {
        Some(CmpOrdering::Greater) => true,
        Some(_) => false,
        None => latest != current,
    }
}

fn compare_versions(left: &str, right: &str) -> Option<CmpOrdering> {
    let left = parse_version_numbers(left)?;
    let right = parse_version_numbers(right)?;
    let max_len = left.len().max(right.len());
    for idx in 0..max_len {
        let l = left.get(idx).copied().unwrap_or(0);
        let r = right.get(idx).copied().unwrap_or(0);
        match l.cmp(&r) {
            CmpOrdering::Equal => {}
            non_eq => return Some(non_eq),
        }
    }
    Some(CmpOrdering::Equal)
}

fn parse_version_numbers(version: &str) -> Option<Vec<u64>> {
    let cleaned = version.trim().trim_start_matches(['v', 'V']);
    let mut out = Vec::new();
    for part in cleaned.split('.') {
        let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        let value = digits.parse::<u64>().ok()?;
        out.push(value);
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

fn update_checker() {
    log::info!("update_checker thread started");

    let initial_interval = Duration::from_secs(3);
    let force_ui = std::env::var_os("KAKU_ALWAYS_SHOW_UPDATE_UI").is_some();

    let update_file_name = config::DATA_DIR.join("check_update");

    // Check if we already know about a newer version from the cached file.
    // If so, show notification immediately without waiting.
    // Respect check_for_updates so disabled users don't get startup notifications.
    if configuration().check_for_updates {
        if let Ok(content) = std::fs::read_to_string(&update_file_name) {
            if let Ok(cached_release) = serde_json::from_str::<Release>(&content) {
                let current = wezterm_version();
                if is_newer(&cached_release.tag_name, current) {
                    log::info!(
                        "update_checker: cached release {} is newer than current {}, showing notification",
                        cached_release.tag_name,
                        current
                    );
                    std::thread::sleep(initial_interval);
                    let my_sock =
                        config::RUNTIME_DIR.join(format!("gui-sock-{}", unsafe { libc::getpid() }));
                    let socks = wezterm_client::discovery::discover_gui_socks();
                    if force_ui || socks.is_empty() || socks.first() == Some(&my_sock) {
                        persistent_toast_notification_with_click_to_open_url(
                            "Update Available",
                            &format!("{} is available. Click to update.", cached_release.tag_name),
                            "kaku://update",
                        );
                    }
                }
            }
        }
    }

    // Compute how long we should sleep for;
    // if we've never checked, give it a few seconds after the first
    // launch, otherwise compute the interval based on the time of
    // the last check.
    let update_interval = Duration::from_secs(configuration().check_for_updates_interval_seconds);

    let delay = update_file_name
        .metadata()
        .and_then(|metadata| metadata.modified())
        .map_err(|_| ())
        .and_then(|systime| {
            let elapsed = systime.elapsed().unwrap_or(Duration::new(0, 0));
            update_interval.checked_sub(elapsed).ok_or(())
        })
        .unwrap_or(initial_interval);

    log::info!(
        "update_checker: sleeping for {:?}",
        if force_ui { initial_interval } else { delay }
    );
    std::thread::sleep(if force_ui { initial_interval } else { delay });
    log::info!("update_checker: woke up, starting check loop");

    let my_sock = config::RUNTIME_DIR.join(format!("gui-sock-{}", unsafe { libc::getpid() }));

    loop {
        // Figure out which other wezterm-guis are running.
        // We have a little "consensus protocol" to decide which
        // of us will show the toast notification or show the update
        // window: the one of us that sorts first in the list will
        // own doing that, so that if there are a dozen gui processes
        // running, we don't spam the user with a lot of notifications.
        let socks = wezterm_client::discovery::discover_gui_socks();

        log::info!(
            "update_checker: check_for_updates={}",
            configuration().check_for_updates
        );
        if configuration().check_for_updates {
            log::info!("update_checker: fetching release info...");
            match get_latest_release_info() {
                Ok(latest) => {
                    log::info!("update_checker: got release {}", latest.tag_name);
                    let current = wezterm_version();
                    if is_newer(&latest.tag_name, current) || force_ui {
                        log::info!(
                            "latest release {} is newer than current build {}",
                            latest.tag_name,
                            current
                        );

                        log::info!("update_checker: socks={:?}, my_sock={:?}", socks, my_sock);
                        if force_ui || socks.is_empty() || socks[0] == my_sock {
                            log::info!("update_checker: showing notification");
                            persistent_toast_notification_with_click_to_open_url(
                                "Update Available",
                                &format!("{} is available. Click to update.", latest.tag_name),
                                "kaku://update",
                            );
                        } else {
                            log::info!(
                                "update_checker: skipping notification (not primary instance)"
                            );
                        }
                    }

                    if let Some(parent) = update_file_name.parent() {
                        config::create_user_owned_dirs(parent).ok();
                    }

                    // Record the time of this check
                    if let Ok(f) = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&update_file_name)
                    {
                        serde_json::to_writer_pretty(f, &latest).ok();
                    }
                }
                Err(e) => {
                    log::warn!("update_checker: failed to get release info: {}", e);
                }
            }
        }

        std::thread::sleep(Duration::from_secs(
            configuration().check_for_updates_interval_seconds,
        ));
    }
}

pub fn start_update_checker() {
    static CHECKER_STARTED: AtomicBool = AtomicBool::new(false);
    if let Ok(false) =
        CHECKER_STARTED.compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
    {
        // Initialize the notification system early so macOS shows the permission
        // dialog on first launch, rather than lazily when a notification fires.
        #[cfg(target_os = "macos")]
        wezterm_toast_notification::macos_initialize();

        // Register callback so notification clicks open update in a new tab
        // instead of spawning a separate window process.
        wezterm_toast_notification::set_update_click_callback(|| {
            crate::frontend::run_kaku_update_from_menu();
        });

        // Check if we just completed an update and show notification
        check_update_completed();

        std::thread::Builder::new()
            .name("update_checker".into())
            .spawn(update_checker)
            .expect("failed to spawn update checker thread");
    }
}

fn check_update_completed() {
    let marker_file = config::DATA_DIR.join("update_completed");
    if !marker_file.exists() {
        return;
    }

    // Check if marker file is recent (within last 5 minutes)
    // This prevents showing stale notifications from old failed updates
    let is_recent = marker_file
        .metadata()
        .and_then(|m| m.modified())
        .map(|t| t.elapsed().map(|e| e.as_secs() < 300).unwrap_or(false))
        .unwrap_or(false);

    if is_recent {
        if let Ok(version) = std::fs::read_to_string(&marker_file) {
            let version = version.trim();
            if !version.is_empty() {
                log::info!("update_completed: showing notification for {}", version);
                wezterm_toast_notification::persistent_toast_notification(
                    "Updated",
                    &format!("Successfully updated to {}.", version),
                );
            }
        }
    } else {
        log::info!("update_completed: skipping stale marker file");
    }

    // Always remove the marker file
    let _ = std::fs::remove_file(&marker_file);
}

#[cfg(test)]
mod tests {
    use super::is_newer;

    #[test]
    fn semver_numeric_comparison() {
        assert!(is_newer("0.1.10", "0.1.9"));
        assert!(!is_newer("0.2.0", "0.11.0"));
        assert!(!is_newer("0.1.1", "0.1.1"));
        assert!(is_newer("v0.1.2", "0.1.1"));
    }
}
