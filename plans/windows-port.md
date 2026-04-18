# Windows Port Plan

Port Kaku from macOS-only to native Windows 10/11 by restoring the Win32 window backend and DX12 GPU path that were gated off from the upstream WezTerm fork.

## Scope

**Building:**
- `Kaku.exe` running natively on Windows 10/11
- DX12 GPU-accelerated rendering
- AI panel and CLI commands on Windows
- Dark/light mode detection via Windows registry

**Not building:**
- Bundled shell suite on Windows (no zsh, yazi, lazygit auto-install)
- Windows code-signing / notarization pipeline
- Linux support
- MSI/MSIX installer (plain `.exe` first)
- Full PowerShell shell integration (stub only)

---

## Phase 1 — GPU Backend ✅ ALREADY DONE

`Cargo.toml:287` already declares:

```toml
# Metal for macOS, DX12 for Windows
wgpu = { version = "25.0.2", default-features = false, features = ["metal", "dx12", "wgsl"] }
```

No action required. **Validation** (still worth running at the end of Phase 2): `cargo check -p kaku-gui --target x86_64-pc-windows-msvc` passes without wgpu errors.

---

## Phase 2 — Win32 Window Backend

**Files:** `window/src/os/windows/` (new, ~40 files cherry-picked), `window/src/os/mod.rs`
**Effort:** 1–2 weeks

### 2.1 — Add upstream WezTerm remote and find divergence commit

No `upstream` remote exists in this fork (`git remote -v` shows only `origin`). Add it first, then locate the divergence point:

```bash
git remote add upstream https://github.com/wezterm/wezterm.git
git fetch upstream
git merge-base HEAD upstream/main
```

Use the returned SHA as the baseline for cherry-picking in 2.2.

### 2.2 — Cherry-pick Win32 backend from WezTerm

Copy `window/src/os/windows/` from the identified WezTerm commit. Key files:

| File | Purpose |
|------|---------|
| `connection.rs` | Win32 event loop |
| `window.rs` | HWND creation, resize, input handling |
| `keycodes.rs` | VK_ → `KeyCode` translation |
| `clipboard.rs` | `CF_UNICODETEXT` clipboard |
| `menu.rs` | HMENU for system menu |

### 2.3 — Wire into `window/src/os/mod.rs`

```rust
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use self::macos::*;

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use self::windows::*;

pub mod parameters;
```

### 2.4 — Declare Windows dependencies for the `window` crate

`window/Cargo.toml` has **no** `[target.'cfg(windows)'.dependencies]` section today — it must be added fresh. The root workspace already lists `winapi = "0.3.9"`, `windows = "0.33.0"`, and `winreg = "0.10"`, but they are unused by the `window` crate.

Add to `window/Cargo.toml`:

```toml
[target.'cfg(windows)'.dependencies]
winapi = { workspace = true, features = [/* per cherry-picked code */] }
windows = { workspace = true }
winreg = { workspace = true }
```

The `windows = "0.33.0"` entry at root currently declares **no features**. Cherry-picked code will require explicit features; likely set:

```toml
windows = { version = "0.33.0", features = [
    "Win32_Foundation",
    "Win32_Graphics_Direct3D12",
    "Win32_Graphics_Dxgi_Common",
    "Win32_UI_WindowsAndMessaging",
    "Win32_UI_Input_KeyboardAndMouse",
    "Win32_System_LibraryLoader",
    "Win32_System_Registry",
    "Win32_System_Threading",
] }
```

Refine once the cherry-picked code compiles.

### 2.5 — Fix compile errors

Common issues to expect:
- Cocoa-specific types in `window/src/lib.rs` → gate behind `#[cfg(target_os = "macos")]`
- `winapi` feature list mismatch between the cherry-picked Win32 code and this fork's version

### 2.6 — Audit existing `cfg(windows)` islands for conflicts

This fork is **not** fully stripped of Windows code: 155 `cfg(windows)` hits across 51 files remain. Before declaring Phase 2 done, audit the heaviest hotspots for conflicts with the cherry-picked backend:

| File | Hits | Risk |
|------|------|------|
| `crates/wezterm-input-types/src/lib.rs` | 54 | VK_ keycode translations may duplicate cherry-picked `window/src/os/windows/keycodes.rs` |
| `crates/wezterm-uds/src/lib.rs` | 8 | UDS has Windows shim — verify still needed |
| `crates/pty/src/lib.rs` | 7 | ConPTY path — confirm it still compiles |
| `mux/src/localpane.rs` | 4 | Process spawning on Windows |
| `window/src/spawn.rs`, `window/src/egl.rs` | 3 each | May collide with cherry-picked wiring |

**Validation:** `cargo build -p window --target x86_64-pc-windows-msvc` clean build.

---

## Phase 3 — Config Paths

**Files:** `config/src/lib.rs`, `kaku-gui/src/frontend.rs`, `kaku/src/doctor.rs`, `kaku/src/reset.rs`
**Effort:** ~2 hours

### Problem

`config/src/lib.rs:93` defines `HOME_DIR` via `dirs_next::home_dir()`. Callers across the workspace build the config path as:

```rust
config::HOME_DIR.join(".config").join("kaku")
```

On Windows this resolves to `C:\Users\X\.config\kaku`, not the conventional `%APPDATA%\kaku`. `dirs-next` is already a workspace dep (root `Cargo.toml:99`), so the fix is to introduce a single helper and sweep callers — not patch per-file path strings.

### 3.1 — Introduce `KAKU_CONFIG_DIR` in `config/src/lib.rs`

Alongside `HOME_DIR` at `config/src/lib.rs:93`, add:

```rust
pub static ref KAKU_CONFIG_DIR: PathBuf = {
    #[cfg(windows)]
    { dirs_next::config_dir().expect("can't find config dir").join("kaku") }
    #[cfg(not(windows))]
    { HOME_DIR.join(".config").join("kaku") }
};
```

Export it from the crate root. On Windows this yields `%APPDATA%\kaku`; unchanged elsewhere.

### 3.2 — Sweep callers

Replace every `HOME_DIR.join(".config").join("kaku")` with `KAKU_CONFIG_DIR.clone()` (or `&*KAKU_CONFIG_DIR`). Known sites:

- `kaku-gui/src/frontend.rs:85-91` — shell-integration candidate path (will also be gated in Phase 5)
- `kaku/src/doctor.rs` — path construction inside the 28 user-facing message sites; most messages are string literals, but any `.join(...)` chains should switch
- `kaku/src/reset.rs` — config-dir removal (the zsh subtree removal stays macOS-only; the top-level kaku config dir moves to the helper)
- `kaku/src/assistant_config.rs` — assistant config file lookup

Run `rg '\.config.*kaku' --type rust` after the sweep; only the new helper definition should remain.

### 3.3 — `kaku.lua` location

`kaku.lua` follows the same pattern automatically once lookups go through `KAKU_CONFIG_DIR`: `%APPDATA%\kaku\kaku.lua` on Windows.

**Validation:** On Windows, `kaku config open` opens `%APPDATA%\kaku\kaku.lua`. On macOS, `~/.config/kaku/kaku.lua` still opens as before (rollback safety).

---

## Phase 4 — Implement Windows Branches of Platform-Gated CLI

**Files:** 8 source files
**Effort:** ~1 week

Every CLI file in scope **already has** `#[cfg(target_os = "macos")]` branches — macOS gating is not the work. The work is filling the currently-empty `#[cfg(not(target_os = "macos"))]` / `#[cfg(windows)]` arms with Windows-appropriate behavior.

| File | Existing macOS gates | Windows branch to add |
|------|----------------------|------------------------|
| `kaku/src/update.rs` (lines 13, 22) | `brew upgrade kaku`, DMG download | Print "Use your package manager" on Windows |
| `kaku/src/kaku_theme.rs` (lines 26, 28, 290, 317) | `defaults write` for appearance | Windows registry dark-mode read/write |
| `kaku/src/init.rs` (lines 21, 30) | `.zshrc` setup | PowerShell profile setup stub |
| `kaku/src/doctor.rs` (lines 591, 601) | zsh/fish checks, Homebrew, macOS proxy detection | Windows-equivalent checks (PowerShell profile, registry proxy) |
| `kaku/src/reset.rs` (lines 21, 30) | `~/.config/kaku/zsh/` + `.zshrc` edits | Keep config-dir removal (now via `KAKU_CONFIG_DIR`); skip shell cleanup |
| `kaku-gui/src/commands.rs` | macOS menu items (lines 649, 2519, 2521) | Already correctly gated — verify nothing leaks |
| `kaku-gui/src/termwindow/mod.rs` | 6 macOS-specific methods | Already gated — verify |
| `kaku-gui/src/frontend.rs` | macOS app delegate calls | Already gated — verify |

### Dark/light mode on Windows

`kaku/src/kaku_theme.rs` needs a `#[cfg(windows)]` branch reading:

```
HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize\AppsUseLightTheme
```

via the `windows` crate (already in workspace).

---

## Phase 5 — Shell Integration on Windows

**Files:** `assets/shell-integration/powershell/kaku.ps1` (new), `kaku-gui/src/frontend.rs`
**Effort:** ~2 hours

### 5.1 — Skip bundled zsh suite on Windows

Only one Rust site injects a zsh-bundle path; `config/src/` has no shell-integration references (the plan's earlier claim was wrong).

- `kaku-gui/src/frontend.rs:85-91` — wraps `HOME_DIR/.config/kaku/zsh/bin/kaku` into the PATH candidates list. Gate the whole `add_candidate(...)` call behind `#[cfg(not(windows))]` and (per Phase 3) switch the base path to `KAKU_CONFIG_DIR`.

The bulk of zsh injection happens in shell scripts (`scripts/build.sh:259, 263` copies plugin submodules into the app bundle). Windows builds skip `build.sh` entirely, so no gate is needed there — but the new `build-windows.ps1` (Phase 6) simply must not replicate those copies.

### 5.2 — PowerShell stub

Create `assets/shell-integration/powershell/kaku.ps1` (directory is new):

```powershell
$env:TERM_PROGRAM = "Kaku"
# Optional: source user extensions from $env:APPDATA\kaku\powershell\
```

---

## Phase 6 — Build Script

**Files:** `scripts/build-windows.ps1` (new), `Makefile`
**Effort:** ~2 hours

### 6.1 — `scripts/build-windows.ps1`

```powershell
cargo build --locked --release -p kaku -p kaku-gui --target x86_64-pc-windows-msvc
```

No `lipo`, no `.app` bundle, no notarization.  
Output: `target/x86_64-pc-windows-msvc/release/kaku.exe` and `kaku-gui.exe`.

### 6.2 — `Makefile`

Add:
```makefile
build-windows:
	powershell -File scripts/build-windows.ps1
```

---

## Phase 7 — CI

**File:** `.github/workflows/ci.yml`
**Effort:** ~1 hour

Add a `windows-build` job (non-blocking initially):

```yaml
windows-build:
  runs-on: windows-latest
  continue-on-error: true   # flip to false once stable
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: "1.93.0"
        targets: "x86_64-pc-windows-msvc"
    - run: cargo build --locked -p kaku -p kaku-gui
    - run: cargo nextest run --locked
```

---

## Unknowns (Deferred)

| Unknown | Owner | Why deferred |
|---------|-------|--------------|
| Exact WezTerm commit this fork diverged from | Developer — add `upstream` remote then `git merge-base` (see Phase 2.1) | Needed before Phase 2.2 to assess Win32 backend drift |
| DX12 adapter availability in GitHub Actions `windows-latest` | CI run will reveal | May need software rasterizer fallback in tests |
| PowerShell profile injection location | Deferred to full shell integration milestone | Out of scope for MVP |
| Final `windows = "0.33.0"` feature list | Discovered during Phase 2.5 compile fixes | Starter list in Phase 2.4; refine as errors dictate |

---

## Impact Summary

- **Source files changed:** ~15 existing files
- **New files:** `scripts/build-windows.ps1`, `assets/shell-integration/powershell/kaku.ps1`
- **Cherry-picked files:** ~40 files under `window/src/os/windows/`
- **New API:** `config::KAKU_CONFIG_DIR` (Phase 3) replaces the `HOME_DIR.join(".config").join("kaku")` pattern everywhere
- **Existing Windows islands to audit:** 51 files, 155 `cfg(windows)` hits — hotspots tabled in Phase 2.6
- **Rollback safety:** Every change is behind `#[cfg(windows)]` or additive — macOS build is unaffected at any point
