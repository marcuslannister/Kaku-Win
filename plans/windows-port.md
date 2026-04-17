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

## Phase 1 — GPU Backend

**File:** `Cargo.toml` line 287
**Effort:** ~30 min

Change wgpu features to include DX12:

```toml
# Before
wgpu = { version = "25.0.2", default-features = false, features = ["metal", "wgsl"] }

# After
wgpu = { version = "25.0.2", default-features = false, features = ["metal", "dx12", "wgsl"] }
```

DX12 is the most stable wgpu backend on Windows 10+.

**Validation:** `cargo check -p kaku-gui --target x86_64-pc-windows-msvc` passes without wgpu errors.

---

## Phase 2 — Win32 Window Backend

**Files:** `window/src/os/windows/` (new, ~40 files cherry-picked), `window/src/os/mod.rs`
**Effort:** 1–2 weeks

### 2.1 — Find the WezTerm divergence commit

```bash
git log --oneline | tail -1
# or: git merge-base HEAD upstream/main
```

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

### 2.4 — Fix compile errors

Common issues to expect:
- Cocoa-specific types in `window/src/lib.rs` → gate behind `#[cfg(target_os = "macos")]`
- `window/Cargo.toml` already has `[target.'cfg(windows)'.dependencies]` with `winapi` — verify versions match

**Validation:** `cargo build -p window --target x86_64-pc-windows-msvc` clean build.

---

## Phase 3 — Config Paths

**Files:** `config/src/config.rs`, `kaku/src/assistant_config.rs`, `kaku/src/doctor.rs`, `kaku/src/reset.rs`
**Effort:** ~2 hours

`dirs-next` is already in the workspace. Replace all hardcoded `~/.config/kaku` paths with runtime resolution.

### 3.1 — `config/src/config.rs` (lines 1085, 1954)

Add `#[cfg(windows)]` branch using `dirs_next::config_dir()` which returns `%APPDATA%\kaku` on Windows.

### 3.2 — CLI helpers

In `kaku/src/assistant_config.rs`, `kaku/src/doctor.rs`, `kaku/src/reset.rs`:  
Replace `home_dir().join(".config").join("kaku")` with:

```rust
dirs_next::config_dir().unwrap().join("kaku")
```

### 3.3 — Config file path

`kaku.lua` follows the same pattern: `%APPDATA%\kaku\kaku.lua` on Windows.

**Validation:** On Windows, `kaku config open` opens `%APPDATA%\kaku\kaku.lua`.

---

## Phase 4 — Gate macOS-Only CLI Commands

**Files:** 8 source files
**Effort:** ~1 week

| File | macOS-only concern | Windows action |
|------|--------------------|----------------|
| `kaku/src/update.rs` | `brew upgrade kaku`, DMG download | Gate; print "Use your package manager" on Windows |
| `kaku/src/kaku_theme.rs` | `defaults write` for macOS appearance | Gate; add Windows registry dark mode detection |
| `kaku/src/init.rs` | `.zshrc` setup | Gate; add PowerShell profile setup stub |
| `kaku/src/doctor.rs` | zsh/fish checks, Homebrew, macOS proxy detection | Gate the 6 macOS-specific check functions; keep cross-platform checks |
| `kaku/src/reset.rs` | Removes `~/.config/kaku/zsh/`, `.zshrc` edits | Gate zsh cleanup; keep config dir removal |
| `kaku-gui/src/commands.rs` | macOS menu items (lines 649, 2519, 2521) | Already gated — verify nothing leaks |
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

**Files:** `assets/shell-integration/powershell/kaku.ps1` (new), minor gates in `kaku-gui/` and `config/`
**Effort:** ~2 hours

### 5.1 — Skip bundled zsh suite on Windows

Gate all bundled plugin injection in `kaku-gui/src/` and `config/src/` behind `#[cfg(not(windows))]`.

### 5.2 — PowerShell stub

Create `assets/shell-integration/powershell/kaku.ps1`:

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
| Exact WezTerm commit this fork diverged from | Developer — run `git log` | Needed before Phase 2 to assess Win32 backend drift |
| DX12 adapter availability in GitHub Actions `windows-latest` | CI run will reveal | May need software rasterizer fallback in tests |
| PowerShell profile injection location | Deferred to full shell integration milestone | Out of scope for MVP |

---

## Impact Summary

- **Source files changed:** ~15 existing files
- **New files:** `scripts/build-windows.ps1`, `assets/shell-integration/powershell/kaku.ps1`
- **Cherry-picked files:** ~40 files under `window/src/os/windows/`
- **Rollback safety:** Every change is behind `#[cfg(windows)]` or additive — macOS build is unaffected at any point
