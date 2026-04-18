#!/usr/bin/env pwsh
# Windows release build for Kaku.
# Produces target\x86_64-pc-windows-msvc\release\{kaku,kaku-gui}.exe
# No bundling, no code-signing, no notarization.

$ErrorActionPreference = "Stop"

cargo build --locked --release -p kaku -p kaku-gui --target x86_64-pc-windows-msvc
