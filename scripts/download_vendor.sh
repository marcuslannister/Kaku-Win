#!/usr/bin/env bash
set -euo pipefail

# This script downloads plugin dependencies bundled into the Kaku App.
# CLI tools (starship/git-delta/lazygit) are installed via Homebrew at init time.

VENDOR_DIR="$(cd "$(dirname "$0")/../assets/vendor" && pwd)"
mkdir -p "$VENDOR_DIR"

download_pinned_repo() {
	local step="$1"
	local name="$2"
	local repo="$3"
	local ref="$4"
	local dest="$VENDOR_DIR/$name"
	local marker_file="$dest/.kaku-vendor-ref"
	local archive_url="https://codeload.github.com/$repo/tar.gz/$ref"
	local temp_dir
	local extract_dir
	local archive_path
	local source_dir

	echo "[$step/4] Syncing $name @ $ref..."
	if [[ -f "$marker_file" ]] && [[ "$(cat "$marker_file")" == "$ref" ]]; then
		echo "$name already pinned to $ref, skipping."
		return
	fi

	temp_dir="$(mktemp -d)"
	trap 'rm -rf "$temp_dir"' RETURN
	extract_dir="$temp_dir/extract"
	archive_path="$temp_dir/$name.tar.gz"
	mkdir -p "$extract_dir"

	curl --fail --location --silent --show-error --retry 3 --retry-delay 2 "$archive_url" --output "$archive_path"
	tar -xzf "$archive_path" -C "$extract_dir"
	source_dir="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
	if [[ -z "$source_dir" ]]; then
		echo "Failed to unpack $name from $archive_url" >&2
		exit 1
	fi

	rm -rf "$dest"
	mv "$source_dir" "$dest"
	printf '%s\n' "$ref" > "$marker_file"
	trap - RETURN
	rm -rf "$temp_dir"
}

echo "[0/4] Cleaning legacy vendor binaries..."
rm -f "$VENDOR_DIR/starship" "$VENDOR_DIR/delta" "$VENDOR_DIR/zoxide"
rm -rf "$VENDOR_DIR/completions" "$VENDOR_DIR/man"
rm -f "$VENDOR_DIR/README.md" "$VENDOR_DIR/CHANGELOG.md" "$VENDOR_DIR/LICENSE"

# Pin external shell integrations to exact commits so app/release artifacts stay reproducible.
download_pinned_repo "1" "zsh-autosuggestions" "zsh-users/zsh-autosuggestions" "85919cd1ffa7d2d5412f6d3fe437ebdbeeec4fc5"
download_pinned_repo "2" "zsh-syntax-highlighting" "zsh-users/zsh-syntax-highlighting" "1d85c692615a25fe2293bdd44b34c217d5d2bf04"
download_pinned_repo "3" "zsh-completions" "zsh-users/zsh-completions" "84615f3d0b0e943d5b1de862c9552e572c8e70bb"
download_pinned_repo "4" "zsh-z" "agkozak/zsh-z" "cf9225feebfae55e557e103e95ce20eca5eff270"

echo "Vendor dependencies downloaded to $VENDOR_DIR"
