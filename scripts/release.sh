#!/usr/bin/env bash
set -euo pipefail

# Release script for Kaku
# Usage: ./scripts/release.sh
#
# Prerequisites:
#   - Clean git working tree on main branch
#   - gh CLI authenticated (for creating releases)
#   - Apple Developer ID certificate in login Keychain (or set KAKU_SIGNING_IDENTITY)
#   - Notarization credentials in Keychain or env vars (KAKU_NOTARIZE_*)
#
# Environment variables:
#   KAKU_SIGNING_IDENTITY    - Signing identity (auto-detected from Keychain if not set)
#   KAKU_NOTARIZE_APPLE_ID   - Apple ID for notarization
#   KAKU_NOTARIZE_TEAM_ID    - Team ID for notarization
#   KAKU_NOTARIZE_PASSWORD   - App-specific password for notarization
#   HOMEBREW_TAP_TOKEN       - Optional: GitHub token for Homebrew tap (defaults to gh auth token)
#   REQUIRE_HOMEBREW_TAP_UPDATE - Set to 0 to allow release to continue when tap update fails (default: 1)
#   RUN_CLIPPY               - Set to 1 to also run clippy (default: 0)
#   SKIP_TESTS               - Set to 1 to skip tests (default: 0)
#
# Resume flags (skip earlier stages after a mid-release failure):
#   --notarize-only  Skip build; notarize existing dist/Kaku.app, upload, tap.
#   --upload-only    Skip build + notarize; upload existing dist/Kaku.dmg + tap.
#   --tap-only       Only re-dispatch the Homebrew tap update.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Kaku"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/dist}"
PROFILE="${PROFILE:-release-opt}"
BUILD_ARCH="${BUILD_ARCH:-universal}"
RUN_CLIPPY="${RUN_CLIPPY:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
GITHUB_REPO="${GITHUB_REPO:-tw93/Kaku}"
HOMEBREW_TAP_REPO="${HOMEBREW_TAP_REPO:-tw93/homebrew-tap}"
REQUIRE_HOMEBREW_TAP_UPDATE="${REQUIRE_HOMEBREW_TAP_UPDATE:-1}"

# Resume stage flags.
SKIP_BUILD=0
SKIP_NOTARIZE=0
SKIP_UPLOAD=0

for arg in "$@"; do
    case "$arg" in
        --notarize-only) SKIP_BUILD=1 ;;
        --upload-only)   SKIP_BUILD=1; SKIP_NOTARIZE=1 ;;
        --tap-only)      SKIP_BUILD=1; SKIP_NOTARIZE=1; SKIP_UPLOAD=1 ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

die() {
    log_error "$*"
    exit 1
}

is_valid_team_id() {
    [[ "$1" =~ ^[A-Z0-9]{10}$ ]]
}

is_developer_id_application_identity() {
    [[ "$1" == Developer\ ID\ Application:* ]]
}

# Detect version from Cargo.toml if not provided
get_cargo_version() {
    grep '^version =' "$REPO_ROOT/kaku/Cargo.toml" | head -n1 | cut -d'"' -f2
}

# Verify git is clean
check_clean_git() {
    log_info "Checking git status..."
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
        git status
        die "Working tree is not clean. Commit or stash changes before releasing."
    fi

    # Check we're on main branch
    local branch
    branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$branch" != "main" ]]; then
        die "Not on main branch (currently on: $branch). Releases must be from main."
    fi
}

# Verify version consistency across crates
check_version_consistency() {
    log_info "Checking version consistency..."
    local kaku_version kaku_gui_version
    kaku_version=$(grep '^version =' "$REPO_ROOT/kaku/Cargo.toml" | head -n1 | cut -d'"' -f2)
    kaku_gui_version=$(grep '^version =' "$REPO_ROOT/kaku-gui/Cargo.toml" | head -n1 | cut -d'"' -f2)

    if [[ "$kaku_version" != "$kaku_gui_version" ]]; then
        die "Version mismatch: kaku=$kaku_version, kaku-gui=$kaku_gui_version"
    fi

    log_info "Version: $kaku_version"
}

# Check release notes match version
check_release_notes() {
    log_info "Checking release notes..."
    if [[ -x "$REPO_ROOT/scripts/check_release_notes.sh" ]]; then
        "$REPO_ROOT/scripts/check_release_notes.sh"
    else
        log_warn "check_release_notes.sh not found or not executable"
    fi
}

# Check config release metadata is ready
check_release_config() {
    log_info "Checking config release metadata..."
    if [[ ! -x "$REPO_ROOT/scripts/check_release_config.sh" ]]; then
        die "scripts/check_release_config.sh is missing or not executable"
    fi

    "$REPO_ROOT/scripts/check_release_config.sh"
}

extract_release_title() {
    local release_notes_file="$REPO_ROOT/.github/RELEASE_NOTES.md"
    local title

    if [[ ! -f "$release_notes_file" ]]; then
        return 1
    fi

    title=$(awk '/^# / { sub(/^# /, ""); print; exit }' "$release_notes_file")
    if [[ -z "$title" ]]; then
        return 1
    fi

    printf '%s\n' "$title"
}

# Check gh CLI is authenticated
check_gh_auth() {
    log_info "Checking GitHub CLI authentication..."
    if ! command -v gh >/dev/null 2>&1; then
        die "gh CLI not found. Install from https://cli.github.com/"
    fi

    if ! gh auth status >/dev/null 2>&1; then
        die "gh CLI not authenticated. Run: gh auth login"
    fi
}

# Detect Developer ID from Keychain if not set
detect_signing_identity() {
    if [[ -n "${KAKU_SIGNING_IDENTITY:-}" ]]; then
        if ! is_developer_id_application_identity "$KAKU_SIGNING_IDENTITY"; then
            die "KAKU_SIGNING_IDENTITY must be a Developer ID Application certificate, got: $KAKU_SIGNING_IDENTITY"
        fi
        log_info "Using signing identity from environment: $KAKU_SIGNING_IDENTITY"
        return 0
    fi

    log_info "Detecting signing identity from Keychain..."

    # Find Developer ID Application certificates
    local identities
    identities=$(security find-identity -v -p codesigning 2>/dev/null | grep "Developer ID Application" | awk -F '"' '{print $2}' || true)

    local count
    count=$(echo "$identities" | grep -c "^Developer ID Application" || echo "0")

    if [[ "$count" -eq 0 ]]; then
        die "No Developer ID Application certificate found in Keychain.\n" \
            "Install your certificate or set KAKU_SIGNING_IDENTITY environment variable."
    fi

    KAKU_SIGNING_IDENTITY=$(echo "$identities" | grep "^Developer ID Application" | head -n1)
    export KAKU_SIGNING_IDENTITY
    if [[ "$count" -gt 1 ]]; then
        log_warn "Multiple Developer ID Application certificates found, auto-selecting the first match"
    fi
    log_info "Auto-detected signing identity: $KAKU_SIGNING_IDENTITY"
}

validate_release_profile() {
    case "$PROFILE" in
        release|release-opt)
            log_info "Using release build profile: $PROFILE"
            ;;
        *)
            die "Invalid PROFILE=$PROFILE for release flow. Use PROFILE=release or PROFILE=release-opt."
            ;;
    esac
}

resolve_notarization_team_id() {
    local team_id="${KAKU_NOTARIZE_TEAM_ID:-}"

    if [[ -n "$team_id" ]]; then
        if is_valid_team_id "$team_id"; then
            return 0
        fi

        log_warn "Ignoring invalid KAKU_NOTARIZE_TEAM_ID: $team_id"
    fi

    team_id=$(printf '%s\n' "${KAKU_SIGNING_IDENTITY:-}" | sed -n 's/.*(\([A-Z0-9]\{10\}\)).*/\1/p')
    if [[ -n "$team_id" ]] && is_valid_team_id "$team_id"; then
        export KAKU_NOTARIZE_TEAM_ID="$team_id"
        log_info "Derived notarization Team ID from signing identity: $team_id"
        return 0
    fi

    return 1
}

# Check notarization credentials are available
check_notarization_creds() {
    log_info "Checking notarization credentials..."

    local have_creds=0
    local have_team_id=0

    if resolve_notarization_team_id; then
        have_team_id=1
    fi

    # Check environment variables
    if [[ -n "${KAKU_NOTARIZE_APPLE_ID:-}" && -n "${KAKU_NOTARIZE_PASSWORD:-}" && "$have_team_id" -eq 1 ]]; then
        have_creds=1
        log_info "Using notarization credentials from environment variables"
    else
        # Check Keychain
        local apple_id password
        apple_id=$(security find-generic-password -s "kaku-notarize-apple-id" -w 2>/dev/null || true)
        password=$(security find-generic-password -s "kaku-notarize-password" -w 2>/dev/null || true)

        if [[ -n "$apple_id" && -n "$password" && "$have_team_id" -eq 1 ]]; then
            have_creds=1
            log_info "Found notarization credentials in Keychain and resolved Team ID"
        fi
    fi

    if [[ "$have_creds" -eq 0 ]]; then
        log_warn "Notarization credentials not found in environment or Keychain"
        log_warn "Notarization may fail. To set up credentials:"
        log_warn "  export KAKU_NOTARIZE_APPLE_ID='your-apple-id@example.com'"
        log_warn "  export KAKU_NOTARIZE_TEAM_ID='YOURTEAMID'"
        log_warn "  export KAKU_NOTARIZE_PASSWORD='xxxx-xxxx-xxxx-xxxx'"
        log_warn ""
        log_warn "Or store in Keychain:"
        log_warn "  security add-generic-password -s 'kaku-notarize-apple-id' -a 'kaku' -w 'your-apple-id@example.com'"
        log_warn "  security add-generic-password -s 'kaku-notarize-password' -a 'kaku' -w 'your-app-specific-password'"
        read -r -p "Continue anyway? [y/N] " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Run all quality checks
run_checks() {
    log_info "Running format check..."
    make fmt-check

    log_info "Running compilation check..."
    make check

    if [[ "$RUN_CLIPPY" == "1" ]]; then
        log_info "Running clippy..."
        cargo clippy --locked --all-targets -- -D warnings
    fi

    if [[ "$SKIP_TESTS" == "0" ]]; then
        log_info "Running tests..."
        make test
    else
        log_warn "Skipping tests (SKIP_TESTS=1)"
    fi
}

# Build the release
build_release() {
    log_info "Building release (PROFILE=$PROFILE, ARCH=$BUILD_ARCH)..."

    export KAKU_SIGNING_IDENTITY
    export KAKU_REQUIRE_SIGNED_RELEASE=1
    export PROFILE
    export BUILD_ARCH
    export OUT_DIR

    ./scripts/build.sh
}

# Notarize the release
notarize_release() {
    log_info "Submitting for notarization..."
    ./scripts/notarize.sh
}

# Create and push git tag
create_tag() {
    local version="$1"
    local tag="V${version}"
    local head_sha
    local tag_sha
    local remote_tag_sha

    log_info "Creating tag $tag..."
    head_sha=$(git rev-parse HEAD)

    if git show-ref --verify --quiet "refs/tags/$tag"; then
        tag_sha=$(git rev-parse "$tag^{}")
        if [[ "$tag_sha" != "$head_sha" ]]; then
            die "Tag $tag already exists at $tag_sha, but HEAD is $head_sha."
        fi

        log_warn "Tag $tag already exists at current HEAD, reusing it."
    else
        git tag -a "$tag" -m "Release $tag"
    fi

    remote_tag_sha=$(git ls-remote --tags origin "refs/tags/${tag}^{}" | awk 'NR == 1 { print $1 }')
    if [[ -n "$remote_tag_sha" ]]; then
        if [[ "$remote_tag_sha" != "$head_sha" ]]; then
            die "Origin already has tag $tag at $remote_tag_sha, but HEAD is $head_sha."
        fi

        log_warn "Origin already has tag $tag at current HEAD, skipping push."
        return 0
    fi

    log_info "Pushing tag $tag..."
    git push origin "$tag"
}

# Create GitHub Release
create_github_release() {
    local version="$1"
    local tag="V${version}"
    local release_notes_file="$REPO_ROOT/.github/RELEASE_NOTES.md"
    local release_title="$APP_NAME $tag"
    local notes_arg=""
    local release_edit_args=()
    local release_title_from_notes=""

    # Build a cleaned notes file: strip the first heading line and remove blank
    # lines between numbered list items so GitHub doesn't render extra spacing.
    local notes_tmp
    notes_tmp=$(mktemp /tmp/kaku-release-notes.XXXXXX.md)
    # shellcheck disable=SC2064
    trap "rm -f $notes_tmp" RETURN

    if [[ -f "$release_notes_file" ]]; then
        # Skip leading "# Title" line (and following blank line), then collapse
        # blank lines that appear between numbered list items.
        awk '
            NR == 1 && /^# / { next }
            NR == 2 && /^[[:space:]]*$/ { next }
            /^[[:space:]]*$/ { blank=1; next }
            blank { if (!/^[0-9]+\./) printf "\n"; blank=0 }
            { print }
            END { if (blank) printf "\n" }
        ' "$release_notes_file" > "$notes_tmp"

        if [[ -s "$notes_tmp" ]]; then
            notes_arg="--notes-file"
        else
            notes_arg="--generate-notes"
        fi
    else
        notes_arg="--generate-notes"
    fi

    if release_title_from_notes=$(extract_release_title); then
        release_title="$release_title_from_notes"
    fi

    log_info "Creating GitHub Release for $tag..."

    if [[ "$notes_arg" == "--notes-file" ]]; then
        release_edit_args=(--title "$release_title" "$notes_arg" "$notes_tmp")
    else
        release_edit_args=(--title "$release_title")
    fi

    # Check if release already exists
    if gh release view "$tag" -R "$GITHUB_REPO" >/dev/null 2>&1; then
        log_warn "Release $tag already exists, reconciling title, notes, and assets..."
        gh release edit "$tag" \
            -R "$GITHUB_REPO" \
            "${release_edit_args[@]}"
        gh release upload "$tag" \
            -R "$GITHUB_REPO" \
            "$OUT_DIR/Kaku.dmg" \
            "$OUT_DIR/kaku_for_update.zip" \
            "$OUT_DIR/kaku_for_update.zip.sha256" \
            --clobber
    else
        if [[ "$notes_arg" == "--notes-file" ]]; then
            gh release create "$tag" \
                -R "$GITHUB_REPO" \
                "$OUT_DIR/Kaku.dmg" \
                "$OUT_DIR/kaku_for_update.zip" \
                "$OUT_DIR/kaku_for_update.zip.sha256" \
                --title "$release_title" \
                "$notes_arg" "$notes_tmp"
        else
            gh release create "$tag" \
                -R "$GITHUB_REPO" \
                "$OUT_DIR/Kaku.dmg" \
                "$OUT_DIR/kaku_for_update.zip" \
                "$OUT_DIR/kaku_for_update.zip.sha256" \
                --title "$release_title" \
                --generate-notes
        fi
    fi

    log_info "GitHub Release created: https://github.com/${GITHUB_REPO}/releases/tag/$tag"
}

# Optional: Update Homebrew tap
update_homebrew_tap() {
    local version="$1"
    local token=""
    local dmg_sha256
    local dispatch_output
    local workflow_url="https://github.com/${HOMEBREW_TAP_REPO}/actions/workflows/bump.yml"
    local latest_run_url=""

    # Try to get token: env var > gh auth token
    if [[ -n "${HOMEBREW_TAP_TOKEN:-}" ]]; then
        token="$HOMEBREW_TAP_TOKEN"
        log_info "Using HOMEBREW_TAP_TOKEN from environment"
    else
        # Try to get token from gh CLI
        token=$(gh auth token 2>/dev/null || true)
        if [[ -n "$token" ]]; then
            log_info "Using GitHub token from 'gh auth token'"
        fi
    fi

    if [[ -z "$token" ]]; then
        if [[ "$REQUIRE_HOMEBREW_TAP_UPDATE" == "1" ]]; then
            die "No GitHub token available for Homebrew tap update"
        fi
        log_warn "No GitHub token available, skipping Homebrew tap update"
        return 0
    fi

    dmg_sha256=$(shasum -a 256 "$OUT_DIR/Kaku.dmg" | awk '{print $1}')

    log_info "Dispatching Homebrew tap update..."

    # Dispatch workflow to update Homebrew tap
    if ! dispatch_output=$(
        GH_TOKEN="$token" gh api \
        --method POST \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/${HOMEBREW_TAP_REPO}/dispatches" \
        -f "event_type=kaku_release_published" \
        -f "client_payload[version]=$version" \
        -f "client_payload[sha256]=$dmg_sha256" 2>&1
    ); then
        log_warn "Failed to dispatch Homebrew tap update for ${HOMEBREW_TAP_REPO}"
        log_warn "$dispatch_output"
        if [[ "$REQUIRE_HOMEBREW_TAP_UPDATE" == "1" ]]; then
            die "Homebrew tap update dispatch failed. Track the workflow here: $workflow_url"
        fi
        log_warn "Track the workflow here: $workflow_url"
        return 0
    fi

    log_info "Homebrew tap update dispatched"
    log_info "Track the workflow here: $workflow_url"

    latest_run_url=$(gh run list \
        -R "$HOMEBREW_TAP_REPO" \
        --workflow bump.yml \
        --limit 1 \
        --json url,status,displayTitle,event \
        --jq '.[] | select(.displayTitle=="kaku_release_published" and .event=="repository_dispatch") | .url' 2>/dev/null || true)
    if [[ -n "$latest_run_url" ]]; then
        log_info "Latest Homebrew tap run: $latest_run_url"
    fi

    if [[ "$REQUIRE_HOMEBREW_TAP_UPDATE" == "1" ]]; then
        local expected_version="$version"
        local remote_version=""
        local attempt=0
        local max_attempts=12
        local sleep_seconds=15
        while (( attempt < max_attempts )); do
            attempt=$((attempt + 1))
            remote_version=$(gh api "repos/${HOMEBREW_TAP_REPO}/contents/Casks/kakuku.rb?ref=main" --jq '.download_url' 2>/dev/null \
                | xargs -I{} curl -fsSL --max-time 10 {} 2>/dev/null \
                | sed -n 's/^  version "\([^"]*\)"$/\1/p' | head -n1 || true)
            if [[ "$remote_version" == "$expected_version" ]]; then
                log_info "Homebrew tap verified at version ${remote_version}"
                break
            fi
            if [[ -z "$remote_version" ]]; then
                log_warn "Homebrew tap version check attempt ${attempt}/${max_attempts} returned empty result; waiting..."
            else
                log_info "Homebrew tap version check attempt ${attempt}/${max_attempts}: current=${remote_version}, expected=${expected_version}; waiting..."
            fi
            if (( attempt < max_attempts )); then
                sleep "$sleep_seconds"
            fi
        done
        if [[ "$remote_version" != "$expected_version" ]]; then
            die "Homebrew tap version verification timed out: expected ${expected_version}, got ${remote_version:-<empty>}"
        fi
    fi
}

# Main release flow
main() {
    local version

    log_info "Starting release process for $APP_NAME..."

    # Get version
    version=$(get_cargo_version)
    log_info "Releasing version: $version"

    # Always-on pre-flight (cheap and catches misconfiguration on resume too)
    check_clean_git
    check_version_consistency
    check_gh_auth

    if [[ "$SKIP_BUILD" -eq 0 ]]; then
        check_release_notes
        check_release_config
        validate_release_profile
        detect_signing_identity
        check_notarization_creds
        run_checks
        build_release
    else
        log_warn "Skipping build (resume flag)"
        if [[ ! -f "$OUT_DIR/Kaku.dmg" ]]; then
            die "Resume requires prior build; $OUT_DIR/Kaku.dmg not found."
        fi
        detect_signing_identity
    fi

    if [[ "$SKIP_NOTARIZE" -eq 0 ]]; then
        check_notarization_creds
        notarize_release
    else
        log_warn "Skipping notarize (resume flag)"
    fi

    if [[ "$SKIP_UPLOAD" -eq 0 ]]; then
        create_tag "$version"
        create_github_release "$version"
    else
        log_warn "Skipping upload (resume flag); tap dispatch only"
    fi

    update_homebrew_tap "$version"

    log_info "Release $version complete!"
    log_info "Artifacts:"
    log_info "  - $OUT_DIR/Kaku.dmg"
    log_info "  - $OUT_DIR/kaku_for_update.zip"
    log_info "  - $OUT_DIR/kaku_for_update.zip.sha256"
    log_info ""
    log_info "GitHub Release: https://github.com/${GITHUB_REPO}/releases/tag/V${version}"
}

main "$@"
