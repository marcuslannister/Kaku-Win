#!/usr/bin/env bash
set -euo pipefail

# Notarization script for Kaku macOS app
# Usage: ./scripts/notarize.sh [--staple-only]
#
# Prerequisites:
# 1. App must be signed with Developer ID
# 2. Preferred: App Store Connect API Key (rcodesign, avoids notarytool SIGBUS on macOS 26):
#    - Store the JSON key path in Keychain: security add-generic-password -s "kaku-asc-api-key-path" -a "kaku" -w "/path/to/asc_api_key.json"
#    - Generate with: rcodesign encode-app-store-connect-api-key -o asc_api_key.json <issuer-id> <key-id> AuthKey_*.p8
# 3. Fallback: Apple ID + app-specific password via Keychain:
#    - KAKU_NOTARIZE_APPLE_ID, KAKU_NOTARIZE_TEAM_ID, KAKU_NOTARIZE_PASSWORD

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Kaku"
OUT_DIR="${OUT_DIR:-dist}"
APP_BUNDLE="${OUT_DIR}/${APP_NAME}.app"
DMG_PATH="${OUT_DIR}/${APP_NAME}.dmg"
NOTARY_SUBMIT_MAX_ATTEMPTS="${NOTARY_SUBMIT_MAX_ATTEMPTS:-3}"
NOTARY_SUBMIT_RETRY_DELAY="${NOTARY_SUBMIT_RETRY_DELAY:-20}"

STAPLE_ONLY=0
for arg in "$@"; do
	case "$arg" in
	--staple-only) STAPLE_ONLY=1 ;;
	esac
done

is_valid_team_id() {
	[[ "$1" =~ ^[A-Z0-9]{10}$ ]]
}

require_developer_id_signature() {
	local metadata
	local signed_team_id

	metadata=$(codesign -dvvvv "$APP_BUNDLE" 2>&1) || {
		echo "Error: failed to inspect app signature." >&2
		return 1
	}

	if ! grep -q "^Authority=Developer ID Application:" <<<"$metadata"; then
		echo "Error: App must be signed with a Developer ID Application certificate before notarization." >&2
		echo "Rebuild with ./scripts/build.sh after installing a single Developer ID Application certificate, or set KAKU_SIGNING_IDENTITY explicitly." >&2
		echo "$metadata" | grep -E "^(Authority=|TeamIdentifier=|Signature=)" >&2 || true
		return 1
	fi

	signed_team_id=$(echo "$metadata" | awk -F= '/^TeamIdentifier=/{print $2; exit}')
	if ! is_valid_team_id "$signed_team_id"; then
		echo "Error: App signature does not contain a valid TeamIdentifier." >&2
		echo "$metadata" | grep -E "^(Authority=|TeamIdentifier=|Signature=)" >&2 || true
		return 1
	fi
}

# Check if app exists
if [[ ! -d "$APP_BUNDLE" ]]; then
	echo "Error: $APP_BUNDLE not found. Run ./scripts/build.sh first."
	exit 1
fi

# Verify signing
if ! codesign -v "$APP_BUNDLE" 2>/dev/null; then
	echo "Error: App is not signed. Re-run ./scripts/build.sh with a Developer ID Application certificate available."
	exit 1
fi

require_developer_id_signature || exit 1

echo "App: $APP_BUNDLE"
echo "DMG: $DMG_PATH"

# Resolve submission path
if [[ -f "$DMG_PATH" ]]; then
	SUBMISSION_PATH="$DMG_PATH"
else
	SUBMISSION_PATH="$APP_BUNDLE"
fi

if [[ "$STAPLE_ONLY" == "1" ]]; then
	echo "Stapling existing notarization ticket..."
	xcrun stapler staple "$APP_BUNDLE"
	[[ -f "$DMG_PATH" ]] && xcrun stapler staple "$DMG_PATH"
	echo "✅ Staple complete!"
	echo ""
	echo "Verifying notarization:"
	spctl -a -vv "$APP_BUNDLE" 2>&1 || true
	exit 0
fi

staple_and_verify() {
	xcrun stapler staple "$APP_BUNDLE"
	[[ -f "$DMG_PATH" ]] && xcrun stapler staple "$DMG_PATH"
	echo ""
	echo "✅ Done! App is notarized and ready for distribution."
	echo ""
	echo "Verifying notarization:"
	spctl -a -vv "$APP_BUNDLE" 2>&1 || true
}

# Preferred: rcodesign with App Store Connect API Key (avoids notarytool SIGBUS on macOS 26)
ASC_API_KEY_PATH="${KAKU_ASC_API_KEY_PATH:-}"
if [[ -z "$ASC_API_KEY_PATH" ]]; then
	ASC_API_KEY_PATH=$(security find-generic-password -s "kaku-asc-api-key-path" -w 2>/dev/null || true)
fi

if [[ -n "$ASC_API_KEY_PATH" && -f "$ASC_API_KEY_PATH" ]] && command -v rcodesign >/dev/null 2>&1; then
	echo "Submitting via rcodesign (App Store Connect API Key)..."
	echo "  Key: $ASC_API_KEY_PATH"
	echo "  File: $SUBMISSION_PATH"
	echo ""
	if rcodesign notary-submit \
		--api-key-path "$ASC_API_KEY_PATH" \
		--staple \
		--wait \
		"$SUBMISSION_PATH"; then
		echo ""
		echo "✅ Notarization accepted! Stapling ticket..."
		staple_and_verify
	else
		echo "❌ rcodesign notarization failed."
		exit 1
	fi
	exit 0
fi

# Fallback: notarytool with Apple ID + app-specific password
APPLE_ID="${KAKU_NOTARIZE_APPLE_ID:-}"
TEAM_ID="${KAKU_NOTARIZE_TEAM_ID:-}"
PASSWORD="${KAKU_NOTARIZE_PASSWORD:-}"

if [[ -n "$TEAM_ID" ]] && ! is_valid_team_id "$TEAM_ID"; then
	echo "Warning: ignoring invalid KAKU_NOTARIZE_TEAM_ID: $TEAM_ID"
	TEAM_ID=""
fi

if [[ -z "$APPLE_ID" ]]; then
	echo "Checking Keychain for notarization credentials..."
	APPLE_ID=$(security find-generic-password -s "kaku-notarize-apple-id" -w 2>/dev/null || true)
fi
if [[ -z "$PASSWORD" ]]; then
	PASSWORD=$(security find-generic-password -s "kaku-notarize-password" -w 2>/dev/null || true)
fi
if [[ -z "$TEAM_ID" ]]; then
	TEAM_ID=$(codesign -dv "$APP_BUNDLE" 2>&1 | grep TeamIdentifier | head -1 | awk -F= '{print $2}')
	if [[ -n "$TEAM_ID" ]] && ! is_valid_team_id "$TEAM_ID"; then
		TEAM_ID=""
	fi
fi

if [[ -z "$APPLE_ID" || -z "$PASSWORD" || -z "$TEAM_ID" ]]; then
	echo ""
	echo "Error: No notarization credentials found."
	echo ""
	echo "Preferred (rcodesign, avoids notarytool SIGBUS on macOS 26):"
	echo "  1. Create an API key at https://appstoreconnect.apple.com/access/integrations/api"
	echo "  2. rcodesign encode-app-store-connect-api-key -o asc_api_key.json <issuer-id> <key-id> AuthKey_*.p8"
	echo "  3. security add-generic-password -s 'kaku-asc-api-key-path' -a 'kaku' -w '/path/to/asc_api_key.json'"
	echo ""
	echo "Fallback (Apple ID + app-specific password):"
	echo "  security add-generic-password -s 'kaku-notarize-apple-id' -a 'kaku' -w 'your-apple-id@example.com'"
	echo "  security add-generic-password -s 'kaku-notarize-password' -a 'kaku' -w 'your-app-specific-password'"
	exit 1
fi

echo "Submitting via notarytool (Apple ID fallback)..."
echo "  Apple ID: ${APPLE_ID:0:3}***"
echo "  Team ID:  ${TEAM_ID:0:3}***"
echo "  File: $SUBMISSION_PATH"
echo ""
echo "Uploading to Apple notarization service (this may take a few minutes)..."

SUBMIT_OUTPUT=$(xcrun notarytool submit "$SUBMISSION_PATH" \
	--apple-id "$APPLE_ID" \
	--team-id "$TEAM_ID" \
	--password "$PASSWORD" \
	--wait 2>&1) || {
	echo "Notarization submission failed:"
	echo "$SUBMIT_OUTPUT"
	exit 1
}

echo "$SUBMIT_OUTPUT"

if echo "$SUBMIT_OUTPUT" | grep -q "Accepted"; then
	echo ""
	echo "✅ Notarization accepted! Stapling ticket..."
	staple_and_verify
else
	echo ""
	echo "❌ Notarization failed."
	SUBMISSION_ID=$(echo "$SUBMIT_OUTPUT" | grep "id:" | head -1 | awk '{print $2}')
	if [[ -n "$SUBMISSION_ID" ]]; then
		echo "Fetching detailed log..."
		xcrun notarytool log "$SUBMISSION_ID" \
			--apple-id "$APPLE_ID" \
			--team-id "$TEAM_ID" \
			--password "$PASSWORD" 2>&1 || true
	fi
	exit 1
fi
