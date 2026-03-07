#!/bin/bash
# Kaku config version check

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

CURRENT_CONFIG_VERSION=12
CONFIG_DIR="$HOME/.config/kaku"
STATE_FILE="$CONFIG_DIR/state.json"
LEGACY_VERSION_FILE="$CONFIG_DIR/.kaku_config_version"
LEGACY_GEOMETRY_FILE="$CONFIG_DIR/.kaku_window_geometry"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT="$SCRIPT_DIR/state_common.sh"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
	echo -e "${YELLOW}Error: missing shared state script: $COMMON_SCRIPT${NC}"
	exit 1
fi
# shellcheck source=state_common.sh
source "$COMMON_SCRIPT"

# Determine resource dir (always derive from script location, not hardcoded path)
RESOURCE_DIR="$SCRIPT_DIR"
TOOLS_SCRIPT="$RESOURCE_DIR/install_cli_tools.sh"

user_version="$(read_config_version)"

if [[ $user_version -eq 0 && ! -f "$STATE_FILE" ]]; then
	if [[ -f "$LEGACY_VERSION_FILE" || -f "$LEGACY_GEOMETRY_FILE" ]]; then
		legacy_version=0
		if [[ -f "$LEGACY_VERSION_FILE" ]]; then
			candidate="$(tr -d '[:space:]' < "$LEGACY_VERSION_FILE" || true)"
			if [[ "$candidate" =~ ^[0-9]+$ ]]; then
				legacy_version="$candidate"
			fi
		fi

		if [[ $legacy_version -eq 0 ]]; then
			legacy_version="$CURRENT_CONFIG_VERSION"
		fi

		persist_config_version "$legacy_version"
		user_version="$legacy_version"
	fi
fi

# Corrupted state file fallback: repair and continue with safe defaults.
if [[ -f "$STATE_FILE" && $user_version -eq 0 ]]; then
	persist_config_version
	user_version="$CURRENT_CONFIG_VERSION"
fi

# Skip if already up to date or new user
if [[ $user_version -eq 0 || $user_version -ge $CURRENT_CONFIG_VERSION ]]; then
	exit 0
fi

echo -e "${BOLD}Kaku config update available!${NC} v$user_version -> v$CURRENT_CONFIG_VERSION"
echo ""

# Show only current release highlights to keep this prompt short and maintainable.
echo -e "${BOLD}What's new:${NC}"
case "$CURRENT_CONFIG_VERSION" in
12)
	echo "  • Shell text editing: Cmd+A select all, Shift+Arrow selection, ESC to cancel"
	echo "  • AI error fixer: auto-suggests fixes on failure, Cmd+Shift+E to apply"
	echo "  • Type y to launch Yazi, cd+Tab falls back to zsh-z history"
	echo "  • Plugins already loaded by your config are no longer duplicated"
	echo "  • Fixed: delete key after Chinese IME, terminal type for sudo+nano"
	echo "  • Fixed: yazi alias conflict with existing shell aliases"
	echo "  • Fixed: SSH sessions now use xterm-256color for better remote compatibility"
	;;
11)
	echo "  • Shell text editing: Cmd+A select all, Shift+Arrow selection, ESC to cancel"
	echo "  • AI error fixer: auto-suggests fixes on failure, Cmd+Shift+E to apply"
	echo "  • Type y to launch Yazi, cd+Tab falls back to zsh-z history"
	echo "  • Plugins already loaded by your config are no longer duplicated"
	echo "  • Fixed: delete key after Chinese IME, terminal type for sudo+nano"
	;;
*)
	echo "  • Shell integration and reliability improvements"
	echo "  • See project release notes for full details"
	;;
esac
echo ""

read -p "Apply update now? Press Enter to continue, type n to skip: " -n 1 -r
echo

if [[ $REPLY =~ ^[Nn]$ ]]; then
	persist_config_version
	echo -e "${YELLOW}Skipped${NC}"
	echo ""
	echo "Press any key to continue..."
	read -n 1 -s
	exit 0
fi

# Apply updates
if [[ -f "$RESOURCE_DIR/setup_zsh.sh" ]]; then
	KAKU_SKIP_TOOL_BOOTSTRAP=1 bash "$RESOURCE_DIR/setup_zsh.sh" --update-only
else
	echo -e "${YELLOW}Error: missing setup script at $RESOURCE_DIR/setup_zsh.sh${NC}"
	exit 1
fi

if [[ -f "$TOOLS_SCRIPT" ]]; then
	if ! KAKU_AUTO_INSTALL_TOOLS=1 bash "$TOOLS_SCRIPT"; then
		echo ""
		echo -e "${YELLOW}Optional tool installation failed.${NC}"
	fi
fi

persist_config_version

echo ""
echo -e "\033[1;32m🎃 Kaku environment is ready! Enjoy coding.\033[0m"
echo ""
echo "Press any key to continue..."
read -n 1 -s

# Replace current process with the user's login shell
TARGET_SHELL="$(detect_login_shell)"
exec "$TARGET_SHELL" -l
