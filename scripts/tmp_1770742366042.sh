set -e
a="яндекс музыки на моём пк"

# common aliases → desktop IDs / flatpak IDs
flatpak_id=""
case "$(echo "$a" | tr '[:upper:]' '[:lower:]')" in
  steam) flatpak_id="com.valvesoftware.Steam" ;;
  discord) flatpak_id="com.discordapp.Discord" ;;
  telegram|telegram-desktop) flatpak_id="org.telegram.desktop" ;;
  spotify) flatpak_id="com.spotify.Client" ;;
esac

# 1) desktop entry
if command -v gtk-launch >/dev/null 2>&1; then
  gtk-launch "$a" >/dev/null 2>&1 && echo "OK" && exit 0 || true
fi

# 2) run as command in PATH
if command -v "$a" >/dev/null 2>&1; then
  ("$a" >/dev/null 2>&1 &)
  echo "OK"
  exit 0
fi

# 3) flatpak fallback
if [ -n "$flatpak_id" ] && command -v flatpak >/dev/null 2>&1; then
  if flatpak info "$flatpak_id" >/dev/null 2>&1; then
    (flatpak run "$flatpak_id" >/dev/null 2>&1 &)
    echo "OK"
    exit 0
  fi
fi

# 4) snap fallback
if command -v snap >/dev/null 2>&1; then
  if snap list "$a" >/dev/null 2>&1; then
    (snap run "$a" >/dev/null 2>&1 &)
    echo "OK"
    exit 0
  fi
fi

echo "App not found: $a" >&2
exit 1
