if command -v gnome-text-editor >/dev/null 2>&1; then
  gnome-text-editor & disown
  echo "OK"
elif command -v gedit >/dev/null 2>&1; then
  gedit & disown
  echo "OK"
elif command -v kate >/dev/null 2>&1; then
  kate & disown
  echo "OK"
else
  xdg-open . &>/dev/null & disown
  echo "OK"
fi
