systemctl list-units --type=service --state=running | grep -E 'display|ssh'
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head