#!/bin/sh

set -eu

# Prepare host keys on first start, then launch sshd in the background.
mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd

exec python3 /app/main.py "$@"
