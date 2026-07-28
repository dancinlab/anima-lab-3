#!/usr/bin/env bash
# Launch a command on a research host — via a script, never inline.
#
# Assembling tmux/quoting inside an ssh argument breaks in transit and becomes a
# SILENT no-op: the launcher returns 0, nothing runs, and the absence looks
# exactly like a job that has not finished yet. This session lost two scoring
# runs that way, and a third when overlapping waiters pkill'd each other's
# process. So: write the command to a file, copy the file, run the file.
#
#   scripts/run_remote.sh summer@192.168.50.60 /path/local_script.sh
#
# The remote copy lands beside the repo checkout on the host and is executed
# there; its own nohup/redirect decides detachment. Verify by reading the log the
# script writes, never by trusting this command's exit code.
set -u
[ $# -ge 2 ] || { echo "usage: $0 <user@host> <local-script> [remote-name]" >&2; exit 2; }
HOST=$1
LOCAL=$2
NAME=${3:-$(basename "$LOCAL")}
REMOTE_DIR=${REMOTE_DIR:-anima-clm-pure}

scp -o ConnectTimeout=15 "$LOCAL" "$HOST:$REMOTE_DIR/$NAME" >/dev/null || exit 1
ssh -o ConnectTimeout=15 "$HOST" bash "$REMOTE_DIR/$NAME"
