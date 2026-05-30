#!/usr/bin/env bash
# PreToolUse hook (Edit|Write|NotebookEdit): refuse to modify the LUMI mount.
#
# The mount /mnt/lumi2 (and /mnt/lumi) lags the local repo and is updated by
# `git pull` on LUMI — editing it directly caused a merge conflict before
# (see memory: feedback_edit_local_not_mnt). Always edit the local repo at
# /home/nordling/PycharmProjects/CESM2_emulator_from_lumi and push instead.
#
# Reads the tool call JSON on stdin; exit 2 blocks the call and shows the
# message to Claude. jq isn't installed here, so parse with python3.
path=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)

case "$path" in
    /mnt/lumi/*|/mnt/lumi2/*)
        echo "BLOCKED: refusing to edit '$path' on the LUMI mount." >&2
        echo "Edit the local repo (/home/nordling/PycharmProjects/CESM2_emulator_from_lumi) and push; pull on LUMI. Editing the mount caused a merge conflict before." >&2
        exit 2
        ;;
esac
exit 0
