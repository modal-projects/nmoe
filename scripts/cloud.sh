#!/bin/bash
# cloud.sh - Provision Prime instance and bootstrap nmoe
#
# Usage:
#   bash scripts/cloud.sh              # Interactive: create pod, bootstrap, SSH
#   bash scripts/cloud.sh --status     # List pods
#   bash scripts/cloud.sh --teardown   # Tear down pods
#   bash scripts/cloud.sh --avail      # Show B200 availability
#   bash scripts/cloud.sh --bootstrap POD_ID  # Bootstrap existing pod
#
# Requires:
#   prime (pip install prime-cli && prime login)
#   jq (apt install jq / brew install jq)

set -euo pipefail

err() { echo "[cloud] ERROR: $1" >&2; exit 1; }

command -v prime &>/dev/null || err "prime not found. Install: pip install prime-cli && prime login"
command -v jq &>/dev/null || err "jq not found. Install: apt install jq"

bootstrap_pod() {
  local pod_id="$1"
  echo "[cloud] Bootstrapping pod $pod_id..."
  prime pods ssh "$pod_id" -- 'bash -s' << 'BOOTSTRAP'
set -e
[ ! -d ~/nmoe ] && git clone https://github.com/noumena-network/nmoe.git ~/nmoe
cd ~/nmoe && git pull --ff-only && bash scripts/bootstrap.sh
BOOTSTRAP
  echo "[cloud] Bootstrap complete!"
}

wait_for_pod() {
  local pod_id="$1"
  echo "[cloud] Waiting for pod $pod_id..."
  while true; do
    status=$(prime pods status "$pod_id" -o json 2>/dev/null | jq -r '.status' || echo "")
    case "$status" in
      ACTIVE) echo "[cloud] Pod ready!"; return 0 ;;
      ERROR|TERMINATED) err "Pod failed: $status" ;;
      *) printf "."; sleep 5 ;;
    esac
  done
}

case "${1:-}" in
  --status|-s)
    prime pods list
    ;;

  --avail|-a)
    prime availability list --gpu-type B200_180GB
    ;;

  --teardown|-t)
    echo "[cloud] Current pods:"
    prime pods list
    echo ""
    read -p "Enter pod ID to terminate (or 'all'): " pod_id
    if [ "$pod_id" = "all" ]; then
      prime pods list -o json | jq -r '.pods[].id' | while read -r id; do
        [ -z "$id" ] && continue
        echo "[cloud] Terminating $id..."
        prime pods terminate "$id" --yes || true
      done
    elif [ -n "$pod_id" ]; then
      prime pods terminate "$pod_id" --yes
    fi
    echo "[cloud] Done"
    ;;

  --bootstrap|-b)
    [ -z "${2:-}" ] && err "Usage: cloud.sh --bootstrap POD_ID"
    wait_for_pod "$2"
    bootstrap_pod "$2"
    echo ""
    echo "[cloud] Connecting..."
    prime pods ssh "$2"
    ;;

  *)
    echo "[cloud] B200 Availability:"
    prime availability list --gpu-type B200_180GB --gpu-count 8
    echo ""
    echo "[cloud] Creating pod (follow prompts)..."
    echo ""
    prime pods create --gpu-type B200_180GB --gpu-count 8

    echo ""
    echo "[cloud] Pod created! To bootstrap, run:"
    echo "  bash scripts/cloud.sh --bootstrap <POD_ID>"
    echo ""
    echo "Or wait for it and bootstrap now:"
    read -p "Enter pod ID (or press Enter to skip): " pod_id
    if [ -n "$pod_id" ]; then
      wait_for_pod "$pod_id"
      bootstrap_pod "$pod_id"
      echo ""
      echo "[cloud] Ready! Commands:"
      echo "  n speedrun    # Quick repro (~5 min)"
      echo "  n train       # Production training"
      echo ""
      echo "[cloud] Connecting..."
      prime pods ssh "$pod_id"
    fi
    ;;
esac
