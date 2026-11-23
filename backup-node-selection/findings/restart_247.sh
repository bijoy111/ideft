#!/bin/bash
# Restart the campaign at the next step boundary with the 19 good backup nodes.
cd ~/ideft/backup-node-selection/code
until grep -q "done check_client16_128_homog" ../findings/campaign.out; do sleep 3; done
pkill -INT -f "python3 campaign.py --nodes 15 "
while pgrep -f "python3 campaign.py --nodes 15 " >/dev/null; do sleep 2; done
while tmux has-session -t campaign 2>/dev/null; do sleep 1; done
echo "[$(date +%H:%M:%S)] restarting with 19 backup nodes (node16 excluded)" >> ../findings/campaign.out
tmux new -d -s campaign "python3 campaign.py --nodes 20 --exclude-nodes 16 --iface enp65s0f0np0 2>&1 | tee -a ../findings/campaign.out"
