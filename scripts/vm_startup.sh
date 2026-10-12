#!/bin/bash
# GCE startup-script: resume the experiment pipeline after a spot preemption,
# and power the VM off 20 min after the pipeline finishes (so it never idles).
# Active only while /home/harsh/vd/AUTORUN exists.
U=harsh
D=/home/$U/vd
[ -f $D/AUTORUN ] || exit 0
pgrep -f "$D/scripts/finisher.sh" >/dev/null || setsid nohup bash $D/scripts/finisher.sh >/dev/null 2>&1 < /dev/null &
[ -f $D/runs/ALL_DONE ] && exit 0
for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done
pgrep -f "scripts/run_all.sh" >/dev/null && exit 0
sudo -u $U bash -c "cd $D && echo \"=== boot resume \$(date)\" >> pipeline.log && nohup bash scripts/run_all.sh >> pipeline.log 2>&1 < /dev/null &"
