#!/usr/bin/env bash
# Local watchdog: restart the spot VM whenever it is preempted (the VM's
# startup-script resumes the pipeline); exit once runs/ALL_DONE exists.
ZONE=${ZONE:-us-central1-c}
VM=${VM:-violence-l4}
i=0
while true; do
  st=$(gcloud compute instances describe $VM --zone $ZONE --format="value(status)" 2>/dev/null)
  if [ "$st" = "TERMINATED" ] || [ "$st" = "STOPPED" ]; then
    echo "$(date '+%H:%M:%S') $VM is $st -> starting"
    gcloud compute instances start $VM --zone $ZONE --quiet 2>&1 | tail -1
  elif [ "$st" = "RUNNING" ]; then
    r=$(timeout 90 gcloud compute ssh $VM --zone $ZONE --quiet --command "test -f ~/vd/runs/ALL_DONE && echo ALLDONE; tail -1 ~/vd/pipeline.log" 2>/dev/null | tail -2)
    echo "$(date '+%H:%M:%S') running | $(echo "$r" | tail -1 | cut -c1-150)"
    echo "$r" | grep -q ALLDONE && { echo "pipeline finished"; exit 0; }
  fi
  i=$((i + 1))
  sleep 120
done
