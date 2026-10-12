#!/bin/bash
# Runs as root: power the VM off 20 min after the pipeline finishes, so a
# finished spot VM never sits idle and billing.
while [ ! -f /home/harsh/vd/runs/ALL_DONE ]; do sleep 60; done
sleep 1200
shutdown -h now
