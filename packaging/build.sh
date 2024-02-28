#!/bin/sh
set -e
(cd .. ; \
rm -r packaging/Microscope-Calibration; \
mkdir packaging/Microscope-Calibration; \
git archive --format=tar HEAD | tar xvf - -C packaging/Microscope-Calibration \
)

apptainer build --force overfocus.sif overfocus.def 
apptainer overlay create --size 1024 overfocus.sif
