#!/bin/bash

# Runs a profiler, spits results to prof.txt
/usr/local/bin/kernprof -l ./PlateIsolator.py
python -m line_profiler PlateIsolator.py.lprof > prof.txt
rm PlateIsolator.py.lprof
rm PlateIsolator.py.lprof.prof
rm PlateIsolator.py.prof
subl prof.txt