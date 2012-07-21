#!/bin/bash

if [ -z "$1" ]; then
    TERM="dumb"
    PLOT_FILE=/dev/stdout
else
    TERM="png"
    PLOT_FILE="$1"
fi

(
gnuplot 2>/dev/null <<EOF
set term $TERM
set output "$PLOT_FILE"

plot "results_backprop.txt" title "backprop", "results_dropout.txt" title "dropout"

EOF
) 

echo

