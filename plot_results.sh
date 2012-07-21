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

set title "Number of errors on MNIST test set."
set xlabel "epoch"
set ylabel "num errors"
set xrange [1:3000]
set yrange [*:10000]
set logscale y
plot "results_backprop.txt"  title "backprop", \
     "results_dropout.txt"  title "dropout"

EOF
) 

echo

