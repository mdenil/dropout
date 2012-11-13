#!/bin/bash

if [ -z "$1" ]; then
    TERM="dumb"
    PLOT_FILE=/dev/stdout
else
    TERM="png"
    PLOT_FILE="$1"
fi

SMOOTHING=0.0

(
gnuplot 2>/dev/null <<EOF
set term $TERM
set output "$PLOT_FILE"

set title "Number of errors on MNIST test set."
set xlabel "epoch"
set ylabel "num errors"
set xrange [10:3000]
#set yrange [100:10000]
#set yrange [80:1000]
#set yrange [80:500]
#set yrange [80:220]
#set logscale y
set logscale x
plot "<cat results_backprop.txt | bin/moving_average $SMOOTHING" title "backprop", \
     "<cat results_dropout.txt | bin/moving_average $SMOOTHING"  title "dropout"

EOF
) 

