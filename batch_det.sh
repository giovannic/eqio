#! /bin/bash

for i in {1..100}
do
	Rscript R/detio.R $i 1000 10 $i data/rainfall_coeffs.csv outputs/det_season/ &
done
