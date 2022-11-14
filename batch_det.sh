#! /bin/bash

for i in {1..64}
do
	Rscript R/detio.R $i 100 10 $i outputs/det/ &
done
