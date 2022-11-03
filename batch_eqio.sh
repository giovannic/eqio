#! /bin/bash

for i in {1..100}
do
	Rscript R/eqio.R $i 10000 $i outputs/eq/ &
done
