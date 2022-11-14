#! /bin/bash

for i in {1..64}
do
	Rscript R/ibmio_net.R $i 100 10 $i .5 36500 1e5 outputs/ibm_net/ &
done
