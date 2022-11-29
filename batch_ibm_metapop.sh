#! /bin/bash

for i in {1..100}
do
	Rscript R/ibmio_metapop.R $i 100 10 $i .5 36500 1e5 3 outputs/ibm_metapop/ &
done
