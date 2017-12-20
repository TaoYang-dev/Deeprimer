#!/bin/bash

#This script merges the uprimer file with the coverage analysis summary files, output new files to a folder named Perbarcode.
#Need to use metafile IonCode_Uprimer_Bed_matched.txt

#Example: Merge_files.sh [batch_id] [coverage_primer_matching file]


#mkdir -p /home/ionadmin/TaoY/Phoenix_shadow/Coverage_uprimer_merged/Batch_$1/Perbarcode
if [ "$1" = "-h" ];then
	echo -e "[Command]: Merge_uprimer_coverage.sh [batch_id] [coverage_primer_matching file]\n"
	exit
fi


mkdir -p ${DeeprimerPATH}/data/Coverage/Processed/Batch_${1}/Perbarcode

ionD=${DeeprimerPATH}/data/Coverage/Raw
uprimerD=${DeeprimerPATH}/data/Uprimers
outD=${DeeprimerPATH}/data/Coverage/Processed/Batch_${1}/Perbarcode


#Join ioncode and primer
cntl=$(cat $2 | awk -v id="Batch_$1" '$2 == id {print}'| wc -l)

for ((i=1;i<=cntl;i++))
do
	line=$(cat $2 | awk -v id="Batch_$1" '$2 == id {print}' | head -$i | tail -1)
	read ioncode  batch_id uprimer <<< $(echo $line | awk '{print $1, $2, $3}')
	
	join -1 4 -2 1 -o 1.1,1.2,1.3,1.4,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,2.10,1.5,1.6,1.7,1.8,1.9,1.10,1.11,1.12,1.13,1.14,1.15 <(sort -k4,4 ${ionD}/${batch_id}/${ioncode}) <(sort -k1,1 ${uprimerD}/${uprimer}) | awk -v X=$uprimer '{OFS = "\t";print $0,X}'> ${outD}/${ioncode}.${batch_id}.Umerged.txt
done
