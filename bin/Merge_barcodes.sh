#!/bin/bash

#This script obtains the insert sequences and combines all the barcodes in the same batch, the output file is named All_batch_insert_[batch_id].txt
#Need to use bedtools > v2.25.0

#Example: Merge_barcodes 60112 ${DeeprimerPATH}/data/Coverage/Processed/Batch_60112/

if [ "$1" = "-h" ];then
	echo -e "[Command]: Merge_barcodes.sh <batch_id> <Batch_location>\n"
	exit
fi

#bedbin=/home/ionadmin/TaoY/src/bedtools2/bin
hg19=${DeeprimerPATH}/data/HG19/hg19.fa
cd ${2}

cat ${2}/Perbarcode/*.Umerged.txt | sed 's/ /\t/g' | sort -k1,1 -V -k2,2n -k3,3n -k4,4 -V > ${2}/All_batch_$1.bed

fastaFromBed -fi ${hg19} -bed ${2}/All_batch_$1.bed -fo ${2}/Insert_seq_$1.fa

cat ${2}/Insert_seq_$1.fa | grep -v ">chr" > ${2}/seq_only_$1.txt

awk 'NF{NF--};1' ${2}/All_batch_$1.bed > ${2}/All_batch_$1.bed.temp1
awk '{print $NF}' ${2}/All_batch_$1.bed > ${2}/All_batch_$1.bed.temp2

paste ${2}/All_batch_$1.bed.temp1 ${2}/seq_only_$1.txt ${2}/All_batch_$1.bed.temp2 -d'\t' | sed 's/ /\t/g' > ${2}/All_batch_insert_$1.txt

rm ${2}/seq_only_$1.txt ${2}/All_batch_$1.bed ${2}/Insert_seq_$1.fa ${2}/All_batch_$1.bed.temp1 ${2}/All_batch_$1.bed.temp2





