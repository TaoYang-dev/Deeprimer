#!/bin/bash

#This script merges the primers in the same panel and same barcode, the new coverage statistics will be the mean of the merged ones.
#The output file name is barcode_pooled_[batch_id].bed


#EXAMPLE: PoolbyPrimer.sh 68627 ${DeeprimerPATH}/data/Coverage/Processed/Batch_${1}

if [ "$1" = "-h" ];then
	echo -e "[Command]: PoolbyPrimer.sh <batch_id> <batch_location>\n"
	exit
fi

batch_file=All_batch_insert_${1}.txt
#Extract primer information
#NOTE: Coordinate adjustment: some of the starting point is fwdp end location+1, some is fwdp end location, this is not consistent, needed to adjust
cat ${2}/${batch_file} | awk '{OFS="\t"; print $1,$2,$3,$4,$5,$6,$26,$7,$8,$25,$9,$10,$11,$12,$13}' | \
		sort -k1.1 -V -k2,2n -k3,3n -k4,4 -V -k5,5 -k6,6 -k7,7 | uniq > ${2}/batch_${1}_amp_sorted.uniq.bed

# create unique key
cat ${2}/${batch_file} | awk '{OFS="\t"; key=$1"&"$2"&"$3"&"$4"&"$5"&"$6"&"$26; print key, $15, $17, $18, $19, $20, $21}' > ${2}/key_value_${1}.txt

awk '{
	a1[$1]+=$2;
	N[$1]++;
	a2[$1]+=$3;
	a3[$1]+=$4;
	a4[$1]+=$5;
	a5[$1]+=$6;
	a6[$1]+=$7
     } 
     END {
     for (key in a1) printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", key, a1[key]/N[key],
     a2[key]/N[key], a3[key]/N[key], a4[key]/N[key], a5[key]/N[key], a6[key]/N[key], N[key])
     }' key_value_${1}.txt > key_pooled_${1}.txt

cat ${2}/key_pooled_$1.txt | sed 's/\&/\t/g' | sort -k1.1 -V -k2,2n -k3,3n -k4,4 -V -k5,5 -k6,6 -k7,7 > ${2}/key_pooled_${1}.sorted.txt

echo -e "chr\tins.start\tins.end\tamp.name\tfwdp.seq\trevp.seq\tfwdU.seq\trevU.seq\tinsert.seq\tfwdMaxTm\trevMaxTm\tfwdNumTBlock\trevNumTBlock\tpass\tgc_count\tfwd_e2e\trev_e2e\ttotal_reads\tfwd_reads\trev_reads\tN_barcode\tPanel" > ${2}/barcode_pooled_$1.bed

paste ${2}/batch_${1}_amp_sorted.uniq.bed ${2}/key_pooled_${1}.sorted.txt -d'\t' | \
awk '$1 == $16 && $2 == $17 && $3 == $18 && $4 == $19 && $5 == $20 && $6 == $21 && $7 == $22 {OFS="\t"; print $1,$2,$3,$4,$5,$6,$8,$9,$10,$11,$12,$13,$14,$15,$23,$24,$25,$26,$27,$28,$29,$7}' >> ${2}/barcode_pooled_$1.bed

#Clean up intermediate files
rm ${2}/batch_${1}_amp_sorted.uniq.bed ${2}/key_value_${1}.txt \
		${2}/key_pooled_${1}.txt ${2}/key_pooled_${1}.sorted.txt

