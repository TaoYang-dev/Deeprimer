#!/bin/bash

#This script will combine all features together and create performance metrics
#total_e2e = fwdp_e2e + revp_e2e
#strand_bias = diff(fwdp_e2e + revp_e2e)
#normalized_total_e2e_bypanel 
#normalized_total_bypanel
#EXAMPLE: ./Get_raw_input.sh [outfile_name] [file1] [file2] [...] 

if [ "$1" = "-h" ];then
	echo "EXAMPLE: ./Get_raw_input.sh [outfile_name] [file1] [file2] [...]"
	exit
fi

#create total_e2e and strand bias
awk -v c1="fwd_e2e" -v c2="rev_e2e" '
function abs(x){return ((x < 0.0) ? -x : x)}
NR==1 {
    for (i=1; i<=NF; i++) {
        ix[$i] = i
    };
    OFS = "\t";
    print $0, "total_e2e", "strand_bias"
}
NR>1 {OFS = "\t";
	toe = $ix[c1]+$ix[c2]; 
	bias = abs($ix[c1]-$ix[c2]);
	print $0, toe, bias}' <(paste ${@:2} -d"\t")  > $1.temp1

#create normalized data
awk '{  
	N[$22]++;
	a1[$22]+=$18;
        a2[$22]+=$41
	} 
	END {OFS = "\t";
	for (key in a1)
        print  key, a1[key]/N[key], a2[key]/N[key]}' <(sed '1d' $1.temp1)  > $1.temp2

sed -i 's/ /\t/g' $1.temp2
sed -i 's/ /\t/g' $1.temp1

join -1 22 -2 1 <(sed '1d' $1.temp1 | sort -k22) <(sort -k1 $1.temp2) -t $'\t' | cut -f1 --complement> $1.temp3

cat $1.temp1 | head -1 | cut -f22 --complement > $1.head


cat $1.head $1.temp3 > $1.hd.temp3

awk 'NR == 1 {OFS = "\t";
	print $0, "mean_tot", "mean_toe", "norm_tot", "norm_toe", "norm_bias"}
NR>1 {OFS = "\t";
	if ($42==0 && $43 ==0){a1=0;a2=0;a3=0;print $0, a1, a2, a3}
	else if ($42!=0 && $43==0){a1=$18/$42;a2=0;a3=0;print $0, a1, a2, a3}
	else if ($42==0 && $43!=0){a1=0;a2=$40/$43;a3=$41/$43;print $0, a1, a2, a3}
	else {a1=$18/$42;a2=$40/$43;a3=$41/$43;print $0, a1, a2, a3}
	}' $1.hd.temp3 > $1
	
rm -f $1.temp* $1.head $1.hd.temp3



