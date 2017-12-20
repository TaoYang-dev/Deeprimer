#!/bin/bash


# This script will extract a specific class of samples based on given performance metric, and give each of them the user-specified label
# Fields 16-20 are performance metrics: fwd_e2e, rev_e2e, total_reads, fwd_reads, rev_reads

# EXAMPLE: extract the samples that has total reads less than 100 and give them label 0
#./extract_class.sh total_reads V:0:0-100 source_file

if [ "$1" = "-h" ];then
	echo -e "[Command]: ./extract_class.sh [performance_metric] [criterion:class_label:range] [source_file]\nEXAMPLE: ./extract_class.sh total_reads V:0:0-100 source_file\n"
	exit
fi

CR=$(echo $2 | cut -f1 -d":")
lb=$(echo $2 | cut -f2 -d":")
rgl=$(echo $2 | cut -f3 -d":" | cut -f1 -d"-")
rgu=$(echo $2 | cut -f3 -d":" | cut -f2 -d"-")

if ! grep -q "$1" "$3"; then
   echo -e "$1 does not exist, create it and add it to source_file\n Or choose from: \n'total_reads', 'total_e2e', 'strand_bias', 'norm_tot', 'norm_toe', 'norm_bias'"
   exit
fi

if [ "$CR" = "V" ] || [ "$CR" = "v" ]; then
	awk -v c1="$1" -v l=${lb} -v b1=${rgl} -v b2=${rgu} '
		NR==1 {
    		for (i=1; i<=NF; i++) {
        		ix[$i] = i
    		};
    		OFS = "\t";
    		print $0, "class"
		}
		NR>1 && $ix[c1] >= b1 && $ix[c1] < b2 {OFS = "\t"; print $0, l}' $3 > ${3}_class_${1}_${lb}.txt
elif [ "$CR" = "P" ] || [ "$CR" = "p" ];then
	nll=$(cat $3 | wc -l)
	pnl=$(echo ${nll}*${rgl} | bc | cut -d'.' -f1)
	pnu=$(echo ${nll}*${rgu} | bc | cut -d'.' -f1)
	rg=$(($pnu-$pnl))
	
	fn=$(awk -v c1="$1" '
                NR==1 {
                for (i=1; i<=NF; i++) {
                        ix[$i] = i
                };
                }
                END {print ix[c1]}' $3)
	
	cat $3 | head -1 | awk '{OFS="\t";print $0, "class"}' > ${3}_class_${1}_${lb}.txt
	sort -k ${fn}n <(sed '1d' $3) \
		| head -$pnu | tail -$rg \
		| awk -v x=$lb '{OFS="\t"; print $0,x}' \
		| shuf  >> ${3}_class_${1}_${lb}.txt
else
	echo "Range spcification is wrong: choose from [VvPp]"
	exit
fi
