#!/bin/bash


#EXAMPLE: ./Get_mappability.sh 68627 ${DeeprimerPATH}/data/Coverage/Processed/Batch_68627

if [ "$1" = "-h" ];then
	echo -e "[Command]: ./Get_mappability.sh [batch_id] [batch_location]\n"
	exit
fi



ucscmap=${DeeprimerPATH}/data/Mappability/ucsc_36mermappability_trimmed.sorted.bedGraph
bedbin=/home/ionadmin/TaoY/src/bedtools2/bin
chrsize=${DeeprimerPATH}/data/HG19/chr.sizes

id=$1

awk '{OFS = "\t";
	fpl=length($5);
	rpl=length($6);
	insl=length($9);
	print $1,$3-insl+1,$3}' <(sed '1d' ${2}/barcode_pooled_${id}.bed) > ${2}/Batch_${id}_insert.bed

awk '{OFS = "\t";
        fpl=length($5);
        rpl=length($6);
        insl=length($9);
        print $1,($3-insl+1)-fpl,$3-insl}' <(sed '1d' ${2}/barcode_pooled_${id}.bed) > ${2}/Batch_${id}_fwdp.bed

awk '{OFS = "\t";
        fpl=length($5);
        rpl=length($6);
        insl=length($9);
        print $1,$3+1,$3+rpl}' <(sed '1d' ${2}/barcode_pooled_${id}.bed) > ${2}/Batch_${id}_revp.bed


#Get mappability for each part
for i in insert revp fwdp
do	
	sort -V ${2}/Batch_${id}_${i}.bed > ${2}/Batch_${id}_${i}.sorted.bed	
	${bedbin}/bedtools map -a ${2}/Batch_${id}_${i}.sorted.bed -b $ucscmap -g $chrsize -null 0 -c 4,4 -o mean,min  > ${2}/Batch_${id}_${i}.mappability
	#sed -i '$d' Batch_${id}_${i}.mappability
	
	#bring the order back
	
	awk '{key1 = $1"&"$2"&"$3; OFS="\t"; print key1, $4, $5}' ${2}/Batch_${id}_${i}.mappability > ${2}/Batch_${id}_${i}.file1
	awk '{key2 = $1"&"$2"&"$3; print key2}' ${2}/Batch_${id}_${i}.bed > ${2}/Batch_${id}_${i}.file2

	awk 'FNR == NR { lineno[$1] = NR; next} {print lineno[$1], $0;}' ${2}/Batch_${id}_${i}.file2 ${2}/Batch_${id}_${i}.file1 | sort -k1,1n | cut -d' ' -f2- | sed 's/\&/\t/g' >  ${2}/Batch_${id}_${i}.inorder.mappability
	
	rm ${2}/Batch_${id}_${i}.file1 ${2}/Batch_${id}_${i}.file2 ${2}/Batch_${id}_${i}.mappability ${2}/Batch_${id}_${i}.sorted.bed ${2}/Batch_${id}_${i}.bed
done


#Add header
echo -e "fwdp.mean.map\tfwd.min.map\trevp.mean.map\trevp.min.map\tins.mean.map\tins.min.map" > ${2}/Batch_${id}_mappability.txt

#write the mappability records
paste ${2}/Batch_${id}_fwdp.inorder.mappability ${2}/Batch_${id}_revp.inorder.mappability ${2}/Batch_${id}_insert.inorder.mappability | awk '{OFS="\t"; print $4, $5, $9, $10, $14, $15}' >> ${2}/Batch_${id}_mappability.txt

#clean intermediate files
rm ${2}/Batch_${id}_*inorder* 

