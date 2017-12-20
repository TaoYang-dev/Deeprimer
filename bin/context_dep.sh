#!/bin/bash

#Example: context_dep.sh 68627 ${DeeprimerPATH}/data/Coverage/Processed/Batch_68627

if [ "$1" = "-h" ];then
	echo "EXAMPLE: context_dep.sh [batch_id] [batch_location]"
	exit
fi

od=$PWD
cd $2

unalias grep 2>/dev/null

sed '1d' barcode_pooled_$1.bed > barcode_pooled_$1.bed_nh

for i in $(cat barcode_pooled_$1.bed_nh | cut -f22 | sort | uniq)
do
	pid=$(echo $i | sed 's/\_primers\_U\.txt//')
	cat barcode_pooled_$1.bed | grep ${pid} > ${pid}.subpanel 	

	cat ${pid}.subpanel | cut -f9 > ${pid}_insert.seqs
	awk '{nam=">insert"NR"\n"; OFS = ""; print nam, $0}' ${pid}_insert.seqs > ${pid}_insert.fa
	
	cat ${pid}.subpanel | cut -f5 > ${pid}_fwdp.seqs
	awk '{nam=">fwdp"NR"\n"; OFS = ""; print nam, $0}' ${pid}_fwdp.seqs > ${pid}_fwdp.fa
	
	cat ${pid}.subpanel | cut -f6 > ${pid}_revp.seqs
	awk '{nam=">revp"NR"\n"; OFS = ""; print nam, $0}' ${pid}_revp.seqs > ${pid}_revp.fa
	
	#index insert ref
	bowtie2-build ${pid}_insert.fa ${pid}_insert 2> temp.log

	cat ${pid}_fwdp.fa ${pid}_revp.fa > ${pid}_primer_ref.fa

        #indexing reference
        bowtie2-build primer_ref.fa primer_ref 2>>temp.log


	# clean the leftovers
	if [ -f fwdpvsp.sam ]; then
		rm -f fwdpvsp.sam revpvsp.sam fwdpvsi.sam revpvsi.sam
		touch fwdpvsp.sam revpvsp.sam fwdpvsi.sam revpvsi.sam
	else
		touch fwdpvsp.sam revpvsp.sam fwdpvsi.sam revpvsi.sam
	fi

	nll=$(cat ${pid}.subpanel | wc -l)
	for j in `seq 1 $nll`
	do
		line=$(cat ${pid}.subpanel | head -$j | tail -1)
		
		echo $line | awk '{nam=">fwdp"NR"\n"; OFS = ""; print nam,$5}' > fwdpseq.fa
		echo $line | awk '{nam=">revp"NR"\n"; OFS = ""; print nam,$6}' > revpseq.fa


		#alignment
		bowtie2 -x ${pid}_primer_ref -f fwdpseq.fa --min-score G,5,2 --nofw --local --no-hd >> fwdpvsp.sam 2>>temp.log
		bowtie2 -x ${pid}_primer_ref -f revpseq.fa --min-score G,5,2 --nofw --local --no-hd >> revpvsp.sam 2>>temp.log
		bowtie2 -x ${pid}_insert -f fwdpseq.fa --min-score G,5,2 --local --no-hd >> fwdpvsi.sam 2>>temp.log
		bowtie2 -x ${pid}_insert -f revpseq.fa --min-score G,5,2 --local --no-hd >> revpvsi.sam 2>>temp.log
		
		rm -f fwdpseq.fa revpseq.fa fwdp_ref.fa revp_ref.fa fwdp_ref.*.bt2 revp_ref.*.bt2
	done 

	rm -f temp.log ${pid}_insert.*.bt2 *.seqs ${pid}_insert.fa ${pid}_fwdp.fa ${pid}_revp.fa ${pid}_primer_ref.*

	for k in fwdpvsp revpvsp fwdpvsi revpvsi
	do
		awk '{split($12, a, ":");
			split($13, b, ":");
			OFS = "\t";
			if(a[1]=="AS")
				print $5, a[3], b[3]; 
			else 
				print $5, 0, 0}' ${k}.sam > ${k}.extraction
		rm -f ${k}.sam
	done
	
	#get the key for merging purpose
	awk '{OFS="\t"; print $1,$2,$3,$4,$5,$6,$22}' ${pid}.subpanel > ${pid}.key

	paste ${pid}.key fwdpvsp.extraction fwdpvsi.extraction revpvsp.extraction revpvsi.extraction -d"\t" > ${pid}.context
	rm -f *extraction ${pid}.key ${pid}.subpanel
done

# merge all panels and sort
echo -e "fwdpvsp_mq\tfwdpvsp_a1\tfwdpvsp_a2\tfwdpvsi_mq\tfwdpvsi_a1\tfwdpvsi_a2\trevpvsp_mq\trevpvsp_a1\trevpvsp_a2\trevpvsi_mq\trevpvsi_a1\trevpvsi_a2" > bypanel_ctxdep_$1.txt
cat *.context | sort -k1.1 -V -k2,2n -k3,3n -k4,4 -V -k5,5 -k6,6 -k7,7 | cut -f1-7 --complement | awk '{OFS = "\t"; if (NF==12) print $0; else print 0,0,0,0,0,0,0,0,0,0,0,0}'>> bypanel_ctxdep_$1.txt

rm -f *.context

cd $od
