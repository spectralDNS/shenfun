#!/bin/bash
recite() {
 	awk '$0~/key:/{print $2 }' demos/papers.pub | while read -r line ; do 
		doconce replace '['$line']_' ':cite:'\`$line\` source/$1
	done
}
recite $1
