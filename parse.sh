#!/bin/bash
# example ./parse.sh path/to/file.conllu: 
#./parse.sh data/UD_2.4/it/it_isdt-ud-dev.conllu

FILE="$1"

echo "$FILE"

if [ ! -f "$FILE" ]
then
    echo "File not found!"
    exit
fi

if [[ ! "$FILE" == *.conllu ]]
then
    echo "Wrong file extension. Needed .conllu"
    exit
fi

sed '/^# .*/d' "$FILE" > tmp.conllu  	
sed '/[0-9]\+-/d' tmp.conllu > "$FILE"		
rm tmp.conllu