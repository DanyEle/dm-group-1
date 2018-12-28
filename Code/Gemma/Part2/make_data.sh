#!/bin/bash
i=0
filename='allResults.txt'
rm -r $filename
for fldr in cityblock cosine euclidean
do
  file=$fldr'/results.txt'

  while IFS= read -r var
  #read lines
  do
    if [ "$var" = "eps NumClusters numNoise Silhouette" ]
    then
      if [ $i -eq 0 ]
      then
        echo "distance version" $var &>>$filename
      fi
    else
      z=$((i / 14))
      n=$(($z%5))
      echo $fldr $n $var &>>$filename
    fi
    ((++i))
  done < "$file"
done
