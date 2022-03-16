#!/bin/bash

# Roughly compares difference in numerical results between different adapter runs.
# The values should not be completely different, but might vary in non-significant digits
# due to FP on different architectures/OS.
#
# It just strips away all "non-numerical" looking data and then compares files field by field.
# Might need future parsing fixes for newly added adapters (awk will complain when this happen)
#
# Input: folder where results should be compared and (optional) maximum relative difference between
# numerical values in the reference and obtained files ( averaged over the file and over the individual field)

avg_diff_limit="0.01"
max_diff_limit="0.01"


if [ $# -lt 2 ]; then
  echo 1>&2 "Usage: $0 folder1 folder2"
  exit 1
else
  folder1=$1
  folder2=$2
  shift
  shift
fi

for i in "$@";
do case $i in
    --avg_diff=*)
    avg_diff_limit="${i#*=}"
    ;;
    --max_diff=*)
    max_diff_limit="${i#*=}"
    ;;
    *)
    echo "Unknown options $i. Possible options: --avg_diff, --max_diff "; exit 1;;
esac
done

ret=0

# Get the list of files that differ
diff_result=$( diff -rq $folder1 $folder2 )

# split result into two, depending whether files are unique in one folder
# or just differ
diff_files=$( echo "$diff_result" | sed '/Only/d' )
only_files=$( echo "$diff_result" | sed '/differ/d')


echo "------------- Comparing files -------------"

# Pairwise compare files
if [ -n "$diff_files" ]; then
  mapfile -t array_files < <( echo "$diff_files"  | sed 's/Files\|and\|differ//g' )
  arr_len="${#array_files[@]}"

  # loop through differing files
  for (( i = 0; i<${arr_len}; i = i + 1));
  do
    file1=$( echo "${array_files[i]}" | awk '{print $1}' )
    file2=$( echo "${array_files[i]}" | awk '{print $2}' )

    filename=$(basename $file1)

    # Filtering section. We compare numbers and text seperately

    # prefiltering dates, timestamps and other words that signal a line with
    # constantly changing values (that do not actually affect the results)
    # Explanation of what the regex blocks below match:
    #
    #   List of words which occur in lines we do not want to check ;
    #   Time format ;
    #   Date format ;
    #   Weekday format ;
    #   Line number specifier of form [x]:[00] ;
    #   another line number specifier of fom [1,[2,3]] ;
    #   sequence of whitespaces ;
    #   hexadecimals (will commonly describe memory locations);
    #   phrase "[number] sec(onds)"
    #   delete anything after 'Run finished' \\ avoids Benchmark times that get printed below

    pre_filter='/Timestamp\|[rR]untime\|[vV]ersion\|[rR]evision\|Unexpected\|Host:/d;
                s/[0-9][0-9][:\.][0-9][0-9][:\.][0-9][0-9]//g;
                s/[0-9][0-9]\/[0-9][0-9]\/[0-9][0-9][0-9][0-9]//g;
                s/Mon\|Tue\|Wed\|Thu\|Fri\|Sat\|Sun//g;
                s/\[.\+\]:[0-9]\+//g;
                s/\[\[[0-9]\+,[0-9]\],[0-9]\]://g;
                s/\s*$//g;
                s/0x[0-9a-f]//g;
                s/\([0-9]*[\.]\)\?[0-9]\+ sec//g;
                /Run finished/q'

    # numerical filter, looks to find numbers of any format
    num_filter='[-]\?\([0-9]*[\.]\)\?[0-9]\+\([eE][+-]\?[0-9]\+\)\?'

    # # exponential filter, DELETES exponent! (if exponent should not be interpreted)
    # exp_filter='s/[eE][+-][0-9]\+//g'
    exp_filter=''

    # text filter. This also explicitly deletes all numeric patterns
    txt_filter="/[a-df-zA-DF-Z]/!d ; s/$num_filter//g"

    # Apply filters
    file1_num=$( cat "$file1" | sed "$pre_filter" | grep -o "$num_filter" | sed "$exp_filter")
    file2_num=$( cat "$file2" | sed "$pre_filter" | grep -o "$num_filter" | sed "$exp_filter")

    file1_txt=$( cat "$file1" | sed "$pre_filter" | sed "$txt_filter")
    file2_txt=$( cat "$file2" | sed "$pre_filter" | sed "$txt_filter")


    # Create side-by-side views
    txt_diff=$( diff -y --speed-large-files --suppress-common-lines <(echo "$file1_txt") <(echo "$file2_txt") )
    num_diff=$( paste <(echo "$file1_num") <(echo "$file2_num") )

    # Debug commands. Helpful for checking the state of filtered output when adjusting filters.

    # diff -y --speed-large-files --suppress-common-lines <(echo "$file1_txt") <(echo "$file2_txt") > DEBUG_TXT_DIFF
    # paste <(echo "$file1_num") <(echo "$file2_num") > DEBUG_NUM_DIFF
    # cat "$file1" | sed "$num_filter" > DEBUG_F1
    # cat "$file2" | sed "$num_filter" > DEBUG_F2

    # Pairwise compare file fields and compute average/maximum relative difference
    echo "Comparing values in '$filename'..."


    if [ -n "$num_diff" ]; then
      rel_max_difference=$( export max_diff_limit; export avg_diff_limit; echo "$num_diff" | awk 'function abs(v) {return v < 0.0 ? -v : v}
      function log10(v) {return log(v)/log(10)}
      BEGIN {
        max_diff=0.0;
        sum=0.0;
        total_entries=0;
        exp_threshold=-12;
      }
      {
        r=NF/2;
        for(i = 1; i <= r; i++) {
          total_entries += 1;
          if (log10(abs($i)) > exp_threshold && log10(abs($(i + r))) > exp_threshold) {
            ind_diff = abs((($(i + r)-$i)/$i ));
            sum += ind_diff;
            if  (ind_diff > max_diff ) {
              max_diff = ind_diff;
              # printf("NR: %d; max: %g; ind: %g; sum: %g | Out: %g; refOut: %g\n", NR, max_diff, ind_diff, sum, $(i + r), $i) > "/dev/stderr";
            }
          }
        }
      }
      END {
        if (total_entries == 0) {
          printf("This file has no numeric entries!\n") > "/dev/stderr";
        }
        else {
          diff=sum/total_entries;
          if (diff > ENVIRON["avg_diff_limit"] || max_diff > ENVIRON["max_diff_limit"]) {
            print diff, max_diff;
          }
        }
      }' )
    fi

    if [ -n "$rel_max_difference" ]; then
      # Split by space and transform into the array
      difference=( $rel_max_difference )
      echo -e "> Numerical difference in $filename"
      # echo -e "$num_diff"
      echo -e "Average: ${difference[0]} ; Maximum: ${difference[1]} ${NC}"
      ret=1
    fi
    if [ -n "$txt_diff" ]; then
      echo -e "> Text difference in $filename"
      echo -e "$txt_diff"
      ret=1
    fi
  done
fi

# Files that are present only in reference or output folder
if [ -n "$only_files" ]; then
  echo -e "> $only_files"
  ret=1
fi
echo "----------- Comparison finished -----------"

exit $ret
