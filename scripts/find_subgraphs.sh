#!/bin/bash

USAGE="Usage: $0 subgraph-support graph1 graph2 graph3 ... graphN"

if [ "$#" == "0" ]; then
        echo "$USAGE"
        exit 1
fi

echo "Cleaning outputs/ and .temp/"
rm -rf outputs/
rm -rf .temp/
rm -rf pdf/

PARAMS=""
FARG=""
NARG=""
while (( "$#" )); do
  case "$1" in
    -f)
      FARG="$FARG $2"
      shift 2
      ;;
    -n)
      NARG="$NARG $2"
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

# takes in .json coreIR file, outputs grami_in.txt and op_types.txt
echo "Starting .dot conversion..."
python scripts/convert_dot.py ${FARG}
echo "Finished .dot conversion"

ITER=0
for I in $NARG
do
        # Takes in grami_in.txt produces orig_graph.pdf
        echo "Graphing original graph"
        python scripts/graph_output.py .temp/grami_in_$ITER.txt orig_graph

        # Takes in grami_in.txt and subgraph support, produces Output.txt
        echo "Starting GraMi subgraph mining..."
        cd GraMi
        ./grami -f ../.temp/grami_in_$ITER.txt -s $I -t 1 -p 0 > grami_log.txt
        cd ../
        echo "Finished GraMi subgraph mining"

		touch .temp/grami_out.txt
		cat GraMi/Output.txt >> .temp/grami_out.txt

        # Takes in Output.txt produces subgraphs.pdf
        echo "Graphing subgraphs"
        python scripts/graph_output.py ./GraMi/Output.txt subgraphs_$ITER
        
		ITER=$(expr $ITER + 1)
done

# Takes in Output.txt, produces a bunch of .json arch files in /subgraph/
echo "Converting subgraph files to arch format"
python scripts/convert_subgraphs_to_arch.py .temp/grami_out.txt

# # Finds maximal independent set, looks at grami_in.txt and Output.txt
# echo "Starting maximal independent set analysis, this could take a while..."
# python scripts/find_maximal_ind_set.py > ./outputs/subgraph_eval.txt
# echo "Finished maximal independent set analysis"
