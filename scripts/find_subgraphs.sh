# takes in .json coreIR file, outputs grami_in.txt and op_types.txt
echo "Starting .dot conversion..."
python scripts/convert_dot.py $1
echo "Finished .dot conversion"

# Takes in grami_in.txt produces orig_graph.pdf
echo "Graphing original graph"
python scripts/graph_output.py .temp/grami_in.txt orig_graph

# Takes in grami_in.txt and subgraph support, produces Output.txt
echo "Starting GraMi subgraph mining..."
cd GraMi
./grami -f ../.temp/grami_in.txt -s $2 -t 1 -p 1 > grami_log.txt
cd ../
echo "Finished GraMi subgraph mining"

# Takes in Output.txt produces subgraphs.pdf
echo "Graphing subgraphs"
python scripts/graph_output.py ./GraMi/Output.txt subgraphs

# Takes in Output.txt, produces a bunch of .json arch files in /subgraph/
echo "Converting subgraph files to arch format"
python scripts/convert_subgraphs_to_arch.py ./GraMi/Output.txt

# Finds maximal independent set, looks at grami_in.txt and Output.txt
echo "Starting maximal independent set analysis, this could take a while..."
python scripts/find_maximal_ind_set.py
echo "Finished maximal independent set analysis"