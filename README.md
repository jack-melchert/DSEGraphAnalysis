# APEX
Subgraph mining, maximal independent set analysis, and graph merging flow for coreir graphs produced from Halide Applications. Use in the PE design space exploration project to identify potential interesting candidate PE architectures.

Usage:
```
python dse_graph_analysis.py -f <app>_compute.json 0 1 2 … 
```
```<app>_compute.json``` must be a CoreIR compute file produced by Halide-to-Hardware (https://github.com/StanfordAHA/Halide-to-Hardware)
  
You can choose to merge subgraphs from multiple applications:
```
python dse_graph_analysis.py -f <app1>_compute.json 0 1 2 … -f <app2>_compute.json 0 1 2 … 
```
  
This produces the PEak code and verilog for the customized PE in outputs/
The visualization of the PE architecture is arch_graph.pdf
You can examine which subgraphs are merged into the PE by looking at the pdfs produced in the pdf/ folder
The subgraph indexes that you pass into dse_graph_analysis.py for each application will be merged to produce the final PE architecture
  
