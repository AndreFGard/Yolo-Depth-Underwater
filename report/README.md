Compile the PDF:
- typst compile report/main.typ

Watch and recompile on save:
- typst watch report/main.typ

# Tables and figures
To guarantee we version everything, that e verything is reproducible and that no results grow stale, ideally we should use python scripts in  report/build/ to generate visualizations and csv files.

That is, the ideal flow is:
1. a tool from scripts/ or src/yolo_depth_underwater/ generates a csv file (with rounded floats etc)
1. that table.csv is versioned
1. a build script reads that csv, formats it, takes whatever are the interesting metrics, names columns, orders rows, etc, and writes a corresponding report/tables/table.csv

These build scripts can import yolo_depth_underwater and use visualization code or other metrics that are shared there.

Each build script can implement its own policy for accepting not having the necessary inputs at hand, for hard to version input files.