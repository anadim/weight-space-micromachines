# Optional per-machine environment (the original runs sourced a file like this to put a private venv and Python headers
# on the path; Triton needs Python.h to build its launcher). Edit for your machine, e.g.:
#   export PATH=/path/to/venv/bin:$PATH
#   export C_INCLUDE_PATH=/path/to/python3.12/include
export TOKENIZERS_PARALLELISM=false
