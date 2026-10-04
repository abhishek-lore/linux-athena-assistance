#!/usr/bin/env fish
set SCRIPT_DIR (dirname (status --current-filename))
cd $SCRIPT_DIR
source venv/bin/activate.fish
set -x LD_LIBRARY_PATH (python -c "import nvidia.cublas as cb, nvidia.cudnn as cd; print(cb.__path__[0]+'/lib:'+cd.__path__[0]+'/lib')")
python main.py
