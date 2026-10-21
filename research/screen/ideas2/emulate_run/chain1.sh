cd ~/zangir-research/prototype/screen/ideas2/emulate_run
venv/bin/python runner.py --run synth_lite --tasks_dir synth_all --model gemini-2.5-flash-lite --chunk 20 --cap 12
venv/bin/python runner.py --run real_lite --tasks_dir sop/data --model gemini-2.5-flash-lite --chunk 20 --cap 12
