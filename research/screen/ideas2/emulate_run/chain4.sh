cd ~/zangir-research/prototype/screen/ideas2/emulate_run
venv/bin/python runner.py --run synth_flash --tasks_dir synth_all --model gemini-2.5-flash --chunk 20 --cap 28
venv/bin/python runner.py --run real_flash --tasks_dir sop/data --model gemini-2.5-flash --chunk 20 --cap 16
