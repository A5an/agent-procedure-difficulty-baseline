cd ~/zangir-research/prototype/screen/ideas2/emulate_run
while pgrep -f chain2.sh >/dev/null; do sleep 10; done
venv/bin/python runner.py --run real_flash --tasks_dir sop/data --model gemini-2.5-flash --chunk 20 --cap 16
