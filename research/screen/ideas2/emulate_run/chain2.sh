cd ~/zangir-research/prototype/screen/ideas2/emulate_run
while pgrep -f chain1.sh >/dev/null; do sleep 10; done
venv/bin/python runner.py --run synth_flash --tasks_dir synth_all --model gemini-2.5-flash --chunk 20 --cap 28
