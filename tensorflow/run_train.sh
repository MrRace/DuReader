#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/lab/liujiepeng/MachineComprehension/DuReader/
python run.py --train \
    --embed_size=400 \
	--prepared_dir=../data/DuReader2.0/prepared \
	--segmented_dir=../data/DuReader2.0/segmented \
	--vocab_dir=../data/DuReader2.0/vocab/ \
	--model_dir=../data/DuReader2.0/models/ \
	--result_dir=../data/DuReader2.0/results/ \
	--summary_dir=../data/DuReader2.0/summary/ \
	--log_path=../data/DuReader2.0/logfiles \
	--epochs=30