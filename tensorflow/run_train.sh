#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/lab/liujiepeng/MachineComprehension/DuReader/
python run.py --train \
    --train_files "../data/DuReader2.0/trainset/search.train.json" "../data/DuReader2.0/trainset/zhidao.train.json" \
    --dev_files "../data/DuReader2.0/devset/search.dev.json" "../data/DuReader2.0/devset/zhidao.dev.json" \
    --test_files="" \
    --vocab_dir=../data/DuReader2.0/vocab/ \
    --model_dir=../data/DuReader2.0/models/ \
    --result_dir=../data/DuReader2.0/results/ \
    --summary_dir=../data/DuReader2.0/summary/ \
    --log_path=../data/DuReader2.0/logfiles \
    --epochs=8