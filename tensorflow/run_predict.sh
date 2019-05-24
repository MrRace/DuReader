#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/lab/liujiepeng/MachineComprehension/DuReader/
python run.py --predict \
    --dev_files "../data/DuReader2.0/devset/search.dev.json" "../data/DuReader2.0/devset/zhidao.dev.json" \
    --test_files "../data/DuReader2.0/test2set/search.test1.json" "../data/DuReader2.0/test2set/search.test2.json" "../data/DuReader2.0/test2set/zhidao.test1.json"  "../data/DuReader2.0/test2set/zhidao.test2.json" \
    --vocab_dir=../data/DuReader2.0/vocab/ \
    --model_dir=../data/DuReader2.0/models/ \
    --result_dir=../data/DuReader2.0/results/ \
    --summary_dir=../data/DuReader2.0/summary/ \
    --log_path=../data/DuReader2.0/logfiles \
    --epochs=10