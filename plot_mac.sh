#! /bin/bash

# python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-3ch.yaml -test_only -student_only
# python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-6ch.yaml -test_only -student_only
# python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-9ch.yaml -test_only -student_only
# python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-12ch.yaml -test_only -student_only
# python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-15ch.yaml -test_only -student_only

python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-2ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-4ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-6ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-8ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-10ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-12ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-14ch-8K.yaml -test_only -student_only
python ./ops_calculator.py --config ./config/ghnd_kd-finetune/custom_resnet18_1d_from_resnet18_1d-16ch-8K.yaml -test_only -student_only