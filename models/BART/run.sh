
mkdir -p json_spe_contrast/csds/all/
mkdir -p json_spe_contrast/csds/user/
mkdir -p json_spe_contrast/csds/agent/
mkdir -p json_spe_contrast/csds/final/


python ./prepro/convert_json_format_spe_contrast_csds.py


mkdir -p bert_data_spe_contrast/all/
mkdir -p bert_data_spe_contrast/user/
mkdir -p bert_data_spe_contrast/agent/
mkdir -p bert_data_spe_contrast/final/



python ./preprocess.py -raw_path json_spe_contrast/csds/all/ -save_path bert_data_spe_contrast/all/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/user/ -save_path bert_data_spe_contrast/user/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/agent/ -save_path bert_data_spe_contrast/agent/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/final/ -save_path bert_data_spe_contrast/final/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False



nohup python ./train_abs_spe_contrast.py -mode train -data_path bert_data_spe_contrast_2/all/ -model_path models/bart_spe_contrast_csds_new_t01_rl062 --model_name ./bart_base_chinese --sample_num 3 --role_lambda 0.6 --tokenizer_name ./bert_chinese --max_turn_range 8 --temperature 0.1 -device_id 1 >train_bart_spe_contrast_csds_t01_rl_062.log 2>&1 &
python ./train_abs_spe_contrast.py -mode test -data_path bert_data_spe_contrast_2/all/ -model_path ./models/bart_spe_contrast_csds --model_name ./bart_base_chinese --checkpoint_path ./models/bart_spe_contrast_csds/abs_bart_base2.tar -device_id 2




