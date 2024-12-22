export CUDA_VISIBLE_DEVICES=0


mkdir -p json_spe_contrast/csds/all/
mkdir -p json_spe_contrast/csds/user/
mkdir -p json_spe_contrast/csds/agent/
mkdir -p json_spe_contrast/csds/final/


# bert abs final
mkdir -p bert_data_spe_contrast/all/
mkdir -p bert_data_spe_contrast/user/
mkdir -p bert_data_spe_contrast/agent/
mkdir -p bert_data_spe_contrast/final/

python ./preprocess.py -raw_path json_spe_contrast/csds/all/ -save_path bert_data_spe_contrast/all/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/user/ -save_path bert_data_spe_contrast/user/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/agent/ -save_path bert_data_spe_contrast/agent/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False
python ./preprocess.py -raw_path json_spe_contrast/csds/final/ -save_path bert_data_spe_contrast/final/ -bert_dir ./bert_chinese -log_file logs/preprocess.log -add_ex_label False


nohup python train.py -task abs -mode train -data_path bert_data_spe_contrast/all/ -dec_dropout 0.2  -model_path output/bert_spe_contrast_csds -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 1000 -batch_size 1 -train_steps 5000 -report_every 100 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -log_file logs/bert_spe_contrast_train_csds.log -finetune_bert True -device_id 3 --role_lambda 1.0 --temperature 0.07 > train_bert_spe_contrast_csds.log 2>&1 &

python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -data_path bert_data_spe_contrast/all -log_file logs/bert_abs_val_spe_contrast.log -model_path output/bert_spe_contrast_csds -sep_optim true -use_interval true -max_pos 512 -max_length 300 -alpha 0.95 -min_length 5 -result_path logs/bert_abs_val_spe_contrast_csds -temp_dir temp/ -test_all=True -device_id 1

python train.py -task abs -mode test -batch_size 10 -test_batch_size 10 -log_file logs/bert_test_spe_contrast.log -test_from output/bert_spe_contrast_csds/model.pt -sep_optim true -use_interval true -device_id 2 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path logs/bert_spe_contrast_csds  -temp_dir temp/
