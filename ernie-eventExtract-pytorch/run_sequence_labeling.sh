
data_dir=$1
conf_path=$2
ckpt_dir=$3
predict_data=$4
learning_rate=$5
is_train=$6
max_seq_len=$7
batch_size=$8
epoch=${9}
pred_save_path=${10}
num_head_att=${11}
out_size_att=${12}

if [ "$is_train" = True ]; then
    python sequence_labeling.py \
            --num_epoch ${epoch} \
            --learning_rate ${learning_rate} \
            --tag_path ${conf_path} \
            --train_data ${data_dir}/train.tsv \
            --dev_data ${data_dir}/dev.tsv \
            --test_data ${data_dir}/test.tsv \
            --predict_data ${predict_data} \
            --do_train True \
            --do_predict False \
            --max_seq_len ${max_seq_len} \
            --batch_size ${batch_size} \
            --skip_step 100 \
            --valid_step 200 \
            --checkpoints ${ckpt_dir} \
            --init_ckpt ${ckpt_dir}/best.pt \
            --predict_save_path ${pred_save_path} \
            --device cuda \
            --num_head ${num_head_att} \
            --out_size ${out_size_att} 

else
    python sequence_labeling.py \
            --num_epoch ${epoch} \
            --learning_rate ${learning_rate} \
            --tag_path ${conf_path} \
            --train_data ${data_dir}/train.tsv \
            --dev_data ${data_dir}/dev.tsv \
            --test_data ${data_dir}/test.tsv \
            --predict_data ${predict_data} \
            --do_train False \
            --do_predict True \
            --max_seq_len ${max_seq_len} \
            --batch_size ${batch_size} \
            --skip_step 100 \
            --valid_step 200 \
            --checkpoints ${ckpt_dir} \
            --init_ckpt ${ckpt_dir}/best.pt \
            --predict_save_path ${pred_save_path} \
            --device cuda
            --num_head ${num_head_att} \
            --out_size ${out_size_att} 

fi
