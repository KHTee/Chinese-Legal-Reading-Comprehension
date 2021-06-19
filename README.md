# Chinese Legal Reading Comprehension with Multitask Learning
Repo for NLP 2021 Chinese Legal Reading Comprehension with Multitask Learning

Train:
```
python3 src/train.py --train_file $train_data \
--predict_file $test_data \
--model_type $model_type \
--output_dir output \
--max_seq_length 512 \
--doc_stride 128 \
--max_query_length 50 \
--do_train \
--learning_rate 4e-5 \
--num_epoch 2 \
--batch_size 8 \
--eval_batch_size 128 \
--logging_steps 100 \
--overwrite_output
```

Test:
```
python3 src/train.py --train_file $train_data \
--predict_file $test_data \
--model_type $model_type \
--trained_weight $trained_model \
--output_dir output \
--eval_batch_size 128 \
--max_seq_length 512 \
--doc_stride 128 \
--max_query_length 50 \
--do_eval \
--overwrite_output
```