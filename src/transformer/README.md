# Chinese Legal Reading Comprehension with Multitask Learning

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


Result:
| Model | Dev EM | Dev F1 | Test EM | Test F1 |
| ----- | ----- | ----- | ----- | ----- |
| bert-base-chinese | 64.9 | 76.9 | 65.9 | 76.9 |  
| chinese-bert-wwm | 63.8 | 76.4 | 64.6 | 76.2 |
| ernie-1.0 | 66.8 | 79.1 | 67.3 | 79.0 |
| roberta-wwm-ext | 67.0 | 79.2 | 67.0 | 78.9 |
| electra-small-legal | 63.7 | 76.0 | 62.7 | 74.6 |
| electra-small | 58.8 | 71.8 | 58.2 | 70.9 |