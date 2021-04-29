# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

from absl import flags, app
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering, XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_squad import load_and_cache_examples, RawResult, RawResultExtended, write_predictions, write_predictions_extended

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", None,
                    "Input file for training in SQuaD format.")
flags.DEFINE_string("predict_file", None,
                    "Input file for evaluation in SQuaD format.")
flags.DEFINE_enum("model_type", None, ["bert", "xlnet", "xlm"], "Model type")
flags.DEFINE_string("model_name_or_path", None, "Pretrained models")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")
flags.DEFINE_integer("max_seq_length", 384, "Max sequence length.")
flags.DEFINE_integer("doc_stride", 128,
                     "Doc stride for splitting long documnets.")
flags.DEFINE_integer("max_query_length", 64, "Max question length.")
flags.DEFINE_boolean("do_train", False, "Run training")
flags.DEFINE_boolean("do_eval", False, "Run evaluation")
flags.DEFINE_boolean("evaluate_during_training", False,
                     "Run eval during training at each logging step.")
flags.DEFINE_integer("batch_size", 4, "Train batch size.")
flags.DEFINE_integer("eval_batch_size", 4, "Eval batch size.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
flags.DEFINE_integer(
    "gradient_accumulation_steps", 1,
    "Updates steps to accumulate before backward/update pass.")
flags.DEFINE_float("weight_decay", 0.0, "Weight decay.")
flags.DEFINE_float("adam_epsilon", 1e-8, "Epsilon for Adam optim.")
flags.DEFINE_float("max_grad_norm", 1.0, "Gradient clipping.")
flags.DEFINE_integer("num_train_epochs", 3, "Number of epoch.")
flags.DEFINE_integer("max_steps", -1, "Max steps (overide epoch).")
flags.DEFINE_integer("warmup_steps", 0, "Linear warmup over warmup_steps.")
flags.DEFINE_integer("n_best_size", 20, "Output n best predictions")
flags.DEFINE_integer("max_answer_length", 30, "Max length of answer.")
flags.DEFINE_integer("logging_steps", 50, "Log every X updates steps.")
flags.DEFINE_integer("save_steps", 50, "Save checkpoint every X updates steps.")
flags.DEFINE_boolean("eval_all_checkpoints", False, "Evaluate all checkpoints.")
flags.DEFINE_boolean("overwrite_output_dir", False,
                     "Overwrite output directory.")
flags.DEFINE_integer("seed", 123, "Random seed.")

flags.mark_flag_as_required('train_file')
flags.mark_flag_as_required('predict_file')
flags.mark_flag_as_required('model_type')
flags.mark_flag_as_required('model_name_or_path')
flags.mark_flag_as_required('output_dir')

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_list(tensor):
    """Convert tensor to numpy"""
    return tensor.detach().cpu().tolist()


def train(train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=FLAGS.batch_size)

    if FLAGS.max_steps > 0:
        t_total = FLAGS.max_steps
        FLAGS.num_train_epochs = FLAGS.max_steps // (
            len(train_dataloader) // FLAGS.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // FLAGS.gradient_accumulation_steps * FLAGS.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay': FLAGS.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay': 0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=FLAGS.learning_rate,
                      eps=FLAGS.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=FLAGS.warmup_steps,
                                     t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", FLAGS.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", FLAGS.batch_size)
    logger.info("  Total train batch size = %d",
                FLAGS.batch_size * FLAGS.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                FLAGS.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(FLAGS.num_train_epochs),
                            desc="Epoch",
                            disable=False)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {
                'input_ids':
                    batch[0],
                'attention_mask':
                    batch[1],
                'token_type_ids':
                    None if FLAGS.model_type == 'xlm' else batch[2],
                'start_positions':
                    batch[3],
                'end_positions':
                    batch[4]
            }
            if FLAGS.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]

            if FLAGS.gradient_accumulation_steps > 1:
                loss = loss / FLAGS.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           FLAGS.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % FLAGS.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if FLAGS.logging_steps > 0 and global_step % FLAGS.logging_steps == 0:
                    # Log metrics
                    if FLAGS.evaluate_during_training:
                        evaluate(model, tokenizer)
                        # results = evaluate(model, tokenizer)
                        # for key, value in results.items():
                        #     tb_writer.add_scalar('eval_{}'.format(key), value,
                        #                          global_step)
                    tb_writer.add_scalar('lr',
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) /
                                         FLAGS.logging_steps, global_step)
                    logging_loss = tr_loss

                if FLAGS.save_steps > 0 and global_step % FLAGS.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        FLAGS.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if FLAGS.max_steps > 0 and global_step > FLAGS.max_steps:
                epoch_iterator.close()
                break
        if FLAGS.max_steps > 0 and global_step > FLAGS.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(FLAGS.predict_file,
                                                          tokenizer,
                                                          is_training=False,
                                                          output_examples=True)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=FLAGS.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", FLAGS.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids':
                    batch[0],
                'attention_mask':
                    batch[1],
                'token_type_ids':
                    None if FLAGS.model_type == 'xlm' else
                    batch[2]  # XLM don't use segment_ids
            }
            example_indices = batch[3]
            if FLAGS.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = eval_feature.unique_id
            if FLAGS.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(
                    unique_id=unique_id,
                    start_top_log_probs=to_list(outputs[0][i]),
                    start_top_index=to_list(outputs[1][i]),
                    end_top_log_probs=to_list(outputs[2][i]),
                    end_top_index=to_list(outputs[3][i]),
                    cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(FLAGS.output_dir,
                                          "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(FLAGS.output_dir,
                                     "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(FLAGS.output_dir,
                                             "null_odds_{}.json".format(prefix))

    if FLAGS.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results,
                                   FLAGS.n_best_size, FLAGS.max_answer_length,
                                   output_prediction_file, output_nbest_file,
                                   output_null_log_odds_file,
                                   model.config.start_n_top,
                                   model.config.end_n_top, tokenizer)
    else:
        write_predictions(examples, features, all_results, FLAGS.n_best_size,
                          FLAGS.max_answer_length, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file,
                          FLAGS.null_score_diff_threshold)

    # # Evaluate with the official SQuAD script
    # evaluate_options = EVAL_OPTS(data_file=FLAGS.predict_file,
    #                              pred_file=output_prediction_file,
    #                              na_prob_file=output_null_log_odds_file)
    # results = evaluate_on_squad(evaluate_options)
    # return results


def main(argv):
    if os.path.exists(FLAGS.output_dir) and os.listdir(
            FLAGS.output_dir
    ) and FLAGS.do_train and not FLAGS.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(FLAGS.output_dir))

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    # Set seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    FLAGS.model_type = FLAGS.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[FLAGS.model_type]
    config = config_class.from_pretrained(FLAGS.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(FLAGS.model_name_or_path)
    model = model_class.from_pretrained(
        FLAGS.model_name_or_path,
        from_tf=bool('.ckpt' in FLAGS.model_name_or_path),
        config=config)

    model.to(DEVICE)

    logger.info("Training/evaluation parameters %s", argv)

    # Training
    if FLAGS.do_train:
        train_dataset = load_and_cache_examples(FLAGS.train_file,
                                                tokenizer,
                                                is_training=True,
                                                output_examples=False)
        global_step, tr_loss = train(train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

        # Save the trained model and the tokenizer
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)

        logger.info("Saving model checkpoint to %s", FLAGS.output_dir)
        model.save_pretrained(FLAGS.output_dir)
        tokenizer.save_pretrained(FLAGS.output_dir)

        # # torch.save(FLAGS, os.path.join(FLAGS.output_dir, 'training_FLAGS.bin'))

        # model = model_class.from_pretrained(FLAGS.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(FLAGS.output_dir)
        # model.to(DEVICE)

    if FLAGS.do_eval:
        evaluate(model, tokenizer, prefix="")

    # # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    # results = {}
    # if FLAGS.do_eval:
    #     checkpoints = [FLAGS.output_dir]
    #     if FLAGS.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(
    #                 glob.glob(FLAGS.output_dir + '/**/' + WEIGHTS_NAME,
    #                           recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
    #             logging.WARN)  # Reduce model loading logs

    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)

    #     for checkpoint in checkpoints:
    #         # Reload the model
    #         global_step = checkpoint.split(
    #             '-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(DEVICE)

    #         # Evaluate
    #         evaluate(model, tokenizer, prefix=global_step)
    #         # result = evaluate(model, tokenizer, prefix=global_step)

    #         # result = dict(
    #         #     (k + ('_{}'.format(global_step) if global_step else ''), v)
    #         #     for k, v in result.items())
    #         # results.update(result)

    # logger.info("Results: {}".format(results))

    # return results


if __name__ == '__main__':
    app.run(main)
