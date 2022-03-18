import os
import time
import random
import logging

import torch
import numpy as np
from absl import flags, app
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import (BertTokenizer, BertForQuestionAnswering,
                          ElectraTokenizer, ElectraForQuestionAnswering,
                          BertModel)
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)

from utils import load_and_cache_examples, RawResult, write_predictions
from evaluate import CJRCEvaluator
from model import BertMulti

logger = logging.getLogger(__name__)

AUTO_MODEL_CLASSES = {
    "bert-base-chinese": "bert-base-chinese",
    "ernie": "nghuyong/ernie-1.0",
    "bert-chinese-wwm": "hfl/chinese-bert-wwm",
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "electra": "hfl/chinese-legal-electra-small-discriminator",
}

MODEL_CONFIG = {
    "bert-base-chinese": (BertTokenizer, BertForQuestionAnswering),
    "ernie": (BertTokenizer, BertForQuestionAnswering),
    "bert-chinese-wwm": (BertTokenizer, BertForQuestionAnswering),
    "roberta": (BertTokenizer, BertForQuestionAnswering),
    "electra": (ElectraTokenizer, ElectraForQuestionAnswering),
}

AVAILABLE_MODEL_TYPE = list(AUTO_MODEL_CLASSES.keys())

FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", None,
                    "Input file for training in SQuaD format.")
flags.DEFINE_string("predict_file", None,
                    "Input file for evaluation in SQuaD format.")
flags.DEFINE_enum("model_type", None, AVAILABLE_MODEL_TYPE, "Model type")
flags.DEFINE_string("model_name_or_path", None,
                    "Pretrained models. Override model type")
flags.DEFINE_string("trained_weight", None, "Multitask trained weight")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_float(
    "null_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")
flags.DEFINE_integer("max_seq_length", 512, "Max sequence length.")
flags.DEFINE_integer("doc_stride", 128,
                     "Doc stride for splitting long documnets.")
flags.DEFINE_integer("max_query_length", 64, "Max question length.")
flags.DEFINE_boolean("do_train", False, "Run training")
flags.DEFINE_boolean("do_eval", False, "Run evaluation")
flags.DEFINE_integer("batch_size", 4, "Train batch size.")
flags.DEFINE_integer("eval_batch_size", 4, "Eval batch size.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
flags.DEFINE_integer(
    "gradient_accumulation_steps", 1,
    "Updates steps to accumulate before backward/update pass.")
flags.DEFINE_float("weight_decay", 0.0, "Weight decay.")
flags.DEFINE_float("adam_epsilon", 1e-8, "Epsilon for Adam optim.")
flags.DEFINE_float("max_grad_norm", 1.0, "Gradient clipping.")
flags.DEFINE_integer("num_epoch", 1, "Number of epoch.")
flags.DEFINE_integer("max_steps", -1, "Max steps (overide epoch).")
flags.DEFINE_integer("warmup_steps", 0, "Linear warmup over warmup_steps.")
flags.DEFINE_integer("n_best_size", 3, "Output n best predictions")
flags.DEFINE_integer("max_answer_length", 30, "Max length of answer.")
flags.DEFINE_integer("logging_steps", 0, "Log every X steps.")
flags.DEFINE_integer("eval_steps", 0, "Run eval every X steps.")
flags.DEFINE_integer("save_steps", 0, "Save checkpoint every X steps.")
flags.DEFINE_boolean("overwrite_output", False, "Overwrite output.")
flags.DEFINE_boolean("overwrite_cache", False, "Overwrite cache.")
flags.DEFINE_integer("seed", 123, "Random seed.")
flags.DEFINE_boolean("verbose", False, "Save verbose output.")

flags.mark_flag_as_required('train_file')
flags.mark_flag_as_required('predict_file')
flags.mark_flag_as_required('model_type')
flags.mark_flag_as_required('output_dir')

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
    FLAGS.num_epoch = FLAGS.max_steps // (len(train_dataloader) //
                                          FLAGS.gradient_accumulation_steps) + 1
  else:
    t_total = len(
        train_dataloader) // FLAGS.gradient_accumulation_steps * FLAGS.num_epoch

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

  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", FLAGS.num_epoch)
  logger.info("  Batch size = %d", FLAGS.batch_size)
  logger.info("  Total train batch size = %d",
              FLAGS.batch_size * FLAGS.gradient_accumulation_steps)
  logger.info("  Gradient Accumulation steps = %d",
              FLAGS.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)
  logger.info("  Learning rate = %d", FLAGS.learning_rate)

  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(int(FLAGS.num_epoch), desc="Epoch", disable=False)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  start_time = time.time()

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):
      model.train()
      batch = tuple(t.to(DEVICE) for t in batch)
      inputs = {
          'input_ids': batch[0],
          'attention_mask': batch[1],
          'token_type_ids': batch[2],
          'start_positions': batch[3],
          'end_positions': batch[4],
          'labels': batch[5],
      }

      outputs = model(inputs, do_train=True)
      loss = outputs[0]

      if FLAGS.gradient_accumulation_steps > 1:
        loss = loss / FLAGS.gradient_accumulation_steps

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_grad_norm)

      tr_loss += loss.item()
      if (step + 1) % FLAGS.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1

        if (FLAGS.logging_steps > 0) and (global_step % FLAGS.logging_steps
                                          == 0):
          # Log metrics
          if (FLAGS.eval_steps > 0) and (global_step % FLAGS.eval_steps == 0):
            results = evaluate(model, tokenizer)
            logger.info("eval result @ step global_step: {}".format(results))
            em_overall = results["overall"]["em"]
            f1_overall = results["overall"]["f1"]
            tb_writer.add_scalar('eval_{}'.format("em"), em_overall,
                                 global_step)
            tb_writer.add_scalar('eval_{}'.format("f1"), f1_overall,
                                 global_step)
          tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar('loss',
                               (tr_loss - logging_loss) / FLAGS.logging_steps,
                               global_step)
          logging_loss = tr_loss

      if FLAGS.max_steps > 0 and global_step > FLAGS.max_steps:
        epoch_iterator.close()
        break
    if FLAGS.max_steps > 0 and global_step > FLAGS.max_steps:
      train_iterator.close()
      break

  time_taken = time.time() - start_time
  logger.info("  Training time = %d", time_taken)

  tb_writer.close()

  return global_step, tr_loss / global_step


def evaluate(model, tokenizer, prefix=""):
  dataset, examples, features = load_and_cache_examples(FLAGS.predict_file,
                                                        tokenizer,
                                                        is_training=False)

  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset,
                               sampler=eval_sampler,
                               batch_size=FLAGS.eval_batch_size)

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

      outputs = model(inputs)

    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = eval_feature.unique_id

      result = RawResult(unique_id=unique_id,
                         start_logits=to_list(outputs[0][i]),
                         end_logits=to_list(outputs[1][i]),
                         q_logits=to_list(outputs[2][i]))

      all_results.append(result)

  # Compute predictions
  output_prediction_file = os.path.join(FLAGS.output_dir,
                                        "predictions_{}.json".format(prefix))
  output_nbest_file = os.path.join(FLAGS.output_dir,
                                   "nbest_predictions_{}.json".format(prefix))
  output_null_log_odds_file = os.path.join(FLAGS.output_dir,
                                           "null_odds_{}.json".format(prefix))

  write_predictions(examples, features, all_results, FLAGS.n_best_size,
                    FLAGS.max_answer_length, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file,
                    FLAGS.null_threshold)

  # Evaluate with CJRC competition evaluation script
  evaluator = CJRCEvaluator(FLAGS.predict_file)
  pred_data = CJRCEvaluator.preds_to_dict(output_prediction_file)
  results = evaluator.model_performance(pred_data)

  return results


def main(argv):
  if os.path.exists(FLAGS.output_dir) and os.listdir(
      FLAGS.output_dir) and FLAGS.do_train and not FLAGS.overwrite_output:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output to overcome."
        .format(FLAGS.output_dir))

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # logging
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

  eval_filename = os.path.splitext(os.path.basename(FLAGS.predict_file))[0]

  if FLAGS.do_train:
    log_filename = os.path.join(FLAGS.output_dir, "train.log")
  else:
    log_filename = os.path.join(FLAGS.output_dir,
                                "eval_{}.log".format(eval_filename))

  logging.basicConfig(filename=log_filename,
                      filemode='w',
                      format='%(name)s - %(levelname)s - %(message)s',
                      level=logging.INFO)

  logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S',
      level=logging.INFO)

  # Set seed
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  FLAGS.model_type = FLAGS.model_type.lower()

  if FLAGS.model_name_or_path:
    config = MODEL_CONFIG[FLAGS.model_type]
    tokenizer = config[0].from_pretrained(FLAGS.model_name_or_path)
    plm_model = config[1].from_pretrained(FLAGS.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # plm_model = AutoModel.from_pretrained("thunlp/Lawformer")
  else:
    tokenizer = BertTokenizer.from_pretrained(FLAGS.model_type)
    plm_model = BertModel.from_pretrained(FLAGS.model_type)

  model = BertMulti(plm_model)

  if FLAGS.trained_weight:
    model.load_state_dict(torch.load(FLAGS.trained_weight))

  model.to(DEVICE)

  logger.info("Training/evaluation parameters %s", argv)

  # Training
  if FLAGS.do_train:
    train_dataset, _, _ = load_and_cache_examples(FLAGS.train_file,
                                                  tokenizer,
                                                  is_training=True)
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    logger.info("Saving model checkpoint to %s", FLAGS.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(),
               os.path.join(FLAGS.output_dir, "model.pt"))
    tokenizer.save_pretrained(FLAGS.output_dir)
    plm_model.save_pretrained(FLAGS.output_dir)

  if FLAGS.do_eval:
    results = evaluate(model, tokenizer, prefix=FLAGS.model_type)
    logger.info("do_eval results: {}".format(results))


if __name__ == '__main__':
  app.run(main)
