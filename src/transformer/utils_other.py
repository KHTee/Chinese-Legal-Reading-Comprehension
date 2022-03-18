"""Some utils"""
import json
from collections import namedtuple


def write_some_samples(data_file, output_file, start_line, end_line):
  """Read the input and write certain lines to a new file"""
  with open(data_file, "rb") as f:
    data = json.load(f)["data"]

  examples = []
  for example in data[start_line:end_line + 1]:
    examples.append(example)

  d = {}
  d["data"] = examples

  with open(output_file, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)


def read_a_sample(data_file, line_num):
  """Read and print a sample for viewing"""
  with open(data_file, "r") as f:
    data = json.load(f)["data"]

  line = data[line_num]
  pretty_json = json.dumps(line, indent=2, ensure_ascii=False)
  print(pretty_json.encode("utf-8").decode())


def load_data(path):
  """Load context, question and answer into lists respectively"""
  contexts = []
  questions = []
  answers = []

  with open(path, 'rb') as f:
    data = json.load(f)

  for group in data['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)
  return contexts, questions, answers


def add_end_idx(answers, contexts):
  """Add end position to input data"""
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # for YES/NO ques, append -1 and the end. eg 'answer_start': -1, 'text': 'YES', 'answer_end': -1}
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif start_idx == -1:
      answer['answer_end'] = -1
