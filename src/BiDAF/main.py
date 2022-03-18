from args import get_setup_args
from setup import pre_process
from train import *
from test import *
# from test_dev import *
import pandas as pd
import copy

print('Trimmed context and Y/N included')

args_ = get_setup_args()
args_.train_file = 'data/train.json'
args_.dev_file = 'data/dev.json'
if args_.include_test_examples:
  args_.test_file = 'data/dev.json'
args_.glove_file = 'data/chinese_word_embeddings.txt'
pre_process(args_)

train_main(get_train_args())
test_main(get_test_args())

pred_file = 'save/test/test-01/test_submission.csv'
df = pd.read_csv(pred_file)
df.columns = ['id', 'answer']
df1 = df.replace(np.nan, '', regex=True)
df1.loc[df1['answer'] == 'Y', 'answer'] = 'YES'
df1.loc[df1['answer'] == 'YE', 'answer'] = 'YES'
df1.loc[df1['answer'] == 'N', 'answer'] = 'NO'
result = df1.to_json(orient="records")
parsed = json.loads(result)
with open("save/test/test-01/result.json", "w", encoding="utf-8") as f:
  json.dump(parsed, f, ensure_ascii=False, indent=2)
