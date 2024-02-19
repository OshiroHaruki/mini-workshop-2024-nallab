from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import torch
import evaluate
import numpy as np

# データセットを用意
dataset = load_dataset("alt", "alt-jp")

# tokenizerを用意
en_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
jp_tokenizer = AutoTokenizer.from_pretrained("ku-nlp/bart-base-japanese")
jp_tokenizer.pad_token = en_tokenizer.pad_token

# データセットの前処理(テキストのtokenizer(id化)とattention_mask付与)
tr_data = []
ts_data = []
ev_data = []

for data in dataset["train"]:
    if data["jp_tokenized"] is None:
        continue
    input = en_tokenizer(data["en_tokenized"],  padding='max_length', max_length=120, truncation=True)
    tr_data.append({
                    "input_ids" : input["input_ids"], 
                    "attention_mask" : input["attention_mask"], 
                    "labels" : jp_tokenizer(data["jp_tokenized"],  padding='max_length', max_length=120, truncation=True)["input_ids"]
                    })

for data in dataset["validation"]:
    if data["jp_tokenized"] is None:
        continue
    input = en_tokenizer(data["en_tokenized"],  padding='max_length', max_length=120, truncation=True)
    ev_data.append({
                    "input_ids" : input["input_ids"], 
                    "attention_mask" : input["attention_mask"], 
                    "labels" : jp_tokenizer(data["jp_tokenized"],  padding='max_length', max_length=120, truncation=True)["input_ids"]
                    })

for data in dataset["test"]:
    if data["jp_tokenized"] is None:
        continue
    input = en_tokenizer(data["en_tokenized"],  padding='max_length', max_length=120, truncation=True)
    ts_data.append({
                    "input_ids" : input["input_ids"], 
                    "attention_mask" : input["attention_mask"], 
                    "labels" : jp_tokenizer(data["jp_tokenized"],  padding='max_length', max_length=120, truncation=True)["input_ids"]
                    })

# modelを用意
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# BLEUを利用するための関数を用意
metric = evaluate.load("bleu")  ## bleu→0~1で高いほどいい値(正解の翻訳文に近い)
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels
def compute_bleu(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # -100でパディングされているためpad_token_idへ置き換える
    preds = np.where(preds != -100, preds, jp_tokenizer.pad_token_id)
    decoded_preds = jp_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, jp_tokenizer.pad_token_id)
    decoded_labels = jp_tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}

    prediction_lens = [
        np.count_nonzero(pred != jp_tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# trainのパラメータ設定
training_args = Seq2SeqTrainingArguments(
    output_dir= "./",  # 出力先、これだけ必須
    # --- 以下学習時のオプション ---
    predict_with_generate=True,  # 評価にBLEUやROUGEなどを使う時には必要.
    evaluation_strategy="epoch", # no(default)/epoch
    per_device_train_batch_size= 16, # defaultは8, GPU1枚ごとのバッチサイズを指定
    per_device_eval_batch_size= 16, # defaultは8, 上に同じく
    save_strategy="epoch",  # no/steps(default)/epochで指定
    save_total_limit=2,  # モデルをいくつまで保存するか(容量に余裕があるならいくつか保存しておくと良いと思う)
    load_best_model_at_end=True,  # 学習終了時、最も良かった(metric_for_best_modelで指定した)モデルをloadする
    logging_strategy = "epoch", # logging頻度
    learning_rate=5e-5,  # 学習率、defaultは5e-5
    do_train = True,
    do_eval = True,
    num_train_epochs=1, # epoch数、簡単化で削ってるけどお試しするなら3~10epoch程度に増やすなど
    generation_max_length = 120,  # 生成文章の最大の長さ
    # --- 以下その他いくつか紹介 ---
    # warmup_steps=200,  # 指定したstepまで学習率を線形増加させる、defaultは0
    # metric_for_best_model = "bleu",  # 何を良いモデルとするかの指標
    # greater_is_better = True, # metric_for_best_modelで設定した指標は大きいほうが良いのか(BLEUなど)小さいほうがいいのか(Lossなど)を指定
    # 沢山あるのでHuggingFaceのTrainerクラスのDocumentを参照.
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_bleu,
    train_dataset=tr_data,
    eval_dataset=ev_data,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  ## early_stoppingを有効化する場合はコメント解除
)

# train!
trainer.train()
trainer.evaluate()

# テストデータからランダムに10件生成してみる
with open("generate_test_10.txt", "w") as f:
    for i in range(10):
        t = torch.tensor(ts_data[i]["input_ids"])
        t = torch.reshape(t, (1,-1)).to(device)
        outputs = model.generate(t, max_length=120)
        en_text = en_tokenizer.decode(t[0], skip_special_tokens=True)
        label_text = jp_tokenizer.decode(ts_data[i]["labels"], skip_special_tokens=True)
        generate_text = jp_tokenizer.decode(outputs[0], skip_special_tokens=True)
        f.write(f"入力 : {en_text} \n")
        f.write(f"正解 : {label_text} \n")
        f.write(f"生成文 : {generate_text}\n")
        f.write("  ==========  \n")
