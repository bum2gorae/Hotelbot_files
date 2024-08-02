from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import streamlit as st
import json
import torch
from nltk.tokenize import sent_tokenize
from nltk import download as nltkdownload

# NLTK 필요 라이브러리 다운로드
nltkdownload('punkt')  # 문장 토큰화를 위해 필요

# 데이터 로드 및 전처리
raw_data_ans = open('[Dataset] Module27 (ans).txt', 'r').read()
raw_data_ques = open('[Dataset] Module27(ques).txt', 'r').read()
sent_tokens_ques = sent_tokenize(raw_data_ques)
sent_tokens_ans = sent_tokenize(raw_data_ans)

data = []
for question, answer in zip(sent_tokens_ques, sent_tokens_ans):
    data.append({"question": question, "answer": answer})

# Dataset 객체 생성
dataset = Dataset.from_list(data)

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = ["question: " + q for q in examples['question']]
    targets = [a for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 토크나이저와 모델 로드
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 데이터셋 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# 모델 학습
trainer.train()


def generate_response(question):
    input_text = f"question: {question} </s>"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

try:
    st.header("Jane에게 Hotel에 관해 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            result = generate_response(question)
            st.write("Jane : " + result)
except Exception as e:
    st.error(f"An error occurred: {e}")
