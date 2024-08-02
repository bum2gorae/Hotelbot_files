from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import operator
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import download as nltkdownload
import torch

# nltkdownload('punkt')   # 토큰화 프로그램 처음 사용
# nltkdownload('wordnet') # 표제어 추출 프로그램 처음 사용

# 데이터 로드 및 전처리
raw_data_ans = open('[Dataset] Module27 (ans).txt', 'r').read()
raw_data_ques = open('[Dataset] Module27(ques).txt', 'r').read()
sent_tokens_ques = sent_tokenize(raw_data_ques)
sent_tokens_ans = sent_tokenize(raw_data_ans)

ques_ans_pairs = {sent_tokens_ques[i].lower(): sent_tokens_ans[i] for i in range(len(sent_tokens_ans))}

# 인사말 설정
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# GPT-2 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits.mean(dim=1).detach().numpy()

def generate_response(question, q):
    probs = dict()
    question_vec = encode_text(question, tokenizer, model)
    
    for i in q:
        i_vec = encode_text(i, tokenizer, model)
        sim = cosine_similarity(question_vec, i_vec)
        probs[i] = sim[0][0]

    sorted_d = dict(sorted(probs.items(), key=operator.itemgetter(1), reverse=True))
    return list(sorted_d.items())[0]

def response(question):
    question = question.lower()
    if question != 'bye':
        if question in ['thanks', 'thank you']:
            return "You are welcome.."
        else:
            if greeting(question):
                return greeting(question)
            else:
                resp = generate_response(question, sent_tokens_ques)
                return ques_ans_pairs[resp[0].lower()]
    else:
        return "Bye! take care.."

try:
    st.header("Jane에게 Hotel에 관해 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            result = response(question)
            st.write("Jane : " + result)
except Exception as e:
    st.error(f"An error occurred: {e}")