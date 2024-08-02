from gensim.models.doc2vec import Doc2Vec
import os
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk import download as nltkdownload
from sklearn.metrics.pairwise import cosine_similarity
import random
import operator
import streamlit as st
from transformers import BertTokenizer, BertModel

nltkdownload('punkt')   # 토큰화 프로그램 처음 사용
nltkdownload('wordnet') # 표제어 추출 프로그램 처음 사용

raw_data_ans = open('[Dataset] Module27 (ans).txt','r').read()
raw_data_ques = open('[Dataset] Module27(ques).txt','r').read()
sent_tokens_ques = sent_tokenize(raw_data_ques)
sent_tokens_ans = sent_tokenize(raw_data_ans)

ques_ans_pairs = {}
for i in range(len(sent_tokens_ans)):
    ques_ans_pairs[sent_tokens_ques[i].lower()] = sent_tokens_ans[i]

# 2.1.1 입력 및 응답 목록 작성
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up","hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

# 인사말을 수신하고 반환하는 함수 만들기
def greeting(sentence):
    for word in sentence.split(): # 문장의 각 단어를 살펴봅니다.
        if word.lower() in GREETING_INPUTS: # 단어가 GREETING_INPUT와 일치하는지 확인합니다.
            return random.choice(GREETING_RESPONSES) # Greeting_Response로 답장합니다.
    

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def calc_prob(v1, q):
    probs = dict()
    for i in q:
        inputs = tokenizer(i.lower(), return_tensors='pt')
        v2 = model(**inputs).last_hidden_state.mean(dim=1)
        sim = cosine_similarity(v1.detach().numpy(), v2.detach().numpy())
        probs[i] = sim[0][0]

    sorted_d = dict( sorted(probs.items(), key=operator.itemgetter(1),reverse=True))

    # 유사도가 가장 높은 답변을 반환합니다.
    return list(sorted_d.items())[0]

def response(question):
    # 사용자의 입력을 받고, 소문자로 변환합니다.
    question=question.lower()
    # 사용자가 'bye'를 입력하기 전까지 다음 동작을 반복합니다:
    if(question!='bye'):
        # 사용자가 'thanks' 또는 'thank you'를 입력하면 대화를 종료하고 "You are welcome.."을 출력합니다.
        if(question=='thanks' or question=='thank you' ):
            return "You are welcome.."
        else:
            # 사용자의 인사말에 대한 응답이 있는 경우 해당 응답을 출력합니다.
            if(greeting(question)!=None):
                return greeting(question)

            # 그렇지 않은 경우, 주어진 질문에 대한 답변을 계산하고 출력합니다.
            else:
                inputs = tokenizer(question, return_tensors='pt')
                v1 = model(**inputs).last_hidden_state.mean(dim=1)
                resp = calc_prob(v1, sent_tokens_ques)
                return ques_ans_pairs[resp[0].lower()]

    else:
        return "Bye! take care.."
        
try :        
    #Question
    st.header("Jane에게 Hotel에 관해 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):

            result = response(question)
            st.write("Jane : "+result)
            
except Exception as e:
    st.error(f"An error occurred: {e}")