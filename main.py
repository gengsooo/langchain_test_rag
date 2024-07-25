from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# llm = ChatOpenAI(temperature=0, streaming=True, max_tokens=100)
llm = ChatOpenAI(temperature=0, max_tokens=100)

template = """
    {content}에 대한 설명을 {limit}자 이내로 요약해줘
"""

content = "태윤"

############### PromptTemplate ################
prompt = PromptTemplate(
    template=template,
    input_variables=["content"],
    partial_variables={"limit": 20}
)
prompt_template = prompt.format(content=content)

############### Document Loader ################
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

loader = PyPDFLoader("static/langchain_exam.pdf")
pages = loader.load_and_split()

# print(pages[0])

webLoader = WebBaseLoader("https://n.news.naver.com/mnews/article/023/0003848201?sid=102")
webPages = webLoader.load_and_split()

# print(webPages[0])

############### Text Splitter ################

# 그냥 webPages[0]을 넣으면 Document 객체가 들어가서 타입 오류가 발생, 따라서 text 형태로 변환해서 넣어준다
web_text_content = webPages[0].page_content

# CharacterTextSplitter: 구분자 1개를 기준으로 문서를 분할
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# text_splitter = CharacterTextSplitter(
#     separator='\n', # 구분자: 문장을 어떤 기준으로 하위 청크로 나눌지 결정
#     chunk_size=1000,
#     chunk_overlap=100, # 청크간의 앞뒤로 겹치는 글자 수
#     length_function=len # 청크 사이즈를 글자의 길이로 구분
# )

# RecursiveCharacterTextSplitter: 여러개의 구분자를 기준으로 문서를 분할 (max_token의 수를 최대한 지킬수 있음)
# 먼저 \n\n(줄바꿈)을 기준으로 분할하고 청크가 여전히 너무 클 경우 그 다음에 \n(줄바꿈)을 기준으로 분할, 이래도 너무 클 경우 단어단위로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, # 청크간의 앞뒤로 겹치는 글자 수
    length_function=len # 청크 사이즈를 글자의 길이로 구분
)


texts1 = text_splitter.split_text(web_text_content)
# print(texts1[0])
# print("="*100)
# print(texts1[1])
# print("="*100)
# print(texts1[2])

# 이 외에도 많은 text splitter가 존재함 (코드를 분할하는 스플리터 등등)

############### Text Embeddings ################
# Text Embeddings: 텍스트를 숫자로 변환하여 문장 간의 유사성을 비교할 수 있도록 해줌
# 대부분의 경우 대용량의 말뭉치를 통해 사전학습된 모델을 사용함
# 임베딩 모델의 능력이 중요함, 모델마다 지원하는 언어가 다름
# ex) OPENAI: text-embedding-ada-002, HuggingFace: ko-sbert-nli

from langchain_community.embeddings import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        "안녕하세요",
        "제 이름은 홍길동입니다",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다",
        "Hello World"
    ]
)

# 임베딩된 길이와 임베딩된 벡터의 길이를 출력
# print(len(embeddings), len(embeddings[0]))

# 벡터 유사도 확인
embedded_query_q = embeddings_model.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = embeddings_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")
# print(len(embedded_query_q), len(embedded_query_a))
from numpy import dot
from numpy.linalg import norm
import numpy as np

# 벡터 유사도를 계산할때 가장 많이 사용하는게 cosin similarity
def cos_sim(A,B):
    return dot(A, B)/(norm(A)*norm(B))

# 질문과 답변 문장에 대한 벡터 유사도
# print(cos_sim(embedded_query_q, embedded_query_a))
# 질문과 1번째 임베딩된 문장에 대한 벡터 유사도
# print(cos_sim(embedded_query_q, embeddings[1]))
# 질문과 3번째 임베딩된 문장에 대한 벡터 유사도
# print(cos_sim(embedded_query_q, embeddings[3]))

# 1에 가까울수록 유사도가 높음

############### Vectorstores ################
# 임베딩 모델을 통해 수치화한 텍스트들을 저장
# 사용자의 질문이 들어왔을때 해당 질문과 가장 유사한 문장을 이 저장소에서 찾아서 반환
# Vectorestores 여러 종류가 존재, 대표적으로 Chroma, Faiss 등이 있음
# CRUD(Create, Read, Update, Delete) 기능을 제공함
# 기본적으로 vectorstore는 벡터를 일시적으로 저장, 텍스트와 임베딩 함수를 지정하여
# from_documents() 함수에 보내면, 지정된 임베딩 함수를 통해 텍스트를 벡터로 변환하고, 이를 임시 db로 생성
# 그리고 similarity_search() 함수에 쿼리를 지정해주면 이를 바탕으로 벡터 유사도가 높은 벡터를 찾고 이를 자연어 형태로 출력

import tiktoken

# tiktoken 라이브러리를 사용하여 토크나이저를 가져옴
# cl100k_base: OpenAi에서 사용하는 토크나이저
tokenizer = tiktoken.get_encoding("cl100k_base")

# 이 tokenizer를 이용하여 문장을 토큰화하고, 토큰의 길이를 반환
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 문서를 로드하고 청크로 분할 (위에서 PyPDFLoader로 분할한 pages(리스트 형태) 사용)
text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=tiktoken_len 
)
# tiktoken_len을 기준으로 청크사이즈가 100 이하가 되도록 분할

docs = text_splitter2.split_documents(pages)

# 2. 문서를 임베딩 (허깅페이스 모델 - 로컬엠베딩 사용)
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs={'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs
)

# 3. 허깅페이스 모델을 사용하여 문서를 임베딩하고, Chroma에 저장
db = Chroma.from_documents(docs, hf)

# 4. 쿼리를 통해 가장 유사한 문서를 검색
query = "김태윤의 축구 포지션은?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# persist 함수를 통해 벡터 저장소를 로컬 저장소로 연결하고
# Chroma 객체를 선언할 때 로컬 저장소 경로를 지정하여 필요할때마다 로드할 수 있음

# 로컬에 저장
# db2 = Chroma.from_documents(docs, hf, persist_directory="./chroma_db")
# docs = db2.similarity_search(query)

# 로컬에서 불러오기
# db3 = Chroma(persist_directory="./chroma_db", embeddings_function=hf)
# docs = db3.similarity_search(query)
# print(docs[0].page_content)

# 유사도 점수 확인
# k=3: 가장 유사한 3개의 문서를 반환
docs = db.similarity_search_with_relevance_scores(query, k=3)
print("가장 유사한문서: \n {}\n".format(docs[0][0].page_content))
print("문서 유사도: \n {}".format(docs[0][1]))
print("\n\n2번째로 유사한문서 \n {}\n".format(docs[1][0].page_content))
print("문서 유사도: \n {}".format(docs[1][1]))

############### Retrievers ################
# Retriever: 검색을 잘 할수있게 만들어주는 모듈
# 분할된 텍스트(청크)를 컨텍스트에 주입을 하는 과정이 필요
# 어떤식으로 LLM에게 참고할 자료를 넘겨주는지에 따라 4개의 체인 종류가 존재 (Stuff, Map_reduce, refine, Map_rerank)
# 1. Stuff: 청크를 그대로 컨텍스트에 주입해서 LLM에 넘기는 방식
# 통째로 넘기기 때문에 토큰이슈 발생 가능

# 2. Map_reduce: 청크를 잘게 쪼개고, 최종 컨텍스트 압축본을 만들어 LLM에 넘기는 방식
# API를 여러번 요청하기 때문에 속도이슈 발생 가능

# 3. refine: 청크를 순회하면서 누적 답변을 생성
# 품질이 뛰어나지만 순회해야하므로 속도이슈 발생 가능

# 4. Map_rerank: 질문과 청크를 프롬프트에 넣고, LLM에 답변을 받는데, 답변에 대해 score도 받음
# 청크 각각에 구분된 답변을 얻고, score도 확인할수 있으나 속도이슈 발생 가능

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}),
    return_source_documents=True
)

query = "박지성의 축구 포지션을 알려줘"
result = qa(query)
print(result)

# res = llm.predict(prompt_template)
# print(res)
