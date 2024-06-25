import os
import sys
# 把父路径添加到检索目标中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from core.db import write_persist_var, read_persist_var, read_all_file_record_key, DocumentsHandle, TestDocumentsHandle  # 导入时将自动初始持久化变量
from rebuild.bge_reranker_large import BgeRerank
from rebuild.RedisChatMessageHistory import RedisChatMessageHistory
from config.config import ai_name, openai_api_key, model, project_name_1, openai_api_model_name, langsmith_project, langsmith_api_key, bad_answer, vector_original_path, small_model_loadon, asr_path, sound_color_path
######################################################
import re
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple, Dict
import shutil
import requests
import base64
import json
import numpy as np
import torch
######################################################
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from transformers import pipeline
# import ChatTTS
######################################################
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.jobstores.base import JobLookupError


# Langchain debug
# set_debug(True)

# LangSmith
if langsmith_project and langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project

# 定义辅助功能
######################################################
# 一个会过期的字典
class ExpiringDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_listener(self.job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.scheduler.start()

    def job_listener(self, event):
        try:
            self.scheduler.remove_job(event.job_id)
        except JobLookupError:
            pass

    def set(self, key, value, expire=read_persist_var("history_time")):
        self[key] = value
        self.scheduler.add_job(
            func=self.pop,
            trigger='date',
            run_date=datetime.now() + timedelta(seconds=expire),
            args=[key], 
            id=key
        )
    
    # 定义数据添加处理函数
    def value_add(self, key, value):
        try:
            self.scheduler.remove_job(key)
        except JobLookupError:
            pass
        self.set(key, value)
    
    # 定义任务及键值对清理
    def value_del(self, id):
        try:
            self.scheduler.remove_job(id)
        except JobLookupError:
            pass
        if id in self:
            del self[id]
        else:
            pass
# 特定用惰性列表去重
def dedup(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)
# 历史对话取回
def get_message_history(session_id: str) -> Dict:
    each_line = f"{RedisChatMessageHistory(session_id, url='redis://localhost:6379/0')}".strip().split("\n")
    his = []
    his_filt = []
    for line in each_line:
        his.append(line)
        if ': ' in line:
            speaker, speech = line.split(': ', 1)
            his_filt.append(speech.strip())
    return {"his": his, "his_filt": his_filt}


# 初始化对象
######################################################
# 向量数据库处理类
dh = DocumentsHandle()
tdh = TestDocumentsHandle()
# ASR & TTS 模型
transcriber = pipeline("automatic-speech-recognition", model=asr_path, device=small_model_loadon)
"""
chat = ChatTTS.Chat()
chat.load_models(device=small_model_loadon)
"""
# 向量数据库
if read_persist_var("first_time") == True:
    dh.documents_handle(f"/{project_name_1}/packages/database/qa.md")
    write_persist_var("first_time", False)
elif read_persist_var("first_time") == False:
    dh.load_local()
# 各项配置
system_prompt = read_persist_var('system_prompt')
temperature = read_persist_var("temperature")  # 生成文本的随机性，越接近 0 越会导致文本生成的非常确定重复
max_tokens = read_persist_var("max_tokens")  # 生成 token 数量的上限，更多的 token 可能代表更长的回复文本，但可能会降低精准且消耗更多的计算资源
mode_choose = [
    f"当前模式：{read_persist_var('chat_mode_choose')}", 
    f"历史分析：{read_persist_var('history_analyze')}", 
    f"询问逻辑优化：{read_persist_var('query_enhance')}", 
    f"询问关键词优化：{read_persist_var('query_enhance_1')}"
]
all_keywords = read_persist_var("细化关键词")
# 大语言模型接口
def llm(option: int):

    local = ChatOpenAI(
        api_key="EMPTY", 
        base_url="http://127.0.0.1:8000/v1", 
        temperature=temperature, 
        max_tokens=max_tokens, 
        model=openai_api_model_name
    )

    if option == 0:
        return local
    elif option == 1:
        local_stream = ChatOpenAI(
            api_key="EMPTY", 
            base_url="http://127.0.0.1:8000/v1", 
            temperature=temperature, 
            max_tokens=max_tokens, 
            model=openai_api_model_name, 
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        return local_stream
    elif option == 2:
        if openai_api_key:
            chatgpt = ChatOpenAI(
                api_key=openai_api_key, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                model=model
            )
            return chatgpt
    elif option == 3:
        if openai_api_key:
            chatgpt = ChatOpenAI(
                api_key=openai_api_key, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                model=model, 
                streaming=True
            )
            return chatgpt
        
    return local
# 初始化提示词模板
## 模板的修改一但不经测试可能会导致意想不到的结果，理解力越低的模型，影响越大（ 负面 ）
prompt_rewrite = ChatPromptTemplate.from_template("请重写询问，使其读起来围绕重点，并去除无关的信息。询问：{input}")
prompt_rewrite_1 = ChatPromptTemplate.from_template("目标：{input}\n候选：{chooses_1}\n请只回复与目标最相关的候选对象，如果都没可能相关或无法确定，请回复一个字无。")
prompt_rewrite_1_1 = ChatPromptTemplate.from_template("文本一：{input}\n文本二：{keyword}\n文本一中{simple_keyword}相关概念替换为文本二的概念后便是文本三，且文本三的语境等于文本一。\n请回复我文本三，不要在回复内容中加上 “文本三” 的字样，直接回复我文本三的内容。")
prompt_releated = ChatPromptTemplate.from_template("询问一：{ex_input}\n询问二：{input}\n请问询问二是否可能是询问一的延伸？可能请回答是，不可能请回答否，请只回答一个字。")
prompt_independent = ChatPromptTemplate.from_template("文本：{input}\n请问文本是否可能是一个独立的询问？大概率可能请回答是，大概率不可能请回答否，请只回答一个字。")
prompt_releated_qa = ChatPromptTemplate.from_template("数据：{context}\n询问：{input}\n请问数据是否可能与询问相关？可能请回答是，不可能请回答否，请只回答一个字。")
prompt_analyze = ChatPromptTemplate.from_template("历史对话：{input}\n"f"历史对话由一问一答组成，现在我要接替{ai_name}继续服务客户（ 由 AI 转为人类继续与客户对话 ）。请你向我提供一些对当前对话有用的建议。")
# 过期字典实例
doc_caches = ExpiringDict()
keyword_caches = ExpiringDict()
relate_or_not = ExpiringDict()


# Runnable
## 所有 Runnable 对象必须存在返回值
######################################################
# 定义历史对话关联检查
query_relate = prompt_releated | llm(2) | StrOutputParser()
# 定义询问独立性检查
query_independent = prompt_independent | llm(2) | StrOutputParser()
# 定义数据与询问关联检查
query_relate_qa = prompt_releated_qa | llm(2) | StrOutputParser()
# 定义询问逻辑优化
better_query = (prompt_rewrite | llm(2) | StrOutputParser())
# 定义询问关键词优化（ 选择 ）
@chain
def better_query_1(source: Dict):
    b_q_1 = (
        prompt_rewrite_1 | llm(2) | StrOutputParser()
    )
    found_time = 0
    final_choose = None
    for choose in source["chooses"]:
        if source["input"] in choose:
            found_time = found_time + 1
            final_choose = choose
    if found_time == 0:
        return "无"
    elif found_time == 1:
        return final_choose
    return b_q_1
# 定义询问关键词优化（ 融合 ）
better_query_1_1 = (prompt_rewrite_1_1 | llm(2) | StrOutputParser())
# 定义向量数据库文件上传
@chain
def upload_process(doc_path):
    new_path = f"{vector_original_path}/{doc_path[doc_path.rfind('/') + 1:]}"
    if new_path in read_all_file_record_key():
        dh.file_del(new_path)
    shutil.move(doc_path, vector_original_path)
    dh.documents_handle(new_path)
    return read_all_file_record_key()
# 定义向量数据库文件上传（ 数据库测试专用 ）
@chain
def test_upload_process(doc_path):
    if doc_path in list(tdh.file_record.keys()):
        tdh.file_del(doc_path)
    tdh.documents_handle(doc_path)
    return list(tdh.file_record.keys())
# 定义返回全部记录键（ 数据库测试专用 ）
@chain
def test_read_all_file_record_key(Any):
    return list(tdh.file_record.keys())
# 定义向量数据库单个文件删除
@chain
def del_file_process(doc_path):
    dh.file_del(doc_path)
    return read_all_file_record_key()
# 定义向量数据库单个文件删除（ 数据库测试专用 ）
@chain
def test_del_file_process(doc_path):
    tdh.file_del(doc_path)
    return list(tdh.file_record.keys())
# 定义向量数据库全部文件删除（ 数据库测试专用 ）
@chain
def test_del_all_file_process(Any):
    tdh.all_file_del()
    return list(tdh.file_record.keys())
# 向量数据库取回
@chain
def retrieve(source: Dict) -> List:
    choose = source.get("choose", "default")
    use_threshold = source.get("use_threshold", False)
    k = source.get("k", read_persist_var("reteriver_k"))
    s_t = source.get("score_threshold", read_persist_var("score_threshold"))
    ai_check = source.get("ai_check", False)

    if choose == "default":
        if use_threshold:
            retriever = dh.memory_vdb.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={"k": k, 'score_threshold': s_t}
            )
        else:
            retriever = dh.memory_vdb.as_retriever( 
                search_kwargs={"k": k}
            )
    elif choose == "test":
        if use_threshold:
            retriever = tdh.test_memory_vdb.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={"k": k, 'score_threshold': s_t}
            )
        else:
            retriever = tdh.test_memory_vdb.as_retriever( 
                search_kwargs={"k": k}
            )

    retriever_compress = ContextualCompressionRetriever(
        base_compressor=BgeRerank(), 
        base_retriever=retriever
    )

    docs = retriever_compress.invoke(source['retrive'])
    docs_1 = []
    if docs:
        for context in docs:
            docs_1.append(context.page_content)

    if ai_check:
        selected_docs = []
        for context in docs:
            response = query_relate_qa.invoke({"context": context.page_content, "input": source['retrive']})
            if "是" in response:
                selected_docs.append(context.page_content)

        return selected_docs

    return docs_1
# 语音转文本
@chain
def audio_text(source: Dict):

    sr, y = source["sampling_rate"], source["raw"]
    y = np.array(y)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))  # 防止失真

    if source.get("with_infer", False):
        text = transcriber({"sampling_rate": sr, "raw": y})["text"]
        return [text, simple_infer.invoke({"input": text, "temperature": 0})]
    else:
        return transcriber({"sampling_rate": sr, "raw": y})["text"]
# 文本转语音
"""
@chain
def text_audio(source: Dict):

    top_p = 0.7
    top_k = 20
    temperature = 0.3
    audio_seed_input = 42
    text_seed_input = 42

    torch.manual_seed(audio_seed_input)
    if sound_color_path:
        spk = torch.load(sound_color_path)
    else:
        spk = torch.randn(768)
    params_infer_code = {
        'spk_emb': spk, 
        'temperature': temperature,
        'top_P': top_p,
        'top_K': top_k,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

    torch.manual_seed(text_seed_input)
    text = chat.infer(source['text'], 
                        skip_refine_text=False,
                        refine_text_only=True,
                        params_refine_text=params_refine_text,
                        params_infer_code=params_infer_code
                        )
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    # text_data = text[0] if isinstance(text, list) else text

    return [sample_rate, [float(v) for v in audio_data]]
"""
# 定义对话通用接口
@chain
def infer(source: Dict):
    """针对此接口的调用，均需一次性请求 2 次：
    开启历史分析功能时，暂时无流式传输效果，
    具体调用方式请查看 ../sources/test_only.py 的 api_guid 变量。
    """
    
    # 初始化数据
    b_k_q = None
    if source["from"] == "customer":
        noh = True if "关闭" in mode_choose[1] else False
        ctype = 2 if "数据库对话（ 无限制 ）" in mode_choose[0] else 1 if "数据库对话" in mode_choose[0]  else 0
        b_q = False if "关闭" in mode_choose[2] else True
        b_q_1 = False if "关闭" in mode_choose[3] else True
    else:
        noh = source.get("noh", True)  # Fail Safe 默认 True
        ctype = source.get("ctype", 2)  # Fail Safe 默认 “数据库对话（ 无限制 ）”
        b_q = source.get("better_query", False)  # Fail Safe 默认 False
        b_q_1 = source.get("better_query_1", False)  # Fail Safe 默认 False

    # 处理输入
    def input_prc():
        if b_k_q == None:
            return better_query.invoke({"input": source["input"]}) if b_q else source["input"]
        return b_k_q

    if ctype != 0:
        # 第一次请求
        if source["time"] == 1:
            i_p = input_prc()  # 复用
            if source["id"] in keyword_caches:
                # 判断关键词选择
                k_c = better_query_1.invoke(
                    {
                        "input": i_p, 
                        "chooses": list(all_keywords[keyword_caches[source["id"]][0]].keys()), 
                        "chooses_1": "、".join(list(all_keywords[keyword_caches[source["id"]][0]].keys()))
                    }
                )
                # 融合关键词
                if "无" not in k_c:
                    b_k_q = better_query_1_1.invoke(
                        {
                            "input": keyword_caches[source["id"]][1], 
                            "keyword": f"{k_c}{'' if read_persist_var('细化关键词')[keyword_caches[source['id']][0]][k_c] == '无' else read_persist_var('细化关键词')[keyword_caches[source['id']][0]][k_c]}", 
                            "simple_keyword": keyword_caches[source["id"]][0]
                        }
                    )
            if b_q_1:
                found_time = 0
                found_time_1 = 0
                for simple_keyword in list(all_keywords.keys()):
                    if simple_keyword in i_p:
                        found_time = found_time + 1
                        keyword_caches.value_add(source["id"], [simple_keyword])
                if found_time == 1:
                    keywords = list(all_keywords[keyword_caches[source["id"]][0]].keys())
                    for keyword in keywords:
                        if keyword in i_p:
                            found_time_1 = found_time_1 + 1
                    if found_time_1 == 0:
                        keyword_caches[source["id"]].append(i_p)
                        return {"stop": f"系统检测您可能想询问 “{'、'.join(keywords)}” 中的一个，请回复对应内容，系统会自动优化您的询问（ 回复其它内容将放弃优化 ）。"}

            not_dbtest = True if "dbtest_" not in source["id"] else False  # 复用
            i_p = input_prc()  # 值更新

            if noh:
                if ctype == 1:
                    selected_docs = retrieve.invoke({"use_threshold": "yes", "retrive": i_p, "ai_check": "yes"}) if "dbtest_" not in source["id"] else retrieve.invoke({"choose": "test", "use_threshold": "yes", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                    if doc_caches[source["id"]]:
                        return {"input": i_p}
                    return {"stop": bad_answer}
                if ctype == 2:
                    selected_docs = retrieve.invoke({"retrive": i_p, "ai_check": "yes"}) if "dbtest_" not in source["id"] else retrieve.invoke({"choose": "test", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                    return {"input": i_p}

            # if not noh
            if f"{RedisChatMessageHistory(source['id'], url='redis://localhost:6379/0')}":
                ex_input = get_message_history(source["id"])["his_filt"][-2]
            else:
                ex_input = None

            if ctype == 1:
                # 无历史询问
                if not ex_input:
                    selected_docs = retrieve.invoke({"use_threshold": "yes", "retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "use_threshold": "yes", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                # 当前询问与临近历史询问不相关
                elif "否" in query_relate.invoke({"ex_input": ex_input, "input": i_p}):
                    selected_docs = retrieve.invoke({"use_threshold": "yes", "retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "use_threshold": "yes", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                # 当前询问与临近历史询问相关
                elif "是" in query_relate.invoke({"ex_input": ex_input, "input": i_p}):
                    # 记录相关
                    relate_or_not.value_add(source["id"], True)
                    # 非独立询问
                    if "否" in query_independent.invoke({"input": i_p}):
                        selected_docs = doc_caches[source["id"]]
                        # 对历史取回文本的删除倒计时进行延长
                        ## 防止用户一直追问，且倒计时结束，则删除数据后 AI 无法查看含有历史取回文本的提示词
                        doc_caches.value_add(source["id"], selected_docs)
                    # 独立询问
                    elif "是" in query_independent.invoke({"input": i_p}):
                        docs = retrieve.invoke({"use_threshold": "yes", "retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "use_threshold": "yes", "retrive": i_p, "ai_check": "yes"})
                        docs_1 = doc_caches[source["id"]]
                        docs_1.extend(docs)
                        selected_docs = list(dedup(docs_1, key=lambda doc: tuple(doc.metadata.items())))
                        # 保留合并后的前 10 个 QA 数据，避免出现无限套娃导致取回文本数据过多
                        ## 鉴于是追问的情况下执行此处代码，即使判断当前询问是独立的，10（ 默认配置 ）个数据也应该能做到问题解答了
                        selected_docs = selected_docs[:read_persist_var("reteriver_k_relate")]
                        doc_caches.value_add(source["id"], selected_docs)

                if doc_caches[source["id"]]:
                    return {"input": i_p}
                return {"stop": bad_answer}

            elif ctype == 2:
                if not ex_input:
                    selected_docs = retrieve.invoke({"retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                elif "否" in query_relate.invoke({"ex_input": ex_input, "input": input_prc}):
                    selected_docs = retrieve.invoke({"retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "retrive": i_p, "ai_check": "yes"})
                    doc_caches.value_add(source["id"], selected_docs)
                elif "是" in query_relate.invoke({"ex_input": ex_input, "input": input_prc}):
                    if "否" in query_independent.invoke({"input": input_prc}):
                        selected_docs = doc_caches[source["id"]]
                        doc_caches.value_add(source["id"], selected_docs)
                    elif "是" in query_independent.invoke({"input": input_prc}):
                        docs = retrieve.invoke({"retrive": i_p, "ai_check": "yes"}) if not_dbtest else retrieve.invoke({"choose": "test", "retrive": i_p, "ai_check": "yes"})
                        docs_1 = doc_caches[source["id"]]
                        docs_1.extend(docs)
                        selected_docs = list(dedup(docs_1, key=lambda doc: tuple(doc.metadata.items())))
                        selected_docs = selected_docs[:read_persist_var("reteriver_k_relate")]
                        doc_caches.value_add(source["id"], selected_docs)
                return {"input": i_p}
        
        # 第二次请求
        keyword_caches.value_del(source["id"])
        
        if doc_caches.get(source["id"], False):
            raw_docs = doc_caches[source["id"]]
            docs = ""
            for doc in raw_docs:
                docs += f"{doc}\n"
            docs = f"\n{docs}"

            prompt_d = ChatPromptTemplate.from_messages(
                [
                    ("system", f"{read_persist_var('system_prompt')}"), 
                    MessagesPlaceholder(variable_name="history"),
                    ("human", f"参考：{docs}\n询问：""{input}\n要求：若参考中可能存在询问想要的回复信息，则请结合参考回答的尽可能全面具体，不要编造，不允许省略信息。若不存在，则根据自身知识回答。")
                ]
            )
            prompt_d_repeat = ChatPromptTemplate.from_messages(
                [
                    ("system", f"{read_persist_var('system_prompt')}"), 
                    MessagesPlaceholder(variable_name="history"),
                    ("human", f"参考：{docs}\n询问：""{input}\n请在考虑历史询问的前提下，执行以下要求：若参考中可能存在询问想要的回复信息，则请结合参考回答的尽可能全面具体，不要编造，不允许省略信息。若不存在，则根据自身知识回答。")
                ]
            )
            prompt_d_noh = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt if source["from"] == "customer" else read_persist_var('system_prompt')), 
                    ("human", f"参考：{docs}\n询问：""{input}\n要求：若参考中可能存在询问想要的回复信息，则请结合参考回答的尽可能全面具体，不要编造，不允许省略信息。若不存在，则根据自身知识回答。")
                ]
            )

            def prompt_prc():
                if relate_or_not.get(source["id"], False):
                    prompt = prompt_d_repeat
                    relate_or_not.value_del(source["id"])
                else:
                    prompt = prompt_d
                
                return prompt

            d_a = RunnableWithMessageHistory(
                prompt_prc() | llm(source["llm"]), 
                RedisChatMessageHistory, 
                input_messages_key="input", 
                history_messages_key="history"
            ) | StrOutputParser()
            d_a_noh = prompt_d_noh | llm(source["llm"]) | StrOutputParser()

            return d_a_noh if noh else d_a.invoke({"input": source["input"]}, {"configurable": {"session_id": source["id"]}})  # 第二次请求不进行逻辑优化
    
    # ctype == 0
    # 第一次请求
    if source["time"] == 1:
        i_p = input_prc()  # 复用

        return {"input": i_p}
    
    # 第二次请求
    keyword_caches.value_del(source["id"])
    prompt_b = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt if source["from"] == "customer" else read_persist_var('system_prompt')), 
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ]
    )
    prompt_b_noh = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt if source["from"] == "customer" else read_persist_var('system_prompt')), 
            ("human", "{input}")
        ]
    )
    b_a = RunnableWithMessageHistory(
        prompt_b | llm(source["llm"]), 
        RedisChatMessageHistory, 
        input_messages_key="input", 
        history_messages_key="history"
    ) | StrOutputParser()
    b_a_noh = prompt_b_noh | llm(source["llm"]) | StrOutputParser()

    return b_a_noh if noh else b_a.invoke({"input": source["input"]}, {"configurable": {"session_id": source["id"]}})  # 第二次请求不进行询问逻辑优化
# 定义简易推理接口
@chain
def simple_infer(source: Dict):
    source["ai_check"] = "yes"
    source["retrive"] = source["input"]
    selected_docs = retrieve.invoke(source)

    llm = ChatOpenAI(
        api_key="EMPTY", 
        base_url="http://127.0.0.1:8000/v1", 
        temperature=source["temperature"], 
        max_tokens=max_tokens, 
        model=openai_api_model_name
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), 
            ("human", f"参考：{selected_docs}\n询问：""{input}\n要求：若参考中可能存在询问想要的回复信息，则请结合参考回答的尽可能全面具体，不要编造，不允许省略信息。若不存在，则根据自身知识回答。")
        ]
    )
    prompt_1 = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), 
            ("human", "{input}")
        ]
    )

    if selected_docs:
        answer_chian = prompt | llm | StrOutputParser()
    else:
        answer_chian = prompt_1 | llm | StrOutputParser()

    return answer_chian.invoke({"input": source["input"]})
# 定义数据库对话中的参考资料表现形式
@chain
def plus_links(source: Dict) -> str:
    if doc_caches.get(source["id"], False):
        pattern = r"https://\S+"
        https = re.findall(pattern, f"{doc_caches[source['id']]}")
        dup_https = list(dict.fromkeys(https))  # 去重
        if len(https) > 1:
            https_links = ', '.join(dup_https)
            matches = f"可能有用的资料：{https_links}"
            plus = f"\n\n{matches}"
            return plus
        elif len(https) == 1:
            https_links = https[0]
            matches = f"可能有用的资料：{https_links}"
            plus = f"\n\n{matches}"
            return plus
        return ""
    return ""
# 定义对话分析
@chain
def session_analyze(source: Dict):
    if source["from"] == "customer":
        noh = True if "关闭" in mode_choose[1] else False
    else:
        noh = source.get("noh", True)  # Fail Safe 默认 True
    
    analyze_answer = prompt_analyze | llm(0) | StrOutputParser()

    if noh:
        return analyze_answer.invoke({"input": source["input"]})
    elif not noh:
        his = f"{RedisChatMessageHistory(source['id'], url='redis://localhost:6379/0')}".strip().split("\n")
        if len(his) >= 6:
            return analyze_answer.invoke({"input": f"{his[-6]} {his[-5]} {his[-4]} {his[-3]} {his[-2]} {his[-1]}".replace("Human", "客户").replace("AI", f"{ai_name}")})
        elif len(his) == 4:
            return analyze_answer.invoke({"input": f"{his[-4]} {his[-3]} {his[-2]} {his[-1]}".replace("Human", "客户").replace("AI", f"{ai_name}")})
        elif len(his) == 2:
            return analyze_answer.invoke({"input": f"{his[-2]} {his[-1]}".replace("Human", "客户").replace("AI", f"{ai_name}")})
# 定义历史记录清除
@chain
def clean_history(source: Dict) -> str:
    # 删除内存中的历史对话缓存
    RedisChatMessageHistory(source["id"]).clear()
    # 删除内存中最近一次的数据库取回文本
    doc_caches.value_del(source["id"])
    # 删除内存中关键词优化暂存记录
    keyword_caches.value_del(source["id"])
    # 删除内存中询问关联性暂存记录
    relate_or_not.value_del(source["id"])
    try:
        if source['challenge']:
            requests.post('http://localhost:2031/del_challenge', json={'challenge': source['challenge']})
    except KeyError:
        pass
    return "Done"


