import os
import sys
# 把父路径添加到检索目标中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
######################################################
from config.config import ai_name, weight_size, project_name, project_name_1, default_challenge, concurrency_limit, config_path, vector_original_path, rsa_pub, init_system_prompt
from core.db import write_persist_var, read_persist_var, read_all_file_record_key
from packages.sources.text_only import api_guid
######################################################
import time
from typing import Any, List, Optional, Tuple, Dict
import shelve
from threading import Lock
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import requests
import numpy as np
######################################################
from langserve import RemoteRunnable
######################################################
import gradio as gr


# Langchain debug
# set_debug(True)


# API 接入
def encrypt_challenge(challenge: str) -> str:
    cipher = PKCS1_OAEP.new(RSA.import_key(open(rsa_pub, "rb").read()))
    encrypt = cipher.encrypt(challenge.encode())
    return base64.b64encode(encrypt).decode('utf-8')
password = encrypt_challenge(default_challenge)
better_query = RemoteRunnable(url="http://localhost:2031/better_query/", headers={"P": password})
upload_process = RemoteRunnable(url="http://localhost:2031/upload_process/", headers={"P": password})
test_upload_process = RemoteRunnable(url="http://localhost:2031/test_upload_process/", headers={"P": password})
test_read_all_file_record_key = RemoteRunnable(url="http://localhost:2031/test_read_all_file_record_key/", headers={"P": password})
del_file_process = RemoteRunnable(url="http://localhost:2031/del_file_process/", headers={"P": password})
test_del_file_process = RemoteRunnable(url="http://localhost:2031/test_del_file_process/", headers={"P": password})
test_del_all_file_process = RemoteRunnable(url="http://localhost:2031/test_del_all_file_process/", headers={"P": password})
retrieve = RemoteRunnable(url="http://localhost:2031/retrieve/", headers={"P": password})
audio_text = RemoteRunnable(url="http://localhost:2031/audio_text/", headers={"P": password})
audio_generator = RemoteRunnable(url="http://localhost:2031/audio_generator/", headers={"P": password})
infer = RemoteRunnable(url="http://localhost:2031/infer/", headers={"P": password})
simple_infer = RemoteRunnable(url="http://localhost:2031/simple_infer/", headers={"P": password})
plus_links = RemoteRunnable(url="http://localhost:2031/plus_links/", headers={"P": password})
session_analyze = RemoteRunnable(url="http://localhost:2031/session_analyze/", headers={"P": password})
clean_history = RemoteRunnable(url="http://localhost:2031/clean_history/", headers={"P": password})


# 初始化变量
######################################################
# 向量数据库锁
manually_lock = 0
lock = Lock()
lock_1 = Lock()
# 是否正在 Streaming
streaming = {}
# 对话模式
chatmode = ["基础对话", "数据库对话", "数据库对话（ 无限制 ）"]
# 网页标志
sign_path = f"/{project_name_1}/packages/sources/sign.png"
# 存储对话
messages = {}


# 定义功能函数
######################################################
# 获取 messages id 列表
def get_ids():
    if not messages:
        m_ids = "无"
        return m_ids
    else:
        m_ids = list(messages.keys())
        return m_ids
# 切换配置引发的对话历史删除
def dected_del(id_input):

    if not id_input:
        return [], [], ""
    else:
        clean_history.invoke({"id": id_input})
        if id_input in messages:
            del messages[id_input]
        else:
            pass
        clean_history.invoke({"id": f"dbtest_{id_input}"})
        if f"dbtest_{id_input}" in messages:
            del messages[f"dbtest_{id_input}"]
        else:
            pass

    return [], [], ""
# 接入 ChatGPT 提示
def chatgpt_info():
    gr.Info("如需 ChatGPT 接入，请修改 config.py 设置文件")
# 数据库测试
def test_db(id_input, db_test):

    if db_test == "开启":
        gr.Warning("音频询问暂不支持 Test 数据库")
        c2 = gr.update(value=[], visible=True)
    elif db_test == "关闭":
        c2 = gr.update(value=[], visible=False)

    if not id_input:
        return [], c2, ""
    else:
        clean_history.invoke({"id": id_input})
        if id_input in messages:
            del messages[id_input]
        else:
            pass
        clean_history.invoke({"id": f"dbtest_{id_input}"})
        if f"dbtest_{id_input}" in messages:
            del messages[f"dbtest_{id_input}"]
        else:
            pass
        return [], c2, ""
# 从 API 加载配置
def load_from_api(id_input, chat_mode_choose, history_analyze, db_test, query, chatbot):
    if read_persist_var("chat_mode_choose") == chat_mode_choose:
        if read_persist_var("history_analyze") == history_analyze:
            if read_persist_var("db_test") == db_test:
                if read_persist_var("db_test") == "开启":
                    c2 = gr.update(visible=True)
                elif read_persist_var("db_test") == "关闭":
                    c2 = gr.update(visible=False)
                c = gr.update(value=read_persist_var("chat_mode_choose"))
                h = gr.update(value=read_persist_var("history_analyze"))
                q = gr.update(value=read_persist_var("query_enhance"))
                q_1 = gr.update(value=read_persist_var("query_enhance_1"))
                d = gr.update(value=read_persist_var("db_test"))
                return c, h, q, q_1, d, query, chatbot, c2
            else:
                if read_persist_var("db_test") == "开启":
                    c2 = gr.update(value=[], visible=True)
                elif read_persist_var("db_test") == "关闭":
                    c2 = gr.update(value=[], visible=False)
                c = gr.update(value=read_persist_var("chat_mode_choose"))
                h = gr.update(value=read_persist_var("history_analyze"))
                q = gr.update(value=read_persist_var("query_enhance"))
                q_1 = gr.update(value=read_persist_var("query_enhance_1"))
                d = gr.update(value=read_persist_var("db_test"))
                if not id_input:
                    return c, h, q, q_1, d, "", [], c2
                else:
                    clean_history.invoke({"id": id_input})
                    if id_input in messages:
                        del messages[id_input]
                    else:
                        pass
                    clean_history.invoke({"id": f"dbtest_{id_input}"})
                    if f"dbtest_{id_input}" in messages:
                        del messages[f"dbtest_{id_input}"]
                    else:
                        pass
                    return c, h, q, q_1, d, "", [], c2
        else:
            if read_persist_var("db_test") == "开启":
                c2 = gr.update(value=[], visible=True)
            elif read_persist_var("db_test") == "关闭":
                c2 = gr.update(value=[], visible=False)
            c = gr.update(value=read_persist_var("chat_mode_choose"))
            h = gr.update(value=read_persist_var("history_analyze"))
            q = gr.update(value=read_persist_var("query_enhance"))
            q_1 = gr.update(value=read_persist_var("query_enhance_1"))
            d = gr.update(value=read_persist_var("db_test"))
            if not id_input:
                return c, h, q, q_1, d, "", [], c2
            else:
                clean_history.invoke({"id": id_input})
                if id_input in messages:
                    del messages[id_input]
                else:
                    pass
                clean_history.invoke({"id": f"dbtest_{id_input}"})
                if f"dbtest_{id_input}" in messages:
                    del messages[f"dbtest_{id_input}"]
                else:
                    pass
                return c, h, q, q_1, d, "", [], c2
    else:
        if read_persist_var("db_test") == "开启":
            c2 = gr.update(value=[], visible=True)
        elif read_persist_var("db_test") == "关闭":
            c2 = gr.update(value=[], visible=False)
        c = gr.update(value=read_persist_var("chat_mode_choose"))
        h = gr.update(value=read_persist_var("history_analyze"))
        q = gr.update(value=read_persist_var("query_enhance"))
        q_1 = gr.update(value=read_persist_var("query_enhance_1"))
        d = gr.update(value=read_persist_var("db_test"))
        if not id_input:
            return c, h, q, q_1, d, "", [], c2
        else:
            clean_history.invoke({"id": id_input})
            if id_input in messages:
                del messages[id_input]
            else:
                pass
            clean_history.invoke({"id": f"dbtest_{id_input}"})
            if f"dbtest_{id_input}" in messages:
                del messages[f"dbtest_{id_input}"]
            else:
                pass
            return c, h, q, q_1, d, "", [], c2
def load_from_api_1():
    s = gr.update(value=read_persist_var("system_prompt"))
    c = gr.update(value=read_persist_var("chat_mode_choose"))
    h = gr.update(value=read_persist_var("history_analyze"))
    q = gr.update(value=read_persist_var("query_enhance"))
    q_1 = gr.update(value=read_persist_var("query_enhance_1"))
    d = gr.update(value=read_persist_var("db_test"))
    h_l = gr.update(value=read_persist_var("history_len"))
    h_t = gr.update(value=read_persist_var("history_time"))
    m = gr.update(value=read_persist_var("max_tokens"))
    t = gr.update(value=read_persist_var("temperature"))
    r = gr.update(value=read_persist_var("reteriver_k"))
    r_f = gr.update(value=read_persist_var("reteriver_k_final"))
    r_r = gr.update(value=read_persist_var("reteriver_k_relate"))
    s_t = gr.update(value=read_persist_var("score_threshold"))
    return s, c, h, q, q_1, d, h_l, h_t, m, t, r, r_f, r_r, s_t
# 写入配置
def save_setting(chat_mode_choose, history_analyze, query_enhance, query_enhance_1, db_test):
    write_persist_var("chat_mode_choose", chat_mode_choose)
    write_persist_var("history_analyze", history_analyze)
    write_persist_var("query_enhance", query_enhance)
    write_persist_var("query_enhance_1", query_enhance_1)
    write_persist_var("db_test", db_test)
    gr.Info("已保存至 API")
def save_setting_1(system_command, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, db_test, history_len, history_time, max_tokens, temperature, reteriver_k, reteriver_k_final, reteriver_k_relate, score_threshold):
    write_persist_var("system_prompt", system_command)
    write_persist_var("chat_mode_choose", chat_mode_choose)
    write_persist_var("history_analyze", history_analyze)
    write_persist_var("query_enhance", query_enhance)
    write_persist_var("query_enhance_1", query_enhance_1)
    write_persist_var("db_test", db_test)
    write_persist_var("history_len", history_len)
    write_persist_var("history_time", history_time)
    write_persist_var("max_tokens", max_tokens)
    write_persist_var("temperature", temperature)
    write_persist_var("reteriver_k", reteriver_k)
    write_persist_var("reteriver_k_final", reteriver_k_final)
    write_persist_var("reteriver_k_relate", reteriver_k_relate)
    write_persist_var("score_threshold", score_threshold)
    gr.Info("已保存至 API（ 重启后生效 ）")
# 检索向量数据库
def reterive_gr(id_input, reteriver_text, score_threshold_now, reterive_with_ai, db_choose):
    
    default_doc_value = "暂无，请尝试检索或修改配置后检索。"
    doc, doc_1, doc_2, doc_3, doc_4, doc_5 = default_doc_value, default_doc_value, default_doc_value, default_doc_value, default_doc_value, default_doc_value

    if not id_input:
        gr.Warning("请写入临时对话 ID")
        return (
            gr.update(open=False), 
            gr.update(value=doc), 
            gr.update(open=False), 
            gr.update(value=doc_1), 
            gr.update(open=False), 
            gr.update(value=doc_2), 
            gr.update(open=False), 
            gr.update(value=doc_3), 
            gr.update(open=False), 
            gr.update(value=doc_4), 
        )
    
    if not reteriver_text and not messages.get(id_input, False): 
        gr.Warning("未提供用于查询的相关字段")
        return (
            gr.update(open=False), 
            gr.update(value=doc), 
            gr.update(open=False), 
            gr.update(value=doc_1), 
            gr.update(open=False), 
            gr.update(value=doc_2), 
            gr.update(open=False), 
            gr.update(value=doc_3), 
            gr.update(open=False), 
            gr.update(value=doc_4), 
        )
    else:
        if db_choose == "Default":
            if score_threshold_now:
                if reterive_with_ai == "开启":
                    docs = retrieve.invoke({"use_threshold": "yes", "score_threshold": score_threshold_now, "ai_check": "yes", "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
                else:
                    docs = retrieve.invoke({"use_threshold": "yes", "score_threshold": score_threshold_now, "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
            else:
                if reterive_with_ai == "开启":
                    docs = retrieve.invoke({"ai_check": "yes", "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
                else:
                    docs = retrieve.invoke({"retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
        else:
            if not test_read_all_file_record_key.invoke(""):
                gr.Info("测试数据库缺少数据")
                return (
                    gr.update(open=False), 
                    gr.update(value=doc), 
                    gr.update(open=False), 
                    gr.update(value=doc_1), 
                    gr.update(open=False), 
                    gr.update(value=doc_2), 
                    gr.update(open=False), 
                    gr.update(value=doc_3), 
                    gr.update(open=False), 
                    gr.update(value=doc_4), 
                )
            else:
                if score_threshold_now:
                    if reterive_with_ai == "开启":
                        docs = retrieve.invoke({"choose": "test", "use_threshold": "yes", "score_threshold": score_threshold_now, "ai_check": "yes", "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
                    else:
                        docs = retrieve.invoke({"choose": "test", "use_threshold": "yes", "score_threshold": score_threshold_now, "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
                else:
                    if reterive_with_ai == "开启":
                        docs = retrieve.invoke({"choose": "test", "ai_check": "yes", "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
                    else:
                        docs = retrieve.invoke({"choose": "test", "retrive": reteriver_text if reteriver_text else messages[id_input][-1][0]})
        
        for idx, doc_value in enumerate(docs):
            if idx == 0:
                doc = doc_value
            elif idx == 1:
                doc_1 = doc_value
            elif idx == 2:
                doc_2 = doc_value
            elif idx == 3:
                doc_3 = doc_value
            elif idx == 4:
                doc_4 = doc_value
            elif idx == 5:
                doc_5 = doc_value
        
        return (
            gr.update(open=True if len(docs) >=1 else False), 
            gr.update(value=doc), 
            gr.update(open=True if len(docs) >=2 else False), 
            gr.update(value=doc_1), 
            gr.update(open=True if len(docs) >=3 else False), 
            gr.update(value=doc_2), 
            gr.update(open=True if len(docs) >=4 else False), 
            gr.update(value=doc_3), 
            gr.update(open=True if len(docs) >=5 else False), 
            gr.update(value=doc_4)
        )
# 简易推理
def simple_infer_gr(id_input, reply_for_you_prompt, temperature_now):

    if not id_input:
        gr.Warning("请写入临时对话 ID")
        return gr.update(value="暂无，请点击 “生成回复”")
    
    if not reply_for_you_prompt and not messages.get(id_input, False): 
        gr.Warning("未提供用于生成回复的相关字段")
        return gr.update(value="暂无，请点击 “生成回复”")
    else:
        reply = simple_infer.invoke({"input": reply_for_you_prompt if reply_for_you_prompt else messages[id_input][-1][0], "temperature": temperature_now})
        return gr.update(value=reply)
# 对话分析
def session_analyze_gr(id_input, history_analyze, analyze_show):

    noh = True if history_analyze == "关闭" else False

    if streaming.get(id_input, False):
        return gr.update(value=analyze_show)
    
    if not id_input:
        gr.Warning("请写入临时对话 ID")
        return gr.update(value=analyze_show)
    
    if not messages.get(id_input, False): 
        gr.Warning("没有可用历史对话信息")
        return gr.update(value=analyze_show)
    else:
        if len(messages[id_input]) >= 3:
            analyze_answer = session_analyze_gr.invoke(
                {
                    "noh": noh, 
                    "input": f"客户：{messages[id_input][-3][0]} {ai_name}：{messages[id_input][-3][1]} 客户：{messages[id_input][-2][0]} {ai_name}：{messages[id_input][-2][1]} 客户：{messages[id_input][-1][0]} {ai_name}：{messages[id_input][-1][1]}", 
                    "id": id_input, 
                    "from": "webui"
                }
            )
            return gr.update(value=analyze_answer)
        elif len(messages[id_input]) == 2:
            analyze_answer = session_analyze_gr.invoke(
                {
                    "noh": noh, 
                    "input": f"客户：{messages[id_input][-2][0]} {ai_name}：{messages[id_input][-2][1]} 客户：{messages[id_input][-1][0]} {ai_name}：{messages[id_input][-1][1]}", 
                    "id": id_input, 
                    "from": "webui"
                }
            )
            return gr.update(value=analyze_answer)
        elif len(messages[id_input]) == 1:
            analyze_answer = session_analyze_gr.invoke(
                {
                    "noh": noh, 
                    "input": f"客户：{messages[id_input][-1][0]} {ai_name}：{messages[id_input][-1][1]}", 
                    "id": id_input, 
                    "from": "webui"
                }
            )
            return gr.update(value=analyze_answer)
# 重制配置
def reset_setting():
    write_persist_var("system_prompt", init_system_prompt)
    write_persist_var("chat_mode_choose", "数据库对话（ 无限制 ）")
    write_persist_var("history_analyze", "关闭")
    write_persist_var("query_enhance", "关闭")
    write_persist_var("query_enhance_1", "关闭")
    write_persist_var("db_test", "关闭")
    write_persist_var("history_len", 3)
    write_persist_var("history_time", 7200)
    write_persist_var("max_tokens", 800)
    write_persist_var("temperature", 0)
    write_persist_var("reteriver_k", 20)
    write_persist_var("reteriver_k_final", 5)
    write_persist_var("reteriver_k_relate", 10)
    write_persist_var("score_threshold", 0.3)
    s = gr.update(value=read_persist_var("system_prompt"))
    c = gr.update(value=read_persist_var("chat_mode_choose"))
    h = gr.update(value=read_persist_var("history_analyze"))
    q = gr.update(value=read_persist_var("query_enhance"))
    q_1 = gr.update(value=read_persist_var("query_enhance_1"))
    d = gr.update(value=read_persist_var("db_test"))
    h_l = gr.update(value=read_persist_var("history_len"))
    h_t = gr.update(value=read_persist_var("history_time"))
    m = gr.update(value=read_persist_var("max_tokens"))
    t = gr.update(value=read_persist_var("temperature"))
    r = gr.update(value=read_persist_var("reteriver_k"))
    r_f = gr.update(value=read_persist_var("reteriver_k_final"))
    r_r = gr.update(value=read_persist_var("reteriver_k_relate"))
    s_t = gr.update(value=read_persist_var("score_threshold"))
    gr.Info("API 设置已重制（ 重启后生效 ）")
    return s, c, h, q, q_1, d, h_l, h_t, m, t, r, r_f, r_r, s_t
# 清除历史对话
def clear_session(id_input):
    if not id_input:
        return [], [], ""
    else:
        clean_history.invoke({"id": id_input})
        if id_input in messages:
            del messages[id_input]
        else:
            pass
        clean_history.invoke({"id": f"dbtest_{id_input}"})
        if f"dbtest_{id_input}" in messages:
            del messages[f"dbtest_{id_input}"]
        else:
            pass
        return [], [], ""
# 音频转文本处理（ 可间接推理 ）
def audio_text_gr(id_input, chatbot, audio_query, query, audio_check):
    if not id_input:
        gr.Warning("请写入临时对话 ID")
        return chatbot, query
    else:
        sr, y = audio_query
        y_list = [int(i) for i in y]
        if audio_check:
            text = audio_text.invoke({"sampling_rate": sr, "raw": y_list})
            return chatbot, text
        else:
            answer = audio_text.invoke({"with_infer": "yes", "sampling_rate": sr, "raw": y_list})
            
            if id_input in messages:
                messages[id_input].append(answer)
            else:
                messages[id_input] = [answer]
            
            return messages[id_input], query
# 常规推理（ 流式传输 ）
def model_chat(id_input, db_test, chatbot, chatbot_1, query):

    if not id_input:
        gr.Warning("请写入临时对话 ID")
        return chatbot, chatbot_1, gr.update(interactive=True), query, gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
    elif "dbtest_" in id_input:
        gr.Warning("“dbtest_” 是保留字段，请删除")
        return chatbot, chatbot_1, gr.update(interactive=True), query, gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
    
    if not query:
        return chatbot, chatbot_1, gr.update(interactive=True), query, gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)
    
    streaming[id_input] = True

    if id_input in messages:
        messages[id_input].append([query, f"{ai_name}正在思考..."])
    else:
        messages[id_input] = [[query, f"{ai_name}正在思考..."]]
    
    if db_test == "开启":
        if manually_lock != 1:
            if test_read_all_file_record_key.invoke(""):
                streaming[f"dbtest_{id_input}"] = True
                if f"dbtest_{id_input}" in messages:
                    messages[f"dbtest_{id_input}"].append([query, f"{ai_name}正在思考..."])
                else:
                    messages[f"dbtest_{id_input}"] = [[query, f"{ai_name}正在思考..."]]
                return messages[id_input], messages[f"dbtest_{id_input}"], gr.update(interactive=False), gr.update(value="", visible=False), gr.update(visible=True), gr.update(interactive=False), gr.update(interactive=False)
            gr.Info("测试数据库缺少数据")
        gr.Warning("测试数据库正在被修改，请稍后重试")
    return messages[id_input], [], gr.update(interactive=False), gr.update(value="", visible=False), gr.update(visible=True), gr.update(interactive=False), gr.update(interactive=False)
# ---
def model_chat_1(id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot):

    if not streaming.get(id_input, False):
        return chatbot

    # 初始化数据
    noh = True if history_analyze == "关闭" else False
    ctype = 2 if chat_mode_choose == chatmode[2] else 1 if chat_mode_choose == chatmode[1] else 0
    better_query = False if query_enhance == "关闭" else True
    better_query_1 = False if query_enhance_1 == "关闭" else True
    
    pre_work = infer.invoke(
        {
            "time": 1, 
            "noh": noh, 
            "ctype": ctype, 
            "input": messages[id_input][-1][0], 
            "id": id_input, 
            "better_query": better_query, 
            "better_query_1": better_query_1, 
            "from": "webui", 
            "llm": 0
        }
    )
    if not pre_work.get("stop", False):
        answer = infer.stream(
            {
                "time": 2, 
                "noh": noh, 
                "ctype": ctype, 
                "input": pre_work["input"], 
                "id": id_input, 
                "from": "webui", 
                "llm": 0
            }
        )
        messages[id_input][-1][1] = ""
        for character in answer:
            messages[id_input][-1][1] += character
            time.sleep(0.03)
            yield messages[id_input]
        messages[id_input][-1][1] = f"{messages[id_input][-1][1]}{plus_links.invoke({'id': id_input})}"
        yield messages[id_input]
    else:
        messages[id_input][-1][1] = pre_work["stop"]
        yield messages[id_input]
# ---
def model_chat_1_1(id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot):

    if not streaming.get(f"dbtest_{id_input}", False):
        return chatbot

    if chat_mode_choose == chatmode[0]:
        return chatbot
    
    # 初始化数据
    noh = True if history_analyze == "关闭" else False
    ctype = 2 if chat_mode_choose == chatmode[2] else 1 if chat_mode_choose == chatmode[1] else 0
    better_query = False if query_enhance == "关闭" else True
    better_query_1 = False if query_enhance_1 == "关闭" else True
    
    pre_work = infer.invoke(
        {
            "time": 1, 
            "noh": noh, 
            "ctype": ctype, 
            "input": messages[id_input][-1][0], 
            "id": f"dbtest_{id_input}", 
            "better_query": better_query, 
            "better_query_1": better_query_1, 
            "from": "webui", 
            "llm": 0
        }
    )
    if not pre_work.get("stop", False):
        answer = infer.stream(
            {
                "time": 2, 
                "noh": noh, 
                "ctype": ctype, 
                "input": pre_work["input"], 
                "id": f"dbtest_{id_input}", 
                "from": "webui", 
                "llm": 0
            }
        )
        messages[id_input][-1][1] = ""
        for character in answer:
            messages[id_input][-1][1] += character
            time.sleep(0.03)
            yield messages[id_input]
        messages[id_input][-1][1] = f"{messages[id_input][-1][1]}{plus_links.invoke({'id': id_input})}"
        yield messages[id_input]
    else:
        messages[id_input][-1][1] = pre_work["stop"]
        yield messages[id_input]
# ---
def model_chat_2(id_input, query_1):
    # audio = gr.update(value=None)
    if streaming.get(id_input, False):
        # audio = audio_generator.invoke({"text": messages[id_input][-1][0]})
        del streaming[id_input]
    if streaming.get(f"dbtest_{id_input}", False):
        del streaming[f"dbtest_{id_input}"]
    return (
        # (audio[0], np.array(audio[1])), 
        gr.update(interactive=True), 
        gr.update(value=query_1, visible=True), 
        gr.update(value="", visible=False), 
        gr.update(interactive=True), 
        gr.update(interactive=True)
    )
# 查看 ID 对应对话
def history_watch(history_choose):
    if history_choose in list(messages.keys()):
        return messages[history_choose]
    else:
        return []
# 对话刷新功能（ 临时对话管理 ）
def refresh_watch(history_choose):
    new_history_chooses = gr.update(choices=get_ids())
    if id_input in messages:
        return new_history_chooses, messages[history_choose]
    else:
        return new_history_chooses, []
# Tab 刷新功能（ 临时对话管理 ）
def refresh_tab_watch():
    new_history_chooses = gr.update(choices=get_ids(), value=None)
    return new_history_chooses, []
# 历史对话删除（ 临时对话管理 ）
def del_history(history_choose, chatbot_2):

    if streaming.get(history_choose, False):
        gr.Warning("选中对话生在生成回复，请稍后重试")
        return history_choose, chatbot_2
    
    new_history_chooses = gr.update(choices=get_ids(), value=None)

    if history_choose in list(messages.keys()):
        clean_history.invoke({"id": history_choose})
        del messages[history_choose]
        return new_history_chooses, []
    else:
        return new_history_chooses, []
# 向量数据库刷新功能
def refresh_db_watch():
    with lock:
        with lock_1:
            return gr.update(value=read_all_file_record_key()), gr.update(value=test_read_all_file_record_key.invoke("")), gr.update(choices=list(read_persist_var("细化关键词").keys()), value=None), gr.update(choices=[], value=None)
# 文件上传
def upload_process_gr(file_upload_path):
    with lock:
        gr.Info("新的修改将在重启服务后生效")
        return gr.update(value=upload_process.invoke(file_upload_path))
# 文件上传（ 测试数据库专用 )
def test_upload_process_gr(test_file_upload_path):
    global manually_lock
    with lock:
        if any(value for key, value in streaming.items() if key.startswith("dbtest_")):
            gr.Warning("测试数据库正在被读取，请稍后重试")
            return test_read_all_file_record_key.invoke("")
        manually_lock = 1
        t_u_p = gr.update(value=test_upload_process.invoke(test_file_upload_path))
        manually_lock = 0
        return t_u_p
# 全部文件删除（ 测试数据库专用 ）
def test_del_all_file_process_gr():
    global manually_lock
    with lock:
        if any(value for key, value in streaming.items() if key.startswith("dbtest_")):
            gr.Warning("测试数据库正在被读取，请稍后重试")
            return test_read_all_file_record_key.invoke("")
        manually_lock = 1
        t_d_a_f_p = gr.update(value=test_del_all_file_process.invoke(""))
        manually_lock = 0
        return t_d_a_f_p
# 文件删除
def del_file_process_gr(del_on, del_file):
    with lock:
        if del_on == "Default":
            doc_path = f"{vector_original_path}/{del_file}.md"
            # 校验文件存在与否，防止删除本已不存在的文件后报错
            if doc_path in read_all_file_record_key():
                if len(read_all_file_record_key()) == 1:
                    gr.Info("请确保数据库内至少存在一个文件")
                    return del_file, gr.update(value=read_all_file_record_key()), gr.update(value=test_read_all_file_record_key.invoke(""))
                else:
                    gr.Info("新的修改将在重启服务后生效")
                    return "", gr.update(value=del_file_process.invoke(doc_path)), gr.update(value=test_read_all_file_record_key.invoke(""))
            else:
                gr.Info("无匹配文件（ 可能已被删除 ）")
                return del_file, gr.update(value=read_all_file_record_key()), gr.update(value=test_read_all_file_record_key.invoke(""))
        elif del_on == "Test":
            global manually_lock
            # 校验文件存在与否，防止删除本已不存在的文件后报错
            found_path = False
            for doc_path in test_read_all_file_record_key.invoke(""):
                if doc_path.endswith(del_file):
                    found_path = doc_path
                    break
            if found_path:
                if found_path in test_read_all_file_record_key.invoke(""):
                    if len(test_read_all_file_record_key.invoke("")) == 1:
                        gr.Info("请确保数据库内至少存在一个文件")
                        return del_file, gr.update(value=read_all_file_record_key()), gr.update(value=test_read_all_file_record_key.invoke(""))
                    else:
                        if manually_lock != 2:
                            manually_lock = 1
                            t_d_f_p = gr.update(value=test_del_file_process.invoke(found_path))
                            manually_lock = 0
                            return "", gr.update(value=read_all_file_record_key()), t_d_f_p
                        else:
                            gr.Warning("测试数据库正在被读取，请稍后重试")
                            return "", gr.update(value=read_all_file_record_key()), test_read_all_file_record_key.invoke("")
            else:
                gr.Info("无匹配文件（ 可能已被删除 ）")
                return del_file, gr.update(value=read_all_file_record_key()), gr.update(value=test_read_all_file_record_key.invoke(""))
# 查看对应子关键词
def keyword_list(simple_keyword_choose):
    with lock_1:
        return gr.update(choices=list(read_persist_var("细化关键词")[simple_keyword_choose].keys()), value=None)
# 提取关键词说明
def load_tip(simple_keyword_choose, keyword_choose):
    with lock_1:
        value = read_persist_var("细化关键词")[simple_keyword_choose][keyword_choose].replace("（", "").replace("）", "")
        if value:
            return gr.update(placeholder=f"一段精简的文字用于描述子关键词。\n当前：{value}")
        else:
            return gr.update(placeholder=f"一段精简的文字用于描述子关键词。\n当前：无")
# 将关键词以固定格式加载至文本框（ 方便复写新关键词 ）
def keywords_load(simple_keyword_choose):
    with lock_1:
        return gr.update(value=f"{simple_keyword_choose} {' '.join(key for key in read_persist_var('细化关键词')[simple_keyword_choose].keys())}")
# 定义删除父关键词
def del_simple_keyword(simple_keyword_choose):
    with lock_1:
        if simple_keyword_choose in read_persist_var("细化关键词"):
            with shelve.open(f"{config_path}/persist_var") as pvar_db:
                gr.Info("新的修改将在重启服务后生效")
                copy = pvar_db["细化关键词"]
                del copy[simple_keyword_choose]
                pvar_db["细化关键词"] = copy
        return gr.update(choices=list(read_persist_var("细化关键词").keys()), value=None), gr.update(choices=[], value=None)
# 删除子关键词
def del_keyword(simple_keyword_choose, keyword_choose):
    with lock_1:
        if keyword_choose in read_persist_var("细化关键词")[simple_keyword_choose]:
            with shelve.open(f"{config_path}/persist_var") as pvar_db:
                gr.Info("新的修改将在重启服务后生效")
                copy = pvar_db["细化关键词"]
                del copy[simple_keyword_choose][keyword_choose]
                pvar_db["细化关键词"] = copy
        return gr.update(choices=list(read_persist_var("细化关键词")[simple_keyword_choose].keys()), value=None)
# 关键词修改保存
def keyword_save(simple_keyword_choose, keyword_choose, tip, create_keywords):
    with lock_1:
        if create_keywords:
            words = create_keywords.split()
            keyword_dict = {}
            for word in words[1:]:
                keyword_dict[word] = "无"
            with shelve.open(f"{config_path}/persist_var") as pvar_db:
                gr.Info("新的修改将在重启服务后生效")
                copy = pvar_db["细化关键词"]
                copy[words[0]] = keyword_dict
                pvar_db["细化关键词"] = copy
        if tip:
            if keyword_choose:
                if simple_keyword_choose in read_persist_var("细化关键词"):
                    if keyword_choose in read_persist_var("细化关键词")[simple_keyword_choose]:
                        with shelve.open(f"{config_path}/persist_var") as pvar_db:
                            gr.Info("新的修改将在重启服务后生效")
                            copy = pvar_db["细化关键词"]
                            copy[simple_keyword_choose][keyword_choose] = f"（{tip}）"
                            pvar_db["细化关键词"] = copy
                        return simple_keyword_choose, keyword_choose, gr.update(value=None), gr.update(value=None)
            gr.Info("无匹配子关键词（ 可能已被删除 ）")
            return gr.update(choices=list(read_persist_var("细化关键词").keys()), value=None), gr.update(choices=[], value=None), tip, gr.update(value=None)
        return gr.update(choices=list(read_persist_var("细化关键词").keys()), value=None), gr.update(choices=[], value=None), tip, gr.update(value=None)


# Gradio WebUI
######################################################
if __name__ == "__main__":
    with gr.Blocks(title=f"{ai_name} AI") as tweaks:  # css="footer{display:none !important}"
        gr.Markdown(f"""<p align="center"><img src="file/{sign_path}" style="height: 188px"/><p>""")
        gr.Markdown(f"""<center><font size=8>{project_name} 💬</center>""")
        gr.Markdown(f"""<center><font size=4>当前参数规模：{weight_size}</center>""")

        with gr.Tabs():
            with gr.Tab("测试环境"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # gr.Textbox(label='📢', value="当检测不到麦克风或无法触发复制控件，且使用 Chorme 时，请打开下方链接并将当前地址填入后选择 “Enabled”。\n\nchrome://flags/#unsafely-treat-insecure-origin-as-secure", interactive=False, show_copy_button=True)
                        # with gr.Accordion(label="临时配置", open=True): 
                        id_input = gr.Textbox(label='对话 ID', placeholder="请写入用于临时记录对话的 ID")
                        chat_mode_choose = gr.Radio(
                            choices=chatmode, 
                            label="对话模式", 
                            value=chatmode[2], 
                            interactive=True
                        )
                        # history_analyze 在新一轮对话中将历史对话嵌入 Prompt 中
                        # session_analyze 让 AI 针对先前对话提出建议
                        history_analyze = gr.Radio(
                            choices=["开启","关闭"], 
                            label="历史分析", 
                            value="关闭", 
                            interactive=True
                        )
                        query_enhance = gr.Radio(
                            choices=["开启","关闭"], 
                            label="询问逻辑优化", 
                            value="关闭", 
                            interactive=True
                        )
                        query_enhance_1 = gr.Radio(
                            choices=["开启","关闭"], 
                            label="询问关键词优化", 
                            value="关闭", 
                            interactive=True
                        )
                        db_test = gr.Radio(
                            choices=["开启","关闭"], 
                            label="数据库测试", 
                            value="关闭", 
                            interactive=True
                        )
                        save = gr.Button(value="保存至接口")
                        load = gr.Button(value="从接口加载")
                    with gr.Column(scale=3):
                        with gr.Row():
                            # render_markdown=False
                            chatbot = gr.Chatbot(label='Default', show_copy_button=True)
                            chatbot_1 = gr.Chatbot(label='Test', show_copy_button=True, visible=False)
                        audio_answer = gr.Audio(label="音频回复（ 功能测试中 ）", interactive=False)
                        with gr.Column():
                            with gr.Tabs():
                                with gr.Tab("对话小光"):
                                    audio_query = gr.Audio(sources=["microphone"], label='音频询问')
                                    query = gr.Textbox(lines=2, label='文本询问', placeholder="换行：Enter｜发送消息：Shift + Enter")
                                    query_1 = gr.Textbox(lines=2, label='文本询问', placeholder="请等待回复结束", visible=False)
                                    with gr.Row():
                                        audio_check = gr.Checkbox(value=False, label='音频转文本', interactive=True)
                                        clear_history = gr.Button(value="清除历史")
                                        sumbit = gr.Button(value="发送询问", variant="primary")
                                with gr.Tab("数据库检索"):
                                    with gr.Accordion(label="文档检索（ 语意检索 ）", open=True): 
                                        with gr.Row():
                                            with gr.Column():
                                                reteriver_text = gr.Textbox(lines=2, placeholder="留空默认使用最新询问（ Default 窗口 ）检索文档，反之使用填入文本检索文档。", container=False)
                                                with gr.Accordion(label='文档一', open=False) as doc_show: 
                                                    doc = gr.Markdown(value="暂无，请尝试检索或修改配置后检索。")
                                                with gr.Accordion(label='文档二', open=False) as doc_show_1: 
                                                    doc_1 = gr.Markdown(value="暂无，请尝试检索或修改配置后检索。")
                                                with gr.Accordion(label='文档三', open=False) as doc_show_2: 
                                                    doc_2 = gr.Markdown(value="暂无，请尝试检索或修改配置后检索。")
                                                with gr.Accordion(label='文档四', open=False) as doc_show_3: 
                                                    doc_3 = gr.Markdown(value="暂无，请尝试检索或修改配置后检索。")
                                                with gr.Accordion(label='文档五', open=False) as doc_show_4: 
                                                    doc_4 = gr.Markdown(value="暂无，请尝试检索或修改配置后检索。")
                                            with gr.Column():
                                                score_threshold_now = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="相似阈值（ 0 为无阀值；如需使用 0.3 为建议值 ）", interactive=True)
                                                reterive_with_ai = gr.Radio(
                                                    choices=["关闭", "开启"], 
                                                    label="小光 BUFF（ 利用 AI 二次检索文档 ）", 
                                                    value="关闭", 
                                                    interactive=True
                                                )                                
                                                with gr.Row():
                                                    db_choose = gr.Radio(
                                                        choices=["Default", "Test"], 
                                                        value="Default", 
                                                        show_label=False, 
                                                        interactive=True
                                                    )
                                                    start_reterive = gr.Button(value="检索对应数据库")
                                with gr.Tab("对接相关"):
                                    with gr.Column():
                                        with gr.Accordion(label="特征分析（ 转人工时面向对接人员单次触发 ）", open=True):  
                                            features_analyze = gr.Markdown(value="转人工通常代表小光可能未解决其问题（ 数据库不全或者小光分析有误 ），<br/>因此特征分析时将不参考数据库。")
                                            gr.Textbox(value="由小光针对对话分析后，提供些许关键性建议", interactive=False, container=False)
                                        with gr.Accordion(label="回复建议", open=True): 
                                            reply_for_you_prompt = gr.Textbox(lines=2, placeholder="留空默认使用用户最新询问（ Default 窗口 ）生成回复，反之使用填入文本生成回复。", container=False)
                                            with gr.Accordion(label='回复内容：', open=True): 
                                               reply_for_you = gr.Markdown(value="暂无内容，请点击 “生成回复” 按钮")
                                        with gr.Row():
                                            with gr.Column(scale=3):
                                                temperature_now = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperature（ 值越高回复随机性越高 )", interactive=True)
                                            reply_for_you_start = gr.Button(value="生成回复")
                                            features_analyze_start = gr.Button(value="生成分析")

                chat_mode_choose.input(
                    fn=dected_del, 
                    inputs=[id_input], 
                    outputs=[chatbot, chatbot_1, query]
                )
                history_analyze.input(
                    fn=dected_del, 
                    inputs=[id_input], 
                    outputs=[chatbot, chatbot_1, query]
                )
                query_enhance.input(
                    fn=chatgpt_info, 
                    inputs=[], 
                    outputs=[]
                )
                query_enhance_1.input(
                    fn=chatgpt_info, 
                    inputs=[], 
                    outputs=[]
                )
                db_test.input(
                    fn=test_db, 
                    inputs=[id_input, db_test], 
                    outputs=[chatbot, chatbot_1, query]
                )

                save.click(
                    fn=save_setting, 
                    inputs=[chat_mode_choose, history_analyze, query_enhance, query_enhance_1, db_test], 
                    outputs=[]
                )
                load.click(
                    fn=load_from_api,
                    inputs=[id_input, chat_mode_choose, history_analyze, db_test, query, chatbot],
                    outputs=[chat_mode_choose, history_analyze, query_enhance, query_enhance_1, db_test, query, chatbot, chatbot_1]
                )

                start_reterive.click(
                    fn=reterive_gr, 
                    inputs=[id_input, reteriver_text, score_threshold_now, reterive_with_ai, db_choose], 
                    outputs=[doc_show, doc, doc_show_1, doc_1, doc_show_2, doc_2, doc_show_3, doc_3, doc_show_4, doc_4]
                )
                reply_for_you_start.click(
                    fn=simple_infer_gr, 
                    inputs=[id_input, reply_for_you_prompt, temperature_now], 
                    outputs=[reply_for_you]
                )
                features_analyze_start.click(
                    fn=session_analyze_gr, 
                    inputs=[id_input, history_analyze, features_analyze], 
                    outputs=[features_analyze]
                )

                audio_query.stop_recording(
                    fn=audio_text_gr, 
                    inputs=[id_input, chatbot, audio_query, query, audio_check],
                    outputs=[chatbot, query]
                )
                query.submit(
                    fn=model_chat, 
                    inputs=[id_input, db_test, chatbot, chatbot_1, query], 
                    outputs=[chatbot, chatbot_1, audio_query, query, query_1, clear_history, sumbit]
                ).then(
                    fn=model_chat_1, 
                    inputs=[id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot], 
                    outputs=[chatbot]
                ).then(
                    fn=model_chat_1_1, 
                    inputs=[id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot], 
                    outputs=[chatbot_1]
                ).then(
                    fn=model_chat_2, 
                    inputs=[id_input, query_1], 
                    outputs=[audio_answer, audio_query, query, query_1, clear_history, sumbit]
                )
                
                clear_history.click(fn=clear_session,inputs=[id_input],outputs=[chatbot, chatbot_1, query])
                
                sumbit.click(
                    fn=model_chat, 
                    inputs=[id_input, db_test, chatbot, chatbot_1, query], 
                    outputs=[chatbot, chatbot_1, audio_query, query, query_1, clear_history, sumbit]
                ).then(
                    fn=model_chat_1, 
                    inputs=[id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot], 
                    outputs=[chatbot]
                ).then(
                    fn=model_chat_1_1, 
                    inputs=[id_input, chat_mode_choose, history_analyze, query_enhance, query_enhance_1, chatbot], 
                    outputs=[chatbot_1]
                ).then(
                    fn=model_chat_2, 
                    inputs=[id_input, query_1], 
                    outputs=[audio_answer, audio_query, query, query_1, clear_history, sumbit]
                )

            with gr.Tab("临时对话查看") as history_tweak:
                with gr.Column():
                    history_choose = gr.Radio(
                        choices=get_ids(), 
                        label="ID 选择", 
                        interactive=True, 
                    )
                    chatbot_2 = gr.Chatbot(label='历史对话', show_copy_button=True)
                    with gr.Row():
                        refresh = gr.Button(value="刷新")                           
                        del_session = gr.Button(value="删除此对话")

                history_tweak.select(
                    fn=refresh_tab_watch, 
                    inputs=[], 
                    outputs=[history_choose, chatbot_2]
                )
                
                history_choose.input(
                    fn=history_watch, 
                    inputs=[history_choose], 
                    outputs=[chatbot_2]
                )
                
                refresh.click(
                    fn=refresh_watch, 
                    inputs=[history_choose], 
                    outputs=[history_choose, chatbot_2]
                )
                
                del_session.click(
                    fn=del_history, 
                    inputs=[history_choose, chatbot_2], 
                    outputs=[history_choose, chatbot_2]
                )

            with gr.Tab("数据库管理") as database_tweak:
                with gr.Row():
                    with gr.Column(scale=3):
                        vectordb_file = gr.File(label="Default", value=read_all_file_record_key(), file_types=[".md"], interactive=False)
                        file_upload_button = gr.UploadButton(label="点击上传 .md 文件", file_types=[".md"], interactive=True)
                        test_vectordb_file = gr.File(label="Test", value=[], file_types=[".md"], interactive=True)
                        test_file_upload_button = gr.UploadButton(label="点击上传 .md 文件", file_types=[".md"], interactive=True)
                    with gr.Column(scale=1):
                        with gr.Accordion(label="谨慎操作", open=False): 
                            del_file_on = gr.Radio(
                                choices=["Default", "Test"], 
                                label="作用域", 
                                value="Default", 
                                interactive=True
                            )
                            del_file = gr.Textbox(label="删除单个文件（ 谨慎操作 ）", placeholder="此操作不可逆（ 无需后缀名 ）")
                            del_confirm = gr.Button(value="确认删除")
                        with gr.Accordion(label="关键词编辑器", open=False): 
                            simple_keyword_choose = gr.Radio(
                                choices=list(read_persist_var("细化关键词").keys()), 
                                label="父关键词", 
                                interactive=True, 
                            )
                            keyword_choose = gr.Radio(
                                choices=["无"], 
                                label="子关键词", 
                                interactive=True, 
                            )
                            tip = gr.Textbox(lines=2, label='精简说明', placeholder="一段精简的文字用于描述子关键词。\n当前：无")
                            create_keywords = gr.Textbox(lines=5, label='新增关键词组（ 与现有重复视为修改 ）', placeholder="格式：首词组将视为父关键词，其它视为子关键词，彼此之间以空格相隔。\n样本：加速 战斗加速器 网络加速器\n提示：子关键词说明请在创建后选中关键词添加（ 说明可使 AI 优化询问时更准确 ）")
                            load_keywords = gr.Button(value="加载选中至新增")
                            simple_keyword_del = gr.Button(value="删除选中父关键词")
                            keyword_del = gr.Button(value="删除选中子关键词")
                            save_changes = gr.Button(value="保存说明或词组")
                        refresh_db = gr.Button(value="刷新页面")

                database_tweak.select(
                    fn=refresh_db_watch,
                    inputs=[],
                    outputs=[vectordb_file, test_vectordb_file, simple_keyword_choose, keyword_choose]
                )
                
                file_upload_button.upload(
                    fn=upload_process_gr, 
                    inputs=[file_upload_button], 
                    outputs=[vectordb_file]
                )
                
                test_vectordb_file.upload(
                    fn=test_upload_process_gr, 
                    inputs=[test_vectordb_file], 
                    outputs=[test_vectordb_file]
                )
                
                test_vectordb_file.clear(
                    fn=test_del_all_file_process_gr, 
                    inputs=[], 
                    outputs=[test_vectordb_file]
                )
                
                test_file_upload_button.upload(
                    fn=test_upload_process_gr, 
                    inputs=[test_file_upload_button], 
                    outputs=[test_vectordb_file]
                )
                
                del_confirm.click(
                    fn=del_file_process_gr, 
                    inputs=[del_file_on, del_file], 
                    outputs=[del_file, vectordb_file, test_vectordb_file]
                )
                
                simple_keyword_choose.input(
                    fn=keyword_list, 
                    inputs=[simple_keyword_choose], 
                    outputs=[keyword_choose]
                )
                
                keyword_choose.input(
                    fn=load_tip, 
                    inputs=[simple_keyword_choose, keyword_choose], 
                    outputs=[tip]
                )
                
                load_keywords.click(
                    fn=keywords_load, 
                    inputs=[simple_keyword_choose], 
                    outputs=[create_keywords]
                )
                
                simple_keyword_del.click(
                    fn=del_simple_keyword, 
                    inputs=[simple_keyword_choose], 
                    outputs=[simple_keyword_choose, keyword_choose]
                )
                
                keyword_del.click(
                    fn=del_keyword, 
                    inputs=[simple_keyword_choose, keyword_choose], 
                    outputs=[keyword_choose]
                )
                
                save_changes.click(
                    fn=keyword_save, 
                    inputs=[simple_keyword_choose, keyword_choose, tip, create_keywords], 
                    outputs=[simple_keyword_choose, keyword_choose, tip, create_keywords]
                )
                
                refresh_db.click(
                    fn=refresh_db_watch,
                    inputs=[],
                    outputs=[vectordb_file, test_vectordb_file, simple_keyword_choose, keyword_choose]
                )

            with gr.Tab("应用程序编程接口") as api_tweak:
                with gr.Row():
                    with gr.Column(scale=3):
                        code = gr.Code(
                            value=api_guid, 
                            language="python", 
                            interactive=False, 
                            label="LangServe 提供服务（ 基于 FastAPI ）"
                        )
                    with gr.Column(scale=1):
                        system_command = gr.Textbox(
                            lines= 3, 
                            label="系统指令",  
                            placeholder="用于更改 AI 行为：如人物设定、语言风格、任务模式等...", 
                            interactive = True
                        )
                        chat_mode_choose_1 = gr.Radio(
                            choices=chatmode, 
                            label="对话模式", 
                            value=chatmode[0], 
                            interactive=True
                        )
                        history_analyze_1 = gr.Radio(
                            choices=["开启","关闭"], 
                            label="历史分析", 
                            value="关闭", 
                            interactive=True
                        )
                        query_enhance_2 = gr.Radio(
                            choices=["开启","关闭"], 
                            label="询问逻辑优化", 
                            value="关闭", 
                            interactive=True, 
                        )
                        query_enhance_2_1 = gr.Radio(
                            choices=["开启","关闭"], 
                            label="询问关键词优化", 
                            value="关闭", 
                            interactive=True
                        )
                        db_test_1 = gr.Radio(
                            choices=["开启","关闭"], 
                            label="数据库测试", 
                            value="关闭", 
                            interactive=True
                        )
                        with gr.Accordion(label="进阶配置", open=False):
                            history_len = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="历史对话量", interactive=True)
                            history_time = gr.Slider(minimum=30, maximum=21600, value=7200, step=1, label="历史对话有效时长（ 秒 ）", interactive=True)
                            max_tokens = gr.Slider(minimum=200, maximum=2048, value=800, step=1, label="Max Tokens", interactive=True)
                            temperature = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Temperature", interactive=True)
                            reteriver_k = gr.Slider(minimum=1, maximum=30, value=20, step=1, label="预取回文本量", interactive=True)
                            reteriver_k_final = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="最大取回文本量", interactive=True)
                            reteriver_k_relate = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="最大追问文本量", interactive=True)
                            score_threshold = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.1, label="相似阈值", interactive=True)
                        api_refresh = gr.Button(value="刷新配置")
                        api_save = gr.Button(value="保存配置")
                        api_reset = gr.Button(value="重制配置")


                api_tweak.select(
                    fn=load_from_api_1,
                    inputs=[],
                    outputs=[system_command, chat_mode_choose_1, history_analyze_1, query_enhance_2, query_enhance_2_1, db_test_1, history_len, history_time, max_tokens, temperature, reteriver_k, reteriver_k_final, reteriver_k_relate, score_threshold]
                )
                
                query_enhance_2.input(
                    fn=chatgpt_info, 
                    inputs=[], 
                    outputs=[]
                )
                
                query_enhance_2_1.input(
                    fn=chatgpt_info, 
                    inputs=[], 
                    outputs=[]
                )
                
                api_refresh.click(
                    fn=load_from_api_1,
                    inputs=[],
                    outputs=[system_command, chat_mode_choose_1, history_analyze_1, query_enhance_2, query_enhance_2_1, db_test_1, history_len, history_time, max_tokens, temperature, reteriver_k, reteriver_k_final, reteriver_k_relate, score_threshold]
                )
                
                api_save.click(
                    fn=save_setting_1, 
                    inputs=[system_command, chat_mode_choose_1, history_analyze_1, query_enhance_2, query_enhance_2_1, db_test_1, history_len, history_time, max_tokens, temperature, reteriver_k, reteriver_k_final, reteriver_k_relate, score_threshold], 
                    outputs=[]
                )
                
                api_reset.click(
                    fn=reset_setting, 
                    inputs=[], 
                    outputs=[system_command, chat_mode_choose_1, history_analyze_1, query_enhance_2, query_enhance_2_1, db_test_1, history_len, history_time, max_tokens, temperature, reteriver_k, reteriver_k_final, reteriver_k_relate, score_threshold]
                )

    tweaks.queue(default_concurrency_limit=concurrency_limit)  # WebUI 内最大并发处理量
    tweaks.launch(allowed_paths=[f"/{project_name_1}/packages/sources"], show_api=False, favicon_path=f"{sign_path}", server_name="0.0.0.0", server_port=6006)
    # tweaks.queue(api_open=False).launch(allowed_paths=[f"/{project_name_1}/packages/sources"], show_api=False, favicon_path=f"{sign_path}", server_name="0.0.0.0", server_port=6006)


