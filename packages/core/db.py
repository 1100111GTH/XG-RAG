import os
import sys
# 把父路径添加到检索目标中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config.config import embedding_path, init_system_prompt, config_path, vector_path, vector_original_path, small_model_loadon
######################################################
import shelve  # For persist variable
from threading import Lock
######################################################
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter


# 项目配置
######################################################
# 写入持久化变量
def write_persist_var(name, value):
    with shelve.open(f"{config_path}/persist_var") as pvar_db:
        pvar_db[name] = value
# 读取单个持久化变量
def read_persist_var(name):
    with shelve.open(f"{config_path}/persist_var") as pvar_db:
        return pvar_db[name]
"""
# 全部持久化数据
def read_all_persist_var():
    with shelve.open(f"{config_path}/persist_var") as pvar_db:
        for name in pvar_db:
            print(f"{name}: {pvar_db[name]}")
def read_all_file_record():
    with shelve.open(f"{vector_original_path}/file_record") as pvec_db:
        for doc in pvec_db:
            print(f"{doc}: {pvar_db[doc]}")
"""
# 数据库文件
######################################################
# 写入持久化记录
def write_file_record(doc_path, ids):
    with shelve.open(f"{vector_original_path}/file_record") as pvec_db:
        pvec_db[doc_path] = ids
# 删除单个持久化记录
def del_file_record(doc_path):
    with shelve.open(f"{vector_original_path}/file_record") as pvec_db:
        del pvec_db[doc_path]
# 读取单个持久化记录
def read_file_record(doc_path):
    with shelve.open(f"{vector_original_path}/file_record") as pvec_db:
        return pvec_db[doc_path]
# 读取全部持久化记录键
def read_all_file_record_key():
    with shelve.open(f"{vector_original_path}/file_record") as pvec_db:
        return list(pvec_db.keys())


# 初始化配置
######################################################
# 定义目录文件检测
def local_file_dected(prefix):
    files = os.listdir(config_path)
    for file in files:
        if file.startswith(prefix):
            return True
    return False
# 初始化持久化变量（ 无则创建，有则略过 ）
if local_file_dected("persist_var") == True:
    pass
elif local_file_dected("persist_var") == False:
    write_persist_var("first_time", True)
    write_persist_var("system_prompt", init_system_prompt)
    write_persist_var("chat_mode_choose", "数据库对话（ 无限制 ）")
    write_persist_var("history_analyze", "关闭")
    write_persist_var("query_enhance", "关闭")
    write_persist_var("query_enhance_1", "关闭")
    write_persist_var("db_test", "关闭")
    write_persist_var("history_len", 3)
    write_persist_var("history_time", 7200)  # Second
    write_persist_var("max_tokens", 800)
    write_persist_var("temperature", 0)
    write_persist_var("reteriver_k", 20)
    write_persist_var("reteriver_k_final", 5)
    write_persist_var("reteriver_k_relate", 10)
    write_persist_var("score_threshold", 0.3)
    write_persist_var("细化关键词", {"游戏": {"手机游戏": "（ 手机移动平台上的游戏 ）", "电脑游戏": "（ 桌面端非移动平台上的游戏 ）", "主机游戏": "（ 游戏专用机器上的游戏 ）"}})

# Embedding
# local_embedding = HuggingFaceEmbeddings(model_name=embedding_path, multi_process=True)
local_embedding = HuggingFaceEmbeddings(model_name=embedding_path, model_kwargs={"device": small_model_loadon})


# 定义文档处理（ 向量数据库 ）类
######################################################
class DocumentsHandle:
    
    def __init__(self):
        self.memory_vdb = None
        self._lock = Lock()  # 逐个执行，避免出错

    def documents_handle(self, doc_path):
        with self._lock:
            if self.memory_vdb == None or not self.memory_vdb.docstore._dict:
                loader = TextLoader(doc_path)  # loader = TextLoader(doc_path, encoding='utf8')
                doc = loader.load()
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")], strip_headers=False)
                docs = splitter.split_text(doc[0].page_content)
                self.memory_vdb = FAISS.from_documents(docs, local_embedding)
                self.memory_vdb.save_local(vector_path, "default")
                ids = list(self.memory_vdb.docstore._dict.keys())
                write_file_record(doc_path, ids)
            else:
                loader = TextLoader(doc_path)
                doc = loader.load()
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")], strip_headers=False)
                docs = splitter.split_text(doc[0].page_content)
                ids = self.memory_vdb.add_documents(docs)
                self.memory_vdb.save_local(vector_path, "default")  # 若不 save 则数据将保存在内存中
                write_file_record(doc_path, ids)
    
    def load_local(self):
        with self._lock:
            self.memory_vdb = FAISS.load_local(vector_path, local_embedding, "default", allow_dangerous_deserialization=True)

    # 未做文件校验，调用时传入不存在文件可能 raise KeyError
    def file_del(self, doc_path):
        with self._lock:
            if not read_all_file_record_key():
                raise ValueError("Nothing in vectordb!")
            else:
                ids = read_file_record(doc_path)
                for id in ids:
                    del self.memory_vdb.docstore._dict[id]
                    self.memory_vdb.save_local(vector_path, "default")
                os.remove(doc_path)
                del_file_record(doc_path)

class TestDocumentsHandle:

    def __init__(self):
        self.test_memory_vdb = None
        self._lock = Lock()
        self.file_record = {}  # 定义数据库测试文件记录

    def documents_handle(self, doc_path):
        with self._lock:
            if self.test_memory_vdb == None or not self.test_memory_vdb.docstore._dict:
                loader = TextLoader(doc_path)
                doc = loader.load()
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")], strip_headers=False)
                docs = splitter.split_text(doc[0].page_content)
                self.test_memory_vdb = FAISS.from_documents(docs, local_embedding)
                ids = list(self.test_memory_vdb.docstore._dict.keys())
                self.file_record[doc_path] = ids
            else:
                loader = TextLoader(doc_path)
                doc = loader.load()
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")], strip_headers=False)
                docs = splitter.split_text(doc[0].page_content)
                ids = self.test_memory_vdb.add_documents(docs)
                self.file_record[doc_path] = ids  

    # 未做文件校验，调用时传入不存在文件可能 raise KeyError
    def file_del(self, doc_path):
        with self._lock:
            if not self.file_record:
                raise ValueError("Nothing in test_vectordb!")
            else:
                for id in self.file_record[doc_path]:
                    del self.test_memory_vdb.docstore._dict[id]
                os.remove(doc_path)
                del self.file_record[doc_path]

    def all_file_del(self):
        with self._lock:
            if not self.file_record:
                raise ValueError("Nothing in test_vectordb!")
            else:
                self.test_memory_vdb = None
                for doc in list(self.file_record.keys()):  # 由于空值不会循环，也就等于不会删除不存在的文件导致报错
                    os.remove(doc)
                self.file_record = {}


