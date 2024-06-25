"""
基于 langchain_cohere.rerank.py 重写
from langchain_cohere import CohereRerank
"""


import os
import sys
# 把父路径添加到检索目标中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config.config import reranker_path, small_model_loadon
from core.db import read_persist_var
######################################################
from typing import Optional, Sequence
######################################################
from langchain.schema import Document
from langchain.pydantic_v1 import Extra, root_validator
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder
from torch.nn import DataParallel
from langchain_cohere import CohereRerank


class BgeRerank(BaseDocumentCompressor):

    model_name:str = reranker_path
    # 最终返回的文档数量
    top_n: int = read_persist_var("reteriver_k_final")
    # 将模型启动到特定 GPU
    model: CrossEncoder = CrossEncoder(model_name, device=small_model_loadon)
    # device_ids 设置使用多个的 GPU（ 但似乎并没有达到加速效果，显存占用量还更大了 ）
    # model.model = DataParallel(model.model, device_ids=[0,1])

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        # LangServe 似乎无法支持 numpy.float32 数据类型
        scores = scores.tolist()
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results


