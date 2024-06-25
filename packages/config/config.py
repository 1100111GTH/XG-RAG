import torch


# 基础参数
######################################################
ai_name = "小光"
weight_size = "320 亿"
project_name = "XG RAG"
project_name_1 = "xg_rag"  # 此处建议遵守 Python 变量命名规范（ 全小写、“_” 替换空格和 “-” ）
# 调用 ChatGPT
## 不使用时忽略就好
openai_api_key=""
model="gpt-4o"
# API 相关
ssl_keyfile=""  # 路径
ssl_certfile=""  # 路径
default_challenge = "~AxKj-3vK*,Y;ZS"  # 也可理解为默认密码（ 建议本地环境下使用 ）
max_request = 60  # 最大同时请求
# Swift Deploy
gpu_memory_utilization = 0.9
max_model_len = 3000
# Gradio ( WebUI )
concurrency_limit = 12  # 最大任务同时处理量
# Langsmith
## 建议留空参数，使用本地 Arize Phoneix 可保护数据安全
langsmith_project = ""
langsmith_api_key = ""
# 对话参数
# init_system_prompt = "You are a helpful assistant."
init_system_prompt = f"你叫{ai_name}，一位垂直解答手游相关问题的 AI 专家，由 GTH 独立研发。"
bad_answer = "对不起，我只能回答您关于本平台的相关问题。"


# 模型相关
######################################################
llm_path = f"/{project_name_1}/packages/model/Qwen1.5-32B-Chat-AWQ"
asr_path = f"/{project_name_1}/packages/model/whisper-large-v3"
embedding_path = f"/{project_name_1}/packages/model/bge-large-zh-v1.5"
reranker_path = f"/{project_name_1}/packages/model/bge-reranker-v2-m3"
sound_color_path = f"/{project_name_1}/packages/model/seed_1345_restored_emb.pt"
openai_api_model_name = "qwen1half-32b-chat-awq"


# 无需改动
######################################################
gpu_size = torch.cuda.device_count()  # 检测显卡数量
config_path = f"/{project_name_1}/packages/config"  # 配置文件路径
vector_path = f"/{project_name_1}/packages/database/vector"  # 数据库路径（ Faiss ）
vector_original_path = f"/{project_name_1}/packages/database"  # 数据库路径（ 源文件 ）
rsa_priv = f"/{project_name_1}/packages/sources/keys/private.pem"  # For API password check
rsa_pub = f"/{project_name_1}/packages/sources/keys/public.pem"  # For API password check
tensor_parallel_size = gpu_size - 1  # Swift Deploy（ vLLM ）参数
cuda_visible_devices = ",".join(str(d) for d in range(tensor_parallel_size))  # shell 脚本用环境变量（ LLM ）
cuda_visible_devices_1 = ",".join(str(d) for d in range(gpu_size))  # shell 脚本用环境变量（ Embedding、Reranker ）
small_model_loadon = f"cuda:{tensor_parallel_size}"  # 小模型加载显卡（ Embedding、Reranker ）


# For launch_all.sh & Dockerfile
######################################################
if __name__ == "__main__":
    print(f"project_name={project_name_1}")
    print(f"cuda_visible_devices={cuda_visible_devices}")
    print(f"cuda_visible_devices_1={cuda_visible_devices_1}")
    print(f"openai_api_model_name={openai_api_model_name}")
    print(f"llm_path={llm_path}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    print(f"gpu_memory_utilization={gpu_memory_utilization}")
    print(f"max_model_len={max_model_len}")


