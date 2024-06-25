api_guid = r"""
# 第一步
######################################################
# 请先确保已安装 Python & langserve[client]（ pip3 install langserve[client] ）
import base64
import requests
from langserve import RemoteRunnable
import time


# 第二步
######################################################
# localhost:2031 替换为公网 ip 和 Docker 映射端口
# 在调用 API 前首先获取 Challenge，后加密成 Password
challenge = requests.get("http://localhost:2031/get_challenge").json()

def encrypt_challenge(challenge: str) -> str:
    cipher = PKCS1_OAEP.new(RSA.import_key(open("/xg_rag/packages/sources/key/public.pem", "rb").read()))
    encrypted_challenge = cipher.encrypt(challenge.encode())
    return base64.b64encode(encrypted_challenge).decode('utf-8')

password = encrypt_challenge(challenge["Challenge"])

test_read_all_file_record_key = RemoteRunnable(url="http://localhost:2031/test_read_all_file_record_key/", headers={"P": password})  # 用于检测测试数据库是否存在文件
retrieve = RemoteRunnable(url="http://localhost:2031/retrieve/", headers={"P": password})  # 数据库检索接口
audio_text = RemoteRunnable(url="http://localhost:2031/audio_text/", headers={"P": "password"})  # 音频处理接口
infer = RemoteRunnable(url="http://localhost:2031/infer/", headers={"P": "password"})  # 主要推理接口
simple_infer = RemoteRunnable(url="http://localhost:2031/simple_infer/", headers={"P": password})  # 回复建议接口
plus_links = RemoteRunnable(url="http://localhost:2031/plus_links/", headers={"P": "password"})  # 获取资料链接
session_analyze = RemoteRunnable(url="http://localhost:2031/session_analyze/", headers={"P": password})  # 特征分析接口
clean_history = RemoteRunnable(url="http://localhost:2031/clean_history/", headers={"P": "password"})  # 清除有关缓存（ 数据 ）


# 第三步（ API 示范｜Python ）
######################################################
# 在客户转至人工客服对话前建议调用 session_analyze 接口，接口将返回一段文字可能有助于对接人员理解先前对话并获取相关建议
## 如果 API 配置中已开启历史分析，则无需提供 input 键值对，系统将自动从 Redis 中获取信息，否则建议以 “客户：说的话 小光：说的话” 这类格式提供客户端对话信息（ 提倡仅提供最新三轮对话数据，降低压力 ）
session_analyze.invoke(
    { 
        "input": f"客户：xxxx 小光：xxxx 客户：xxx 小光：xxx", 
        "id": id_input, 
        "from": "customer", 
    }
)

# query 替换为用户提问
# usr_id 替换为账户唯一标识（ “dbtest_” 是项目的保留字符串，请勿加入至 usr_id ）
pre_work = infer.invoke(
    {
        "time": 1, 
        "input": "query", 
        "id": "usr_id", 
        "from": "customer", 
        "llm": 0
    }
)
if not pre_work.get("stop", False):
    answer = infer.stream(
        {
            "time": 2, 
            "input": pre_work["input"], 
            "id": "usr_id", 
            "from": "customer", 
            "llm": 0
        }
    )
    for character in answer:
        time.sleep(0.03)
        print(character)  # print 部分根据需要可自行加工
    print(plus_links.invoke({"id": "usr_id"}))  # 显示提取链接
else:
    print(pre_work["stop"])

# 当客服同学不清楚如何回答用户时可提供此接口帮助生成回复
## 当前仅提供针对 “数据库对话（ 无限制 ）” 模式下的回复
## LLM 的 Temperature 参数可单独提供（ 0 - 1 范围 ）
simple_infer.invoke({"input": "query", "temperature": 0})

# 此接口提供向量数据库检索功能
## 通常情况下只需要提供 retrive 参数即可调用，其它参数默认调用 API 配置 （ 如果未提供 ai_check 即代表不使用 ai 对文档进行二次检索；接口检测逻辑是是否存在值，也就是说 ai_check 为 no 也会被启用，当不需要使用时，直接放弃参数填入即可，use_threshold 同理 ）。
## choose 用于选择作用数据库，另一选项是 test；k 是未经过 Reranker 处理时的取回文档数量；score_threshold 是文档检索时使用的阈值
## Reranker 的 top_n 暂不支持在接口中临时配置，调用的 API 配置
# 当作用于 test 数据库时建议通过 test_read_all_file_record_key 接口判断数据库是否存在数据，没有数据可能会报错
## test_read_all_file_record_key.invoke("") 
retrieve.invoke({"choose": "default", "use_threshold": "yes", "k": 20, "score_threshold": 0.3, "ai_check": "yes", "retrive": "query"})

# 当前接口用于音频转文本或者间接通过 simple_infer 接口生成回复
## 写入 {"with_infer": "yes"} 键值对参数即可生成回复（ List 类型回复，内含两个对象，第一个为转译的文本，第二个为回复 ）
# 目前服务器不会处理 mp3 等音频格式（ 降低服务器压力 ），建议在客户端中使用 ffmpeg 或其它工具生成相关数据
# 建议录制时使用单声道
# ASR 模型使用的是 whisper-large-v3 ，支持多种语言，详情请前往 Hugging Face 查阅 openai/whisper-large-v3 页面
# sampling_rate 代表音频采样率
## 样本：48000
# raw 是音频数据，含纳了音频信号的振幅值
## raw 数据需为 List 类型，且不可是 Numpy 数组，raw 内单个对象类型也需要为 Python int 类型，不可以是 numpy.int 等数据类型
### 样本：[-25, -2, 22, ..., 64, 66, 68]
audio_text.invoke({"with_infer": "yes", "sampling_rate": sr, "raw": y_list})

# 当用户结束对话时建议运行以下代码以清除服务器内存中的相关记录
# 每位用户在服务器中最多存放 3 轮对话记录（ 默认配置，可修改 ）
# 未清除的数据将于 2 小时后释放其内存空间（ 默认配置，可修改 ）
# 调用此接口若提供 challenge 键则会删除服务器中对应的 challenge（ 后续使用此 challenge 生成的 password 将无法访问服务 ）
## 在服务器中 challenge 暂未设置过期时间，为了安全考虑建议在完成对话后调用接口清除以保证安全
clean_history.invoke({"id": "usr_id", "challenge": challenge["Challenge"]})  # 运行完毕后传回 Done（ str 类型 ）


# 第三步（ API 示范｜curl ）
######################################################
# 在 Python 环境外针对 challenge 的 RSA 加密和 base64 编码可以使用 OpenSSL
# 部分代码展示已简化，请查看上方 Python 代码获取具体细节，逻辑类似

# 特征分析
## 如果 API 配置中已开启历史分析，则无需提供 input 键值对，系统将自动从 Redis 中获取信息
curl --location --request POST 'http://localhost:2031/session_analyze/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"input": "客户：xxxx 小光：xxxx 客户：xxx 小光：xxx", "id": id_input, "from": "customer"}}'

# query 替换为用户提问
# usr_id 替换为账户唯一标识（ str 类型；“dbtest_” 是项目的保留字符串，请勿加入至 usr_id ）
curl --location --request POST 'http://localhost:2031/infer/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"time": 1, "input": "query", "id": "usr_id", "from": "customer", "llm": 0 }}'
# 如果没有返回 stop
## 如 {"output":{"input":"如何卸载畅玩游戏"},"callback_events":[],"metadata":{"run_id":"e6d34fbd-281c-4a70-bdc8-1a20237cee0f"}}
## stop 类似 {"output":{"stop": "对不起，我只能回答您关于本平台的相关问题。"},"callback_events":[],"metadata":{"run_id":"e6d34fbd-281c-4a70-bdc8-1a20237cee0f"}}
curl --location --request POST 'http://localhost:2031/infer/stream' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"time": 2, "input": 这里填入第一次请求返回 input 键的值, "id": "usr_id", "from": "customer", "llm": 0}}'
# 再获取一下链接
curl --location --request POST 'http://localhost:2031/plus_links/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"id": "usr_id"}}'

# 当客服同学不清楚如何回答用户时可提供此接口帮助生成回复
curl --location --request POST 'http://localhost:2031/simple_infer/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"input": "query", "temperature": 0}}'

# 检索数据库
curl --location --request POST 'http://localhost:2031/retrieve/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"ai_check": "yes", "retrive": "query"}}'

# 音频生成回复
curl --location --request POST 'http://localhost:2031/audio_text/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"with_infer": "yes", "sampling_rate": sr, "raw": y_list}}'

# 最后清除对应记录
curl --location --request POST 'http://localhost:2031/clean_history/invoke' \
    --header 'Content-Type: application/json' \
    --header 'P: password' \
    --data-raw '{"input": {"id": "usr_id"}}'
"""


