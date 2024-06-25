import os
import sys
# 把父路径添加到检索目标中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from packages.config.config import project_name, ssl_keyfile, ssl_certfile, default_challenge, max_request, rsa_priv
from packages.core.api_call import (
    better_query, 
    upload_process, 
    test_upload_process, 
    test_read_all_file_record_key, 
    del_file_process, 
    test_del_file_process, 
    test_del_all_file_process, 
    retrieve, 
    audio_text, 
    # text_audio, 
    infer, 
    simple_infer, 
    plus_links, 
    session_analyze, 
    clean_history
)
######################################################
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import asyncio
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from langserve import add_routes
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
######################################################
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


# Phoneix（ 替代 LangSmith ）
endpoint = "http://127.0.0.1:7007/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LangChainInstrumentor().instrument()


# FastAPI
######################################################
app = FastAPI(title=f"{project_name}")

challenges = {default_challenge}  # 默认密码应该在本地环境使用，请尽可能确保不要泄漏
# 生成随机挑战字符串
def generate_challenge():
    return base64.b64encode(os.urandom(32)).decode('utf-8')  # base64 编码确保正常传输，header 中可忽略
# 使用私钥解密挑战
def decrypt_password(encrypted_password: str) -> str:
    encrypted_data = base64.b64decode(encrypted_password)
    cipher = PKCS1_OAEP.new(RSA.import_key(open(rsa_priv, "rb").read()))
    return cipher.decrypt(encrypted_data).decode('utf-8')

add_routes(app, better_query, path="/better_query", disabled_endpoints=["playground"])
add_routes(app, upload_process, path="/upload_process", disabled_endpoints=["playground"])
add_routes(app, test_upload_process, path="/test_upload_process", disabled_endpoints=["playground"])
add_routes(app, test_read_all_file_record_key, path="/test_read_all_file_record_key", disabled_endpoints=["playground"])
add_routes(app, del_file_process, path="/del_file_process", disabled_endpoints=["playground"])
add_routes(app, test_del_file_process, path="/test_del_file_process", disabled_endpoints=["playground"])
add_routes(app, test_del_all_file_process, path="/test_del_all_file_process", disabled_endpoints=["playground"])
add_routes(app, retrieve, path="/retrieve", disabled_endpoints=["playground"])
add_routes(app, audio_text, path="/audio_text", disabled_endpoints=["playground"])
# add_routes(app, text_audio, path="/text_audio", disabled_endpoints=["playground"])
add_routes(app, infer, path="/infer", disabled_endpoints=["playground"])
add_routes(app, simple_infer, path="/simple_infer", disabled_endpoints=["playground"])
add_routes(app, plus_links, path="/plus_links", disabled_endpoints=["playground"])
add_routes(app, session_analyze, path="/session_analyze", disabled_endpoints=["playground"])
add_routes(app, clean_history, path="/clean_history", disabled_endpoints=["playground"])

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")
@app.get("/get_challenge")
async def get_challenge():
    challenge = generate_challenge()
    challenges.add(challenge)
    return {'Challenge': challenge}
@app.post("/del_challenge")
async def del_challenge(password: Request):
    password = await password.json()
    challenge = decrypt_password(password['Password'])
    if challenge == default_challenge:
        Response("Warrning! don't trying to send this again. please contact the admin", status_code=401)
    else:
        pass
    try:
        challenges.remove(challenge)
    except KeyError:
        Response("Error! challenge not found", status_code=400)

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # if request.url.scheme == "http" and "localhost" not in request.url.netloc:
        if request.url.scheme == "http":
            url = request.url.replace(scheme="https", port=2031)  # 确保端口匹配 Langserve
            return RedirectResponse(url=url)
        return await call_next(request)
class SemaphoreMiddleware(BaseHTTPMiddleware):
    '''可在 config.py 中修改最大请求数'''

    def __init__(self, app, max_concurrent_requests: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(self, request: Request, call_next):
        await self.semaphore.acquire()
        try:
            response = await call_next(request)
        finally:
            self.semaphore.release()
        return response
class ChallengeResponseMiddleware(BaseHTTPMiddleware):
    '''加密前为 challenge，加密后为 password'''

    async def dispatch(self, request: Request, call_next):
        if request.url.path != '/get_challenge':
            if request.url.path != '/del_challenge':
                password = request.headers.get('P')
                if not password:
                    return Response("Unauthorized", status_code=401)
                try:
                    challenge = decrypt_password(password)
                except Exception:
                    return Response("Error! please contact the admin", status_code=400)
                if challenge not in challenges:
                    return Response("Unauthorized", status_code=401)
        return await call_next(request)


if ssl_keyfile and ssl_certfile:
    app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(SemaphoreMiddleware, max_concurrent_requests=max_request)
app.add_middleware(ChallengeResponseMiddleware)


if __name__ == "__main__":
    import uvicorn


    if ssl_keyfile and ssl_certfile:
        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)


r"""
          _____                _____                       _____          
         /\    \              /\    \                     /\    \         
        /..\    \            /..\    \                   /..\____\        
       /....\    \           \...\    \                 /.../    /        
      /......\    \           \...\    \               /.../    /         
     /.../\...\    \           \...\    \             /.../    /          
    /.../  \...\    \           \...\    \           /.../____/           
   /.../    \...\    \          /....\    \         /....\    \           
  /.../    / \...\    \        /......\    \       /......\    \   _____  
 /.../    /   \...\ ___\      /.../\...\    \     /.../\...\    \ /\    \ 
/.../____/  ___\...|    |    /.../  \...\____\   /.../  \...\    /..\____\
\...\    \ /\  /...|____|   /.../    \../    /   \../    \...\  /.../    /
 \...\    /..\ \../    /   /.../    / \/____/     \/____/ \...\/.../    / 
  \...\   \...\ \/____/   /.../    /                       \....../    /  
   \...\   \...\____\    /.../    /                         \..../    /   
    \...\  /.../    /    \../    /                          /.../    /    
     \...\/.../    /      \/____/                          /.../    /     
      \....../    /                                       /.../    /      
       \..../    /                                       /.../    /       
        \../____/                                        \../    /        
                                                          \/____/                                 
"""


