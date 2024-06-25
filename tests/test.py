import gradio as gr


chatmode = ["基础对话", "数据库对话", "数据库对话（ 无限制 ）"]
sign_path = "/Volumes/storage/projects/open_source/xg_rag/packages/sources/sign.png"


with gr.Blocks(title=f"小光 AI") as tweaks:  # css="footer{display:none !important}"
    gr.Markdown(f"""<p align="center"><img src="file/{sign_path}" style="height: 188px"/><p>""")
    gr.Markdown(f"""<center><font size=8>XG RAG 💬</center>""")
    gr.Markdown(f"""<center><font size=4>当前参数规模：320 亿</center>""")

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

        with gr.Tab("临时对话查看") as history_tweak:
            with gr.Column():
                history_choose = gr.Radio(
                    choices=["无"], 
                    label="ID 选择", 
                    interactive=True, 
                )
                chatbot_2 = gr.Chatbot(label='历史对话', show_copy_button=True)
                with gr.Row():
                    refresh = gr.Button(value="刷新")                           
                    del_session = gr.Button(value="删除此对话")

        with gr.Tab("数据库管理") as database_tweak:
            with gr.Row():
                with gr.Column(scale=3):
                    vectordb_file = gr.File(label="Default", value=[], file_types=[".md"], interactive=False)
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
                            choices=[], 
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

        with gr.Tab("应用程序编程接口") as api_tweak:
            with gr.Row():
                with gr.Column(scale=3):
                    code = gr.Code(
                        value="", 
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

tweaks.queue(default_concurrency_limit=60)  # WebUI 内最大并发处理量
tweaks.launch(allowed_paths=["/Volumes/storage/projects/open_source/xg_rag/packages/sources"], show_api=False, favicon_path=f"{sign_path}", server_name="0.0.0.0", server_port=6006)