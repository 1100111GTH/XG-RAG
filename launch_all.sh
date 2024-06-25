#!/bin/bash
# c_l_c = caches, logs, comprehensive


# 在脚本目录基础上执行命令
cd "$(dirname "$0")"

python_var=$(python3 ./packages/config/config.py)
export project_name=$(echo "$python_var" | grep 'project_name=' | cut -d'=' -f2)
export cuda_visible_devices=$(echo "$python_var" | grep 'cuda_visible_devices=' | cut -d'=' -f2)
export cuda_visible_devices_1=$(echo "$python_var" | grep 'cuda_visible_devices_1=' | cut -d'=' -f2)
export openai_api_model_name=$(echo "$python_var" | grep 'openai_api_model_name=' | cut -d'=' -f2)
export llm_path=$(echo "$python_var" | grep 'llm_path=' | cut -d'=' -f2)
export tensor_parallel_size=$(echo "$python_var" | grep 'tensor_parallel_size=' | cut -d'=' -f2)
export gpu_memory_utilization=$(echo "$python_var" | grep 'gpu_memory_utilization=' | cut -d'=' -f2)
export max_model_len=$(echo "$python_var" | grep 'max_model_len=' | cut -d'=' -f2)


# 启动 Swift
## export TOKENIZERS_PARALLELISM=false && 
screen -S core -d -m bash -c '
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices && 
script -f /$project_name/c_l_c/swift_check.log -c "swift deploy --model_type $openai_api_model_name --model_cache_dir $llm_path --tensor_parallel_size $tensor_parallel_size --gpu_memory_utilization $gpu_memory_utilization --max_model_len $max_model_len"
'
sleep 3s
while [ $(grep -c "Uvicorn running on" /$project_name/c_l_c/swift_check.log) -eq 0 ]
do
        sleep 3s
        echo "Waiting for SWIFT..."
done
echo "( core ) SWIFT is running"


# 启动 Redis
## Redis 会在启动目录下生成 dump.rdb 文件，以防止突发的文件丢失
screen -S session -d -m bash -c '
cd /$project_name/c_l_c && 
redis-server
'
echo "( session ) Redis may running"


# 启动 LangServe
## export TOKENIZERS_PARALLELISM=false && 
screen -S api -d -m bash -c '
cd /$project_name && 
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices_1 && 
script -f /$project_name/c_l_c/langserve_check.log -c "langchain serve --host 0.0.0.0 --port 2031"
'
sleep 3s
while [ $(grep -c "Application startup" /$project_name/c_l_c/langserve_check.log) -eq 0 ]
do
        sleep 1s
        echo "Waiting for LangServe..."
done
echo "( api ) LangServe is running"


# 启动 Gradio
screen -S tweaks -d -m bash -c '
script -f /$project_name/c_l_c/gradio_check.log -c "python3 /$project_name/packages/tweaks.py"
'
sleep 3s
while [ $(grep -c "Running on local" /$project_name/c_l_c/gradio_check.log) -eq 0 ]
do
        sleep 1s
        echo "Waiting for Gradio..."
done
echo "( tweaks ) Gradio is running"


# 启动 Phoenix
screen -S listen -d -m bash -c '
script -f /$project_name/c_l_c/phoenix_check.log -c "python3 -m phoenix.server.main --port 7007 serve"
'
sleep 3s
while [ $(grep -c "Uvicorn running on" /$project_name/c_l_c/phoenix_check.log) -eq 0 ]
do
        sleep 1s
        echo "Waiting for Phoneix..."
done
echo "( listen ) Phoneix is running"


# @ screen 常用指令
# 查看所有 screen：screen -ls
# 创建新的 screen：screen -S screen_name
# 回到对应 screen：screen -r screen_name
# 与 Attached 共享会话：screen -x screen_name
# 断连当前 screen：（ screen 内 ）Ctrl + a + d
# 终止当前 screen：（ screen 内 ）Ctrl + d
# 终止指向 screen：screen -X -S screen_name quit
# 清除所有已结束 screen：screen -wipe


