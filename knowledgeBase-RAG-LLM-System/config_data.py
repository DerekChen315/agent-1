import os
from dotenv import load_dotenv

# ===================== 加载环境变量 =====================
# 如果本地有 .env 文件，加载它（Streamlit Cloud 上通过 Secrets 注入）
load_dotenv()

# ===================== API Key 配置 =====================
# 优先从环境变量读取，如果没有则使用默认值（仅用于本地测试）
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key:
    # 如果环境变量没有，可以在这里直接填写（仅本地测试用，千万不要提交到 GitHub！）
    # dashscope_api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # 取消注释并填入你的 Key
    raise ValueError(
        "❌ DASHSCOPE_API_KEY 未设置！\n"
        "请在 Streamlit Cloud 的 Settings -> Secrets 中添加，\n"
        "或在本地项目根目录创建 .env 文件并写入 DASHSCOPE_API_KEY=你的Key"
    )

# ===================== 文件路径配置 =====================
md5_path = "./md5.text"

# ===================== Chroma 向量数据库配置 =====================
collection_name = "rag"
persist_directory = "./chroma_db"

# ===================== 文本分割器配置 =====================
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_spliter_char_number = 1000  # 文本分割阈值

# ===================== 检索配置 =====================
similarity_threshold = 1  # 检索返回匹配的文档数量

# ===================== 模型名称配置 =====================
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# ===================== Session 配置 =====================
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
