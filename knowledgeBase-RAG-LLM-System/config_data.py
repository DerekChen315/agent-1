import os
from dotenv import load_dotenv

# ===================== 加载环境变量 =====================
load_dotenv()

# ===================== API Key 配置 =====================
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope_api_key:
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
max_spliter_char_number = 1000

# ===================== 检索配置 =====================
similarity_threshold = 1

# ===================== 模型名称配置 =====================
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# ===================== Session 配置 =====================
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
