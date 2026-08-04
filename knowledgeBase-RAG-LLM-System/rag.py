from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
# ===================== 改动 1：换用 OpenAI 兼容接口 =====================
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# ===================== 删掉下面的 import =====================
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService(object):
    def __init__(self):

        # ===================== 改动 2：用 OpenAIEmbeddings 指向阿里云 =====================
        self.vector_service = VectorStoreService(
            embedding=OpenAIEmbeddings(
                model=config.embedding_model_name,
                openai_api_key=config.dashscope_api_key,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        # ===================== 改动 3：用 ChatOpenAI 指向阿里云 =====================
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            openai_api_key=config.dashscope_api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "我之前问了什么"}, session_config)
    print(res)
