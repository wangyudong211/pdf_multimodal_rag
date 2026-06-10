from openai import OpenAI  # openai：调用兼容 OpenAI API 格式的 LLM 服务（支持第三方部署的模型）
import base64

from init_load_model import store_vec
from retriever_embedding_from_local import text_image_retriever

# llm部分需要调用第三方的或者自己部署的多模态模型

key = '第三方key'
# OpenAI 客户端支持自定义 base_url，可对接任何兼容 OpenAI API 格式的第三方服务
llm_client = OpenAI(api_key=key, base_url="http://maas-api.cn-huabei-1.xf-yun.com/v1")
print(llm_client.models.list())

def text_image_rag(context: list, images: list, query: str):

    generation_prompt = f"""
    根据给定的上下文，必须回答用户的提问，上下文可以是表格、文本或图像。根据上下文给出答案。用户提问是: {query}
    上下文是: {context}\n
    输出:
    """

    def encode_image(image_path):
        """将图片文件读取并编码为 base64 字符串，用于通过 JSON 传输二进制图片数据。"""
        if image_path:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        return None


    image_paths = images
    # messages 是多轮对话的消息列表，role 区分系统指令（system）、用户输入（user）和模型回复（assistant）
    messages = [
        {
            "role": "system",
            "content": "你是一个PDF内容理解助手。"
        },
        {
            "role": "user",
            "content": generation_prompt,
        }
    ]

    # 多模态输入：将图片以 base64 data URL 格式附加到 messages 中
    for image_path in image_paths:
        img_base64 = encode_image(image_path)
        if img_base64:
            messages.append({
                "role": "user",
                "content": [
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img_base64}"  # data URL 格式：直接内嵌图片数据
                      },
                  },
              ],
            })

    chat_response = llm_client.chat.completions.create(
        model="xdeepseekr1",
        messages=messages,
        temperature=0.5,   # temperature：控制输出的随机性，0=确定性输出，1=最随机，RAG 场景建议 0.3~0.7
        top_p=0.99
    )

    return chat_response.choices[0].message.content

def rag(query):
    """RAG 主流程：检索相关文本和图片 → 拼接上下文 → 调用 LLM 生成答案。"""
    text_hits, image_hits = text_image_retriever(query, store_vec.q_client)
    retrieved_images = [i.payload['image_path'] for i in image_hits]  # 列表推导式取出图片路径
    answer = text_image_rag(text_hits, retrieved_images, query)

    for hit in text_hits:
        content = hit.payload['content']
        print(f"内容: {content} | 分数: {hit.score}")

    for hit in image_hits:
        image_path = hit.payload['image_path']
        print(f"图片: {image_path} | 分数: {hit.score}")

    return answer, text_hits, image_hits

if __name__ == '__main__':
    query = '2024年第三季度GDP增长了多少？'
    answer,_,_ = rag(query)  # _ 是 Python 惯例，表示忽略不需要的返回值
    print('llm 的回答是以下内容：')
    print(answer)