import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from init_load_model import store_vec, text_embedding


def text_image_retriever(query, q_client):
    """将查询文本向量化后，分别在文本和图片集合中做向量相似度检索，返回 top-5 结果。"""
    query = text_embedding.get_text_embedding(query)

    # Retrieve text hits
    text_hits = q_client.query_points(
        collection_name="texts",
        query=query,
        limit=5,
    ).points
    # Retrieve image hits
    Image_hits = q_client.query_points(
        collection_name="images",
        query=query,
        limit=5,
    ).points

    return text_hits, Image_hits

def retriever_result_display(text_hits, image_hits):
    """
    text_hits : list
        List of text results with `id`, `payload`, and `score`.
    image_hits : list
        List of image results with `id`, `payload`, and `score`.
    """

    print("\033[1m文本结果:\033[0m")
    for i, hit in enumerate(text_hits, 1):
        print("节点ID:",hit.id)
        content = hit.payload['content']
        content = f"\033[1m{content}"
        print(f"{i}. {content} | Score: {hit.score}")


    print("\n图片结果:")

    # plt.subplots：创建多子图画布，figsize 单位为英寸
    fig, axes = plt.subplots(1, len(image_hits), figsize=(15, 5))  # Adjust figsize as needed
    for ax, hit in zip(axes, image_hits):
        image_path = hit.payload['image_path']
        print(f"Displaying image: {image_path} | Score: {hit.score}")

        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Score: {hit.score}", fontsize=10)


    plt.suptitle("图片结果", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    text_hits, image_hits = text_image_retriever("2024年三季度GDP增长了多少？", store_vec.q_client)
    retriever_result_display(text_hits, image_hits)