import os
import re
import fitz
import faiss
import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import matplotlib.pyplot as plt
import jieba.analyse
import networkx as nx
from itertools import combinations
from collections import Counter

PDF_FOLDER = "data"  # 你的多PDF目录
INDEX_PATH = "bd/index.faiss"
SENTENCES_PATH = "bd/sentences_with_meta.pkl"

client = OpenAI(
    api_key="sk-14abb7efe7b541d4b7e3e871460b583b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# === 提取单个 PDF 的句子（带文件名和页码）===
def extract_sentences_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    result = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split("\n")

        clean_lines = []
        for line in lines:
            line = line.strip()

            # === ✂️ 忽略以下噪声类内容 ===
            if not line or len(line) < 4:
                continue  # 太短跳过
            if re.match(r"^\d{1,3}$", line):  # 页码
                continue
            if any(keyword in line for keyword in [
                "联系我们", "服务号", "扫码关注", "扫描二维码", "版权所有", "Thoughtworks", "洞见",
                "目录", "目录页", "微信", "邮箱", "电话", "公众号"
            ]):
                continue
            clean_lines.append(line)

        merged_text = "".join(clean_lines)

        # === 按中英文标点分句 ===
        sentence_endings = r'([。！？?!.])'
        parts = re.split(sentence_endings, merged_text)
        sentences = [parts[i] + parts[i+1] if i+1 < len(parts) else parts[i]
                     for i in range(0, len(parts), 2)]

        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 5:
                result.append((sent, page_num + 1, filename))
    return result

# === 批量提取所有 PDF ===
def extract_all_pdfs(folder):
    all_sentences = []
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            print(f"📄 正在处理: {fname}")
            full_path = os.path.join(folder, fname)
            all_sentences.extend(extract_sentences_from_pdf(full_path))
    return all_sentences

# === 构建统一向量数据库 ===
def build_faiss_index(sentences_with_meta):
    sentences = [s for s, _, _ in sentences_with_meta]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(sentences, show_progress_bar=True)
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(SENTENCES_PATH, "wb") as f:
        pickle.dump(sentences_with_meta, f)

    return index, sentences_with_meta

# === 载入索引和句子信息 ===
def load_index():
    print("📂 正在加载向量数据库...")
    index = faiss.read_index(INDEX_PATH)
    with open(SENTENCES_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

# === 构建图（可选）===
def build_sentence_graph(sentences_with_meta, window=1):
    G = nx.DiGraph()
    for i, (sent, page, source) in enumerate(sentences_with_meta):
        G.add_node(i, text=sent, page=page, source=source)
        for j in range(1, window + 1):
            if i + j < len(sentences_with_meta):
                G.add_edge(i, i + j)
    return G

# === 检索相似句索引 + 扩展上下文 ===
def search_and_expand(query, index, meta, graph=None, top_k=5, expand_k=1):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    _, indices = index.search(q_vec, top_k)

    matched = indices[0]
    if graph:
        expanded = set(matched)
        for idx in matched:
            expanded.update(list(graph.successors(idx))[:expand_k])
            expanded.update(list(graph.predecessors(idx))[:expand_k])
        return sorted(expanded)
    else:
        return matched

# === 模型回答生成 ===
def generate_answer(query, context):
    if not context.strip():
        prompt = "无可用数据，禁止使用外部知识，请回答“无法回答”。"
    else:
        prompt = (
            "以下是多个PDF文档中的相关片段，请仅基于这些内容作答，"
            "不得使用常识或预训练知识：\n" + context
        )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个文档问答助手，只能基于上下文内容作答。"},
            {"role": "user", "content": f"{prompt}\n\n问题：{query}"}
        ]
    )
    return completion.choices[0].message.content

# === 知识问答入口 ===
def knowledge_qa(query, index, meta, graph=None):
    print("\n🔍 正在检索...")
    match_ids = search_and_expand(query, index, meta, graph)
    if not match_ids:
        print("❌ 未找到相关内容")
        context = ""
    else:
        context_lines = []
        print("✅ 检索到以下片段：")
        for idx in match_ids:
            sent, page, source = meta[idx]
            print(f"[{source} - 第{page}页]: {sent}")
            context_lines.append(f"[{source} 第{page}页] {sent}")
        context = "\n".join(context_lines)
    answer = generate_answer(query, context)
    print("\n💬 回答：", answer)
    
def visualize_sentence_graph(graph, save_path="bd/sentence_graph.png"):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.figure(figsize=(18, 18))  # 可以调整尺寸
    pos = nx.spring_layout(graph, k=0.2)  # 自动布局
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='skyblue')
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='->', width=1.0)
    nx.draw_networkx_labels(graph, pos, font_size=6, font_color='black')

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📍 句间连接图已保存至: {save_path}")
    plt.close()



# === 主函数 ===
def main():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(SENTENCES_PATH):
        print("🚧 初次运行：从多个PDF中构建向量数据库...")
        all_data = extract_all_pdfs(PDF_FOLDER)
        index, meta = build_faiss_index(all_data)
    else:
        index, meta = load_index()

    graph = build_sentence_graph(meta)
    
    visualize_sentence_graph(graph)
    
    while True:
        query = input("\n❓ 请输入问题（或输入 '退出' 结束）：")
        if query.strip().lower() == "退出":
            print("👋 退出问答系统")
            break
        knowledge_qa(query, index, meta, graph)

# === 启动 ===
if __name__ == "__main__":
    main()