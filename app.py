import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.docstore.document import Document
from transformers import pipeline

def load_docs(folder="data"):
    docs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith((".txt", ".md")):
                p = os.path.join(root, fn)
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append(Document(page_content=f.read(), metadata={"source": p}))
    return docs

def build_or_load_index(docs, index_dir=".faiss"):
    if os.path.exists(index_dir):
        return FAISS.load_local(
            index_dir,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    vs.save_local(index_dir)
    return vs

def answer(question, vs, k=4):
    retr = vs.as_retriever(search_kwargs={"k": k})
    ctx_docs = retr.invoke(question)   # ✅ new style
    ctx_text = "\n\n".join(
        [f"[Source {i+1}] {d.metadata.get('source','?')}\n{d.page_content}" for i, d in enumerate(ctx_docs)]
    )

    system_prompt = "You are a helpful assistant. Use the provided context to answer. Cite sources as [1], [2]."
    user_prompt = f"Question: {question}\n\nContext:\n{ctx_text}"

    # ✅ Upgrade model from flan-t5-small → flan-t5-base
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=generator)

    resp = llm.invoke(system_prompt + "\n\n" + user_prompt)  # ✅ use .invoke
    return resp

if __name__ == "__main__":
    docs = load_docs()
    vs = build_or_load_index(docs)
    print("Type your question (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("\n" + answer(q, vs) + "\n")
