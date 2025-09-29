# Advanced HyDE-RAG

This project implements an **advanced Retrieval-Augmented Generation (RAG)** pipeline using the **HyDE technique (Hypothetical Document Embeddings)**.  
Instead of relying solely on the original user query, HyDE augments retrieval by generating a **hypothetical answer** with an LLM and using it to improve similarity search in the vector database. This often leads to retrieving more relevant passages, especially when the user's query is vague or underspecified.

The project was inspired by and adapted from the concepts shown in this video:  
👉 [YouTube: Building Advanced RAG Pipelines](https://www.youtube.com/watch?v=ea2W8IogX80)

---

## 🔑 Key Features

- **PDF ingestion pipeline**  
  - Extracts text page by page using `pypdf`.  
  - Cleans and concatenates pages into a continuous document.  
  - Splits text with a two-stage strategy:  
    - `RecursiveCharacterTextSplitter` for coarse, structure-aware segmentation.  
    - `SentenceTransformersTokenTextSplitter` for strict token-level control.  

- **Vector database with Chroma**  
  - Uses a local persistent ChromaDB instance (`PersistentClient`).  
  - Embeddings generated with `SentenceTransformerEmbeddingFunction` (MiniLM).  
  - Automatic embedding computation on ingestion and query.  

- **HyDE query expansion**  
  - Original query is passed to the LLM to generate a **hypothetical answer**.  
  - The original query and the generated answer are concatenated to form a richer search query.  
  - This joint query significantly improves retrieval quality by anchoring it in a possible answer space.  

- **Query & retrieval visualization**  
  - Dimensionality reduction with UMAP to 2D and 3D.  
  - Side-by-side visualization of:  
    - **Gray** → all dataset chunks.  
    - **Green (hollow)** → top retrieved chunks.  
    - **Red X** → original query.  
    - **Orange X** → augmented query (HyDE).  
  - Plots saved as `2d_umap_query_vs_augmented_retrieval.png` and `3d_umap_query_vs_augmented_retrieval.png`.  

- **Final answer generation**  
  - Retrieved passages are passed along with the original query to the LLM.  
  - Produces a concise, grounded answer.  
  - Example output is stored in `final_anwser.txt`.  

---

## 📂 Repository Structure

```
advanced-hyde-rag/
│
├── data/                           # Input documents (e.g., Microsoft annual report)
├── chroma_db/                      # Persistent Chroma vector store
├── helper_utils.py                 # Utility functions (PDF extraction, word wrap, embedding projection)
├── hyde_query_expansion.py         # Main pipeline implementing HyDE-RAG
├── advanced-Rag.ipynb              # Notebook version for experimentation
├── 2d_umap_query_vs_augmented_retrieval.png  # 2D embedding visualization
├── 3d_umap_query_vs_augmented_retrieval.png  # 3D embedding visualization
├── final_anwser.txt                # Example output from the final LLM stage
└── readme.md                       # Project documentation
```

---

## 🚀 How It Works

1. **Document preprocessing**  
   Load, clean, and chunk the PDF into semantically coherent, token-constrained segments.

2. **Indexing**  
   Store the chunks in a Chroma collection with automatic embeddings.

3. **HyDE query expansion**  
   - Generate a hypothetical answer to the original query with an LLM.  
   - Concatenate original query + hypothetical answer → joint query.  
   - Retrieve top-k most relevant chunks with this enriched query.

4. **Visualization**  
   Project embeddings to 2D/3D with UMAP to compare: dataset distribution, original query, augmented query, and retrieved chunks.

5. **Answer generation**  
   Send the original query and retrieved chunks to the LLM to produce a final grounded answer.

---

## ⚙️ Requirements

- Python 3.9+  
- Conda or venv recommended  
- Install dependencies:  

```bash
pip install -r requirements.txt
```

---

## 📖 References

- HyDE: Precise Zero-Shot Dense Retrieval using Hypothetical Document Embeddings
- YouTube Tutorial (inspiration)
- LangChain & ChromaDB documentation

---

## 📝 Notes

- Tokens are the real computation and billing unit for LLMs → the token-based splitter ensures chunks stay within predictable bounds.
- HyDE works especially well for abstract queries where the answer might not directly contain the user's keywords.
- Current pipeline runs locally with SentenceTransformers embeddings, but can be easily extended to use text-embedding-3-small or other OpenAI embedding models.