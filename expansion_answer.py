from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os 
from openai import OpenAI
import chromadb
import umap  # using UMAP for dimensionality reduction to 2D/3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensures 3D projection is registered

# --- Splitting the document into chunks using LangChain utilities -------------
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

OPENAI_API = os.getenv("OPENAI_API")
llm_client = OpenAI(api_key=OPENAI_API)

# Instantiating a PDF reader on the Microsoft annual report,
# enabling sequential access to each page object for text extraction.
reader = PdfReader("data/microsoft-annual-report.pdf")

# Extracting textual content from each page.
# Using strip() to normalize leading and trailing whitespace.
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filtering out empty pages to retain only meaningful textual content.
pdf_texts = [text for text in pdf_texts if text]

# At this stage, pdf_texts is a list of strings, where each element corresponds
# to the extracted text of a page in the PDF.
"""
print(
    word_wrap(
        pdf_texts[0],
        width = 100,
    )
)
"""



# Defining a character-level recursive splitter.
# This performs a coarse segmentation: it applies a hierarchy of separators,
# starting with paragraphs ("\n\n"), then line breaks, sentence boundaries,
# words, and finally raw characters if no other split is possible.
# This ensures that chunks are cut at semantically meaningful boundaries whenever possible.
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)

# Concatenating pdf_texts (list of page strings) into a single long string
# using "\n\n".join(), then splitting this large document into overlapping chunks.
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts)) 

# At this stage, character_split_texts is a list of strings (chunks),
# each approximately 1000 characters in length.
# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks using character_split_texts: {len(character_split_texts)}")

# --- Token-level refinement ---------------------------------------------------
# Character-based splitting is useful for an initial coarse segmentation,
# but characters are not equivalent to tokens: depending on the language and
# the tokenizer, 1000 characters can correspond to very different token counts.
# To guarantee compatibility with LLMs (which operate on tokens, not characters),
# we apply a second-stage refinement using a token-based splitter.

# Defining a token-based splitter for fine-grained control.
# This ensures that each final chunk respects a strict token budget
# (256 tokens in this case), independent of the raw character length.
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256
)

# Applying the token splitter on top of the character-split chunks.
# This two-stage approach combines:
#   (1) structure-preserving coarse segmentation (characters)
#   (2) strict token-budget enforcement (tokens).
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# At this stage, token_split_texts contains the final set of chunks,
# each respecting both semantic coherence and token-length constraints.
# print("------------------------------------------")
# print(word_wrap(token_split_texts[10]))
# print("------------------------------------------\n\n")
# print(f"\nTotal chunks using character_split_texts: {len(character_split_texts)}")
# print(f"\nTotal chunks using token_split_texts: {len(token_split_texts)}")

# ------------------------------------------------------------------------------
# Up to this point, the pipeline performs the following steps:
# 1. Load the PDF and extract its textual content page by page.
# 2. Concatenate all pages into a continuous document string.
# 3. Apply a character-level splitter (based on spaces, newlines, and punctuation)
#    to produce coarse-grained chunks that preserve semantic boundaries.
# 4. Refine these chunks with a token-level splitter to enforce strict control
#    over the number of tokens per chunk.
#
# The token-based segmentation is essential because:
# - Tokens are the actual unit of computation for LLMs.
# - API usage and costs are billed per token, not per character.
# - The mapping between characters and tokens is language-dependent, so
#   character-based chunks may vary drastically in token length across languages.
#
# By applying both splitters in sequence, we ensure that:
# - Chunks remain semantically coherent (thanks to the character splitter).
# - Chunks strictly fit within a specified token budget (thanks to the token splitter).
# This makes the resulting text segments directly compatible with any LLM workflow.
# ------------------------------------------------------------------------------

# --- Creating and populating a Chroma collection ------------------------------

# Defining the embedding function to be used by Chroma for document ingestion.
# SentenceTransformerEmbeddingFunction applies a local sentence-transformers model
# (e.g., MiniLM), producing vector representations without external API calls.
embedding_function = SentenceTransformerEmbeddingFunction()

# Instantiating a local Chroma client.
# This client manages vector storage and similarity search operations on disk.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Creating a new collection named "microsoft-collection".
# The embedding_function is passed so that whenever documents are added,
# Chroma automatically computes embeddings for them at ingestion time.
chroma_collection = chroma_client.get_or_create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# ------------------------------------------------------------------------------

# Generating unique identifiers for each text chunk.
# For N chunks in token_split_texts, ids is simply ["0", "1", ..., "N-1"].
# This ensures that each chunk can be uniquely referenced inside the collection.
ids = [str(i) for i in range(len(token_split_texts))]

# Adding documents (chunks) to the Chroma collection.
# - ids: unique identifiers assigned to each chunk
# - documents: the raw text of each chunk
# With the embedding_function defined above, embeddings are computed automatically.
chroma_collection.add(ids=ids, documents=token_split_texts)

# Returning the total number of documents stored in the collection.
# This acts as a diagnostic check to confirm that ingestion was successful.
count = chroma_collection.count()
# print(count)

# Defining a natural language query intended for semantic retrieval.
# In this case, the query targets financial information about total yearly revenue.
query = "What was the total revenue for the year?"

# Executing a similarity search directly with the query text.
# Since the collection was initialized with an embedding_function,
# Chroma internally computes the query embedding and performs k-NN search.
results = chroma_collection.query(
    query_texts=[query],
    n_results=5
)

# Extracting the retrieved documents for the first (and only) query.
# In Chroma, 'results["documents"]' is a list of lists:
# - each inner list corresponds to the documents retrieved for one query in 'query_texts'.
# - this allows multiple queries to be processed in a single call,
#   returning separate sets of documents for each query.
# Since we only submitted a single query, [0] selects its corresponding results.
retrieved_documents = results["documents"][0]

# Iterating over each retrieved chunk, printing a wrapped preview
# to improve readability in the console.
# for document in retrieved_documents:
#    print("---------------------------------------------------------------------")
#    print(word_wrap(document))
#    print("---------------------------------------------------------------------")

def augment_query_generated(query, model="gpt-4.1-mini"):
    # Defining the system prompt that frames the model’s role.
    # In this case, the model is instructed to act as a financial research assistant
    # and generate plausible answers that one might expect in annual reports.
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""

    # Constructing the message sequence for the chat API.
    # - The system message sets the context and behavioral constraints.
    # - The user message carries the actual input query.
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    # Sending the prompt and query to the LLM via the chat completion endpoint.
    # The selected model defaults to "gpt-5-nano", but can be overridden.
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    # Extracting the assistant’s generated output.
    # The API returns a list of choices; here, the content of the first choice is selected.
    content = response.choices[0].message.content

    # Returning the model-generated example answer, which can be used
    # for downstream query expansion in the RAG pipeline.
    return content


# Defining the original user query, representing the information need.
original_query = "What was the total profit for the year, and how does it compare to the previous year?"

# Augmenting the query by generating a hypothetical answer with the LLM.
# This "pseudo-answer" simulates what might be found in the target documents.
hypothetical_answer = augment_query_generated(original_query)

# Concatenating the original query with the generated hypothetical answer.
# The two strings are joined with a single space, producing a richer query
# that combines both the question and a possible answer for retrieval.
joint_query = f"{original_query} {hypothetical_answer}"

# Pretty the combined query for readability.
# print(word_wrap(joint_query))

# Executing the semantic search in the Chroma collection.
# - query_texts: the augmented query used for retrieval
# - n_results: number of top matches to return
# - include: request both documents and embeddings for inspection
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)

# Extracting the retrieved document chunks corresponding to the first (and only) query.
retrieved_documents = results["documents"][0]

# -------------------------------------------------------------------------------------
# UMAP projection of collection, queries, and retrieved embeddings
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Extract embeddings for the entire collection
# -------------------------------------------------------------------------------------

# Loading all stored embeddings from the Chroma collection.
# These vectors represent the precomputed embeddings of the PDF chunks.
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]

# -------------------------------------------------------------------------------------
# Train a UMAP model for dimensionality reduction
# -------------------------------------------------------------------------------------

# Fitting a UMAP transformer on the full set of embeddings.
# UMAP reduces high-dimensional vectors (e.g., 384/768/1536 dims) into 2D/3D
# for visualization and qualitative analysis of neighborhood structure.
# 2D
# umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
# UMAP to 3D instead of 2D
umap_transform = umap.UMAP(n_components=3, random_state=0, transform_seed=0).fit(embeddings)
# -------------------------------------------------------------------------------------
# Project the entire dataset into a low-dimensional space
# -------------------------------------------------------------------------------------

# Applying the trained UMAP transformer to all collection embeddings.
# The result is a cloud of points in 2D/3D, ready to be plotted.
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# -------------------------------------------------------------------------------------
# Extract embeddings for the queries
# -------------------------------------------------------------------------------------

# Retrieving the embeddings of the top-k documents returned by the last query call.
retrieved_embeddings = results["embeddings"][0]

# Computing the embedding for the original user query.
original_query_embedding = embedding_function([original_query])

# Computing the embedding for the augmented (question + hypothetical answer) query.
augmented_query_embedding = embedding_function([joint_query])

# -------------------------------------------------------------------------------------
# Project queries and retrieved results in the same UMAP space
# -------------------------------------------------------------------------------------

# Projecting the original query embedding into the UMAP space to compare
# its position relative to the dataset and retrieved chunks.
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)

# Projecting the augmented query embedding; this enables a visual comparison
# of how the augmentation shifts the query in the embedding manifold.
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)

# Projecting the retrieved chunk embeddings to visualize their neighborhood
# relative to the original and augmented queries.
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# -------------------------------------------------------------------------------------
# Visualization 2D : Projected queries and retrieved documents in embedding space
# -------------------------------------------------------------------------------------

# Initializing a new figure for plotting the embedding space.
# plt.figure()

# -------------------------------------------------------------------------------------
# Plotting the full dataset embeddings (all PDF chunks)
# -------------------------------------------------------------------------------------
# Each point represents a chunk from the PDF projected into 2D space.
# Gray color is used to denote the background distribution of the dataset.
# plt.scatter(
#     projected_dataset_embeddings[:, 0],  # x-coordinates of all projected chunks
#     projected_dataset_embeddings[:, 1],  # y-coordinates of all projected chunks
#     s=10,                                # point size (small for background context)
#     color="gray",                        # neutral color to highlight other points
# )

# -------------------------------------------------------------------------------------
# Plotting the retrieved embeddings (top-k results from the query)
# -------------------------------------------------------------------------------------
# Green circles with no fill (only edges) are used to visually distinguish
# retrieved results from the full dataset.
# plt.scatter(
#     projected_retrieved_embeddings[:, 0],   # x-coordinates of retrieved chunks
#     projected_retrieved_embeddings[:, 1],   # y-coordinates of retrieved chunks
#     s=100,                                  # larger size for visibility
#     facecolors="none",                      # hollow markers
#     edgecolors="g",                         # green outline
# )

# -------------------------------------------------------------------------------------
# Plotting the original query embedding
# -------------------------------------------------------------------------------------
# Marked as a red "X" to emphasize its role as the starting point of retrieval.
# plt.scatter(
#     projected_original_query_embedding[:, 0],  # x-coordinate of original query
#     projected_original_query_embedding[:, 1],  # y-coordinate of original query
#     s=150,                                     # even larger size for salience
#     marker="X",                                # "X" marker for distinctiveness
#     color="r",                                 # red color for the original query
# )

# -------------------------------------------------------------------------------------
# Plotting the augmented query embedding
# -------------------------------------------------------------------------------------
# Marked as an orange "X" to visually compare how augmentation shifts the query
# in embedding space relative to the original query.
# plt.scatter(
#     projected_augmented_query_embedding[:, 0],  # x-coordinate of augmented query
#     projected_augmented_query_embedding[:, 1],  # y-coordinate of augmented query
#     s=150,                                      # large for clear comparison
#     marker="X",                                 # "X" marker to match original query
#     color="orange",                             # orange color for augmentation
# )

# -------------------------------------------------------------------------------------
# Figure formatting
# -------------------------------------------------------------------------------------

# Enforcing equal aspect ratio so distances in the scatter plot are meaningful.
# plt.gca().set_aspect("equal", "datalim")

# Using the original query text as the title for context.
# plt.title(f"{original_query}")

# Hiding axes for a cleaner visualization.
# plt.axis("off")

# Rendering the plot to the notebook or interactive window.
# plt.show()

# -------------------------------------------------------------------------------------
# Visualization (3D): Projected queries and retrieved documents in embedding space
# -------------------------------------------------------------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# -------------------------------------------------------------------------------------
# Plotting the full dataset embeddings (all PDF chunks) in 3D
# -------------------------------------------------------------------------------------
# Each point represents a chunk from the PDF projected into 3D space.
# ax.scatter(
#     projected_dataset_embeddings[:, 0],
#     projected_dataset_embeddings[:, 1],
#     projected_dataset_embeddings[:, 2],
#     s=10,
#     color="gray",
#     depthshade=False,  # keep points visually consistent
# )

# -------------------------------------------------------------------------------------
# Plotting the retrieved embeddings (top-k results from the query)
# -------------------------------------------------------------------------------------
# ax.scatter(
#     projected_retrieved_embeddings[:, 0],
#     projected_retrieved_embeddings[:, 1],
#     projected_retrieved_embeddings[:, 2],
#     s=100,
#     facecolors="none",
#     edgecolors="g",
#     depthshade=False,
# )

# -------------------------------------------------------------------------------------
# Plotting the original query embedding
# -------------------------------------------------------------------------------------
# ax.scatter(
#     projected_original_query_embedding[:, 0],
#     projected_original_query_embedding[:, 1],
#     projected_original_query_embedding[:, 2],
#     s=150,
#     marker="X",
#     color="r",
#     depthshade=False,
# )

# -------------------------------------------------------------------------------------
# Plotting the augmented query embedding
# -------------------------------------------------------------------------------------
# ax.scatter(
#     projected_augmented_query_embedding[:, 0],
#     projected_augmented_query_embedding[:, 1],
#     projected_augmented_query_embedding[:, 2],
#     s=150,
#     marker="X",
#     color="orange",
#     depthshade=False,
# )

# -------------------------------------------------------------------------------------
# Figure formatting (3D)
# -------------------------------------------------------------------------------------

# Set equal scaling on all axes so distances are meaningful in 3D.
# Note: box_aspect requires Matplotlib 3.3+
# ax.set_box_aspect((1, 1, 1))

# Title with the original query for context.
# ax.set_title(f"{original_query}")

# Optional: hide grid/axes if you want a cleaner look
# ax.set_axis_off()  # (3D axes do not support full axis-off like 2D)
# ax.grid(False)

# plt.show()



# Gray           → all chunks from the dataset (the full embedding space).
# Green (hollow) → chunks retrieved as relevant to the query.
# Red X          → the original query.
# Orange X       → the expanded query (original + hypothetical answer).



# -------------------------------------------------------------------------------------
# Final answer synthesis: pass original question + retrieved context to an LLM
# -------------------------------------------------------------------------------------

# Using a fresh client for generation, independent from the expansion step.
llm_client2 = OpenAI(api_key=OPENAI_API)

# -------------------------------------------------------------------------------------
# Utility: format retrieved context for the LLM
# -------------------------------------------------------------------------------------

def format_ranked_context(docs, max_chars=1200):
    """
    Formats retrieved documents as a ranked, numbered context block.

    Args:
        docs (List[str]): Retrieved chunks ordered from most to least relevant.
        max_chars (int): Soft cap to truncate overly long chunks for prompt hygiene.

    Returns:
        str: A human-readable, numbered context section (highest rank first).
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        snippet = (d[:max_chars]).strip()
        lines.append(f"[{i}] {snippet}")
    return "\n\n".join(lines)

# Build the context from the previously retrieved documents (ranked: best first).
ranked_context = format_ranked_context(retrieved_documents)

# -------------------------------------------------------------------------------------
# LLM call: concise, grounded answer using only the retrieved context
# -------------------------------------------------------------------------------------

# System message constrains behavior: concise, cite chunks, avoid unsupported claims.
system_msg = (
    "You are a concise financial research assistant. Answer ONLY using the provided context. "
    "If the answer is not present, say you don't know. When supporting statements, cite the "
    "relevant chunk indices like [1], [2]. Prioritize earlier chunks—they are ranked as more relevant."
)

# User message provides the question and the ranked context.
user_msg = (
    f"Question:\n{original_query}\n\n"
    f"Context (ranked, highest relevance first):\n{ranked_context}\n\n"
    "Instructions:\n"
    "- Use the context verbatim where possible.\n"
    "- Prefer earlier chunks if there is conflict.\n"
    "- Keep the answer brief and precise.\n"
)

# Execute the chat completion with a small, conservative temperature.
final_response = llm_client2.chat.completions.create(
    model="gpt-4.1-mini",            # choose a model that supports temperature controls
    temperature=0.25,                # low randomness for factual synthesis
    max_tokens=500,                  # concise final answer
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ],
)

# Extract and display the final, concise answer.
final_answer = final_response.choices[0].message.content
print("\n------------------------------------------------------- Final Answer -------------------------------------------------------")
print(f"Original question: {original_query}")
print(final_answer)
