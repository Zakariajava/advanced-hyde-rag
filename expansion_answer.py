from helper_utils import word_wrap
from pypdf import PdfReader
import os 
from openai import OpenAI
import chromadb

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

def augment_query_generated(query, model="gpt-5-nano"):
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
