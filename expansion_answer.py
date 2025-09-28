from helper_utils import word_wrap
from pypdf import PdfReader
import os 
from openai import OpenAI

# --- Splitting the document into chunks using LangChain utilities -------------
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

OPENAI_API = os.getenv("OPENAI_API")

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

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction