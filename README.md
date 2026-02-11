```
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

# Future work

## 1. Ingestion

### General

- Add logging
- Tidy chunking code
- Batch & multi-threadind/multi-processing for extraction, processing, uploads to qdrant

### Improve doc processing function

- Combine author names
- Group all text into sections
- List flattening: collapse list groups into one chunk
- Store title in metadata
- Semantic merging: merge adjacent chunks if cosine similarity > threshold
- Table routing: send tables to a separate table-RAG pipeline
- Picture routing
- KG seeding: extract entities per chunk and attach entity_ids

### 2. Retrieval

- Make retriever async - see `async_retriever.py` in archive. Note, don't need async for ingestion as that is a CPU/GPU bound task and not I/O bound.
- Add metadata extraction (use glinear 2 model to extract - source file, temporal, tags, doc name)

### 3. Generator

- The prompt returns the source e.g. Source 1 for the reranked ordered nodes. This should instead be the chunk index which comes from the metadata.
