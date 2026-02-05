```
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

# Future work

## 1. Ingestion

### General

- Batch ingest data
- Remove llama index boilerplate
- Add logging

### Improve doc processing function

- Combine author names
- Group all text into sections
- List flattening: collapse list groups into one chunk
- Store title in metadata
- Semantic merging: merge adjacent chunks if cosine similarity > threshold
- Table routing: send tables to a separate table-RAG pipeline
- Picture routing
- KG seeding: extract entities per chunk and attach entity_ids
