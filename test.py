from transcriber import get_transcript, get_video_id
from rag import chunk_transcript, embed_chunks, store_chunks, query

url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

transcript = get_transcript(url)
chunks = chunk_transcript(transcript, 5)
embeddings = embed_chunks(chunks)
collection = store_chunks(chunks, embeddings, get_video_id(url))

answer, timestamps = query(collection, "what is this video about?", get_video_id(url))
print("Answer:", answer)
print("Timestamps:", timestamps)