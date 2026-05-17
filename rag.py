
from groq import Groq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
load_dotenv()
def chunk_transcript(transcript, chunk_size):
    chunks = []
    for i in range(0, len(transcript), chunk_size):
        snippet_chunks = transcript[i: i + chunk_size]
        text = "\n".join([snippet.text for snippet in snippet_chunks])
        start_time = snippet_chunks[0].start
        chunks.append((text, start_time))
    return chunks
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
def embed_chunks(chunks): 
    texts = []
    for chunk in chunks: 
        texts.append(chunk[0])
    embeddings = model.encode(texts)
    return embeddings
def store_chunks(chunks, embeddings, video_id=None):
    client = chromadb.Client()
    collection = client.get_or_create_collection("video_transcripts")
    in_collection = collection.get(where= {"video_id": video_id})
    if in_collection["ids"]:
        print(f"Video with id {video_id} was already added.")
        return collection
    ids = [f"{video_id}_{i}" for i in range(0, len(chunks))]
    texts = [chunk[0] for chunk in chunks]
    metadatas = [{"start_time" : chunk[1], "video_id" : video_id} for chunk in chunks]
    collection.add(ids=ids, documents=texts, embeddings= embeddings.tolist(), metadatas=metadatas)
    return collection
def query(collection, question, video_id):
    question_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings= [question_embedding], n_results = 3, where= {"video_id" : video_id} if video_id else None)
    best_context = "\n".join(results["documents"][0])
    timestamps_of_context = [meta["start_time"] for meta in results["metadatas"][0]]

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a  assistant that answers questions based on video transcripts. Only answer from the context."},
            {"role": "user", "content": f"Context:\n{best_context}\n\nQuestion: {question}"}
        ]
    )
    answer = response.choices[0].message.content
    return (answer, timestamps_of_context)
def summarize(collection, video_id):
    all_chunks = collection.get(where={"video_id": video_id})
    full_transcript = "\n\n".join(all_chunks["documents"])
    
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a assistant that summarizes video transcripts concisely."},
            {"role": "user", "content": f"Please summarize this video transcript in bullet points covering the main topics:\n\n{full_transcript}"}
        ],
        temperature=0.3,
        max_completion_tokens=1024
    )
    return response.choices[0].message.content
