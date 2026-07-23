"""
Script de ingesta de documentos.

Uso:
    python -m app.ingest

Este script:
  1. Lee todos los PDF y TXT de la carpeta ./docs
  2. Divide el texto en chunks con overlap
  3. Genera embeddings con HuggingFace
  4. Guarda todo en ChromaDB (persistente en disco)
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings

def load_documents(docs_path: str) -> list:
    documents = []
    docs_folder = Path(docs_path)

    if not docs_folder.exists():
        print(f"⚠️ La carpeta '{docs_path}' no existe. Créala y añade documentos.")
        return documents

    supported_files = list(docs_folder.glob("*.pdf")) + list(docs_folder.glob("*.txt"))
    if not supported_files:
        print(f"⚠️ No se encontraron archivos PDF o TXT en '{docs_path}'")
        return documents

    for file_path in supported_files:
        try:
            print(f"📄 Cargando: {file_path.name}")

            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue

            docs = loader.load()
            documents.extend(docs)
            print(f"   --> {len(docs)} página(s)/sección(es) cargadas")

        except Exception as e:
            print(f"   ❌ Error cargando {file_path.name}: {e}")

    print(f"\n📚 Total documentos cargados: {len(documents)}")
    return documents

def split_documents(documents: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Documentos divididos en {len(chunks)} chunks")
    print(f"   Tamaño por chunk: ~{settings.CHUNK_SIZE} caracteres")
    print(f"   Solapamiento: {settings.CHUNK_OVERLAP} caracteres")
    return chunks

def create_vector_store(chunks: list) -> Chroma:
    print(f"\n🧠 Generando embeddings con: {settings.EMBEDDING_MODEL}")
    print("   (Esto puede tardar unos minutos la primera vez...)")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )

    print(f"💾 Vector store creado en: {settings.CHROMA_DB_PATH}")
    print(f"   Colección: {settings.CHROMA_COLLECTION_NAME}")
    print(f"   Total de vectores almacenados: {len(chunks)}")

    return vector_store

def run_ingestion():
    print("=" * 60)
    print("🚀 INICIANDO PROCESO DE INGESTA DE DOCUMENTOS")
    print("=" * 60)

    settings.validate()

    print("\n📂 PASO 1: Cargando documentos...")
    documents = load_documents(settings.DOCS_PATH)

    if not documents:
        print("\n⚠️ No hay documentos para procesar. Añade archivos a la carpeta ./docs")
        sys.exit(1)

    print("\n✂️ PASO 2: Dividiendo en chunks...")
    chunks = split_documents(documents)

    print("\n💾 PASO 3: Creando vector store...")
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("✅ INGESTA COMPLETADA EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    run_ingestion()