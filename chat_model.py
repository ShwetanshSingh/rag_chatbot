from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


class LoadModel:
    def load_llm_model(model_repo_id="meta-llama/Meta-Llama-3-8B"):
        # Instantiation of chat model
        llm = HuggingFaceEndpoint(
            repo_id=model_repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03
        )

        chat_model = ChatHuggingFace(llm=llm)
        print("Chat model loaded successfully")             # Debug message
        return chat_model

    # Invocation
    # from langchain_core.messages import(
    #     HumanMessage,
    #     SystemMessage
    # )
    # messages = [
    #     SystemMessage(content="You're a helpful assistant"),
    #     HumanMessage(
    #         content="I have a question"
    #     )
    # ]
    # ai_reply = chat_model.invoke(messages)

    def load_embeddings_model():
        # Instantiation of embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("Embeddings model loaded successfully")       # Debug message
        return embeddings

class LoadVectorStore:
    def load_vector_store(embeddings):
        # Instantiation of vector store
        vector_store = InMemoryVectorStore(embeddings)
        print("Vector store loaded successfully")           # Debug message
        return vector_store