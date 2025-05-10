from chatbot import graph, log_message

try:
    question = "Tell me about Dorothy"
    log_message("Chatbot", "Chatbot started")
    log_message("Chatbot", f"Question to chatbot:\n\t{question}")
    response = graph.invoke({"question": question})
    log_message("Chatbot", f"Response from Chatbot:\n\t{response["answer"]}")
except Exception as e:
    log_message("Error", f"Exception: {e}")

with open("logs.txt", "a") as f:
    f.write("\n----------------------------------------------\n\n")