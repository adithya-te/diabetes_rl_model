import os
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Update this to your actual model location
TINY_LLAMA_PATH = r"C:\Users\adiad\OneDrive\Desktop\diabetes_rl_project\models\tinyllama.gguf"

# Load the LLM
llm = LlamaCpp(
    model_path=TINY_LLAMA_PATH,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=False,
    n_threads=8,
)

# Memory to retain conversation
memory = ConversationBufferMemory()

# Build conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Response logic
def chatbot_response(user_input, context=None):
    return conversation.predict(input=user_input)

# Clear memory logic
def clear_memory():
    global memory
    memory.clear()
