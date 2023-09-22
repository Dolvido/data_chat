import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import YoutubeLoader
from langchain.agents import load_tools
from langchain.agents.load_tools import get_all_tool_names
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import textwrap

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})

# --------------------------------------------------------------
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
template = """Observe: {conv_history}"""

prompt = PromptTemplate(template=template, input_variables=["conv_history"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# --------------------------------------------------------------
# Load a video transcript from YouTube
# --------------------------------------------------------------

video_url = "https://www.youtube.com/watch?v=kgCUn4fQTsc"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_documents(transcript)

# --------------------------------------------------------------
# Summarization with LangChain
# --------------------------------------------------------------

# Add map_prompt and combine_prompt to the chain for custom summarization
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
#print(chain.llm_chain.prompt.template)
#print(chain.combine_document_chain.llm_chain.prompt.template)

# --------------------------------------------------------------
# Test the Falcon model with text summarization
# --------------------------------------------------------------

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)


# -------------------------------------------------------------
# prepare to save conversation hsitory
# --------------------------------------------------------------
# Ensure the "conv_data" directory exists
if not os.path.exists('conv_data'):
    os.makedirs('conv_data')

# Create a unique filename using a timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filepath = os.path.join('conv_data', f'conversation_{timestamp}.txt')

# List to store conversation data
conversation_data = []

# Counter to track the number of iterations
counter = 0
response_chain = ""
while counter < 10:
    print("counter: ", counter)
    
    # The first prediction step
    response = llm_chain.run(output_summary)
    
    # Wrap and print the response text
    wrapped_text = textwrap.fill(response, width=100, break_long_words=False, replace_whitespace=False)
    print(wrapped_text)
    conversation_data.append(f"Counter {counter} - LLM Chain Response:\n{wrapped_text}\n")

    # Write the conversation data to a file within the loop
    try:
        with open(filepath, 'a') as file:  # Note that we open the file in 'a' (append) mode
            file.write(f"Counter {counter} - LLM Chain Response:\n{conversation_data[-1]}\n")
    except Exception as e:
        print(f"Could not write to file: {e}")

    # Set the output of the response chain as the new output summary for the next iteration
    output_summary = response_chain  # Uncomment this line to enable the feedback loop
    # Increment the counter
    counter += 1








