import streamlit as st
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_toolkits.sql.base import create_sql_agent
from typing import Optional, List, Dict, Any, Sequence
import os
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.engine import reflection

# Initialize an empty chat history list
def get_chat_history():
    return []

chat_history = get_chat_history()

# Set OpenAI and SerpApi API keys
os.environ['OPENAI_API_KEY'] = "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"
os.environ["SERPAPI_API_KEY"] = "d7a3c3a53753df43a73197406264924597a7413f36edc7d02ae0cf8ecb789854"


# Initialize OpenAI model
llm = OpenAI(temperature=0)

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize the agent with AgentType.ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Connect to the database
db_user = "root"
db_password = "sampiksonu"
db_host = "localhost"
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")


# Create a ChatOpenAI model
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create an SQLDatabaseToolkit using the database
toolkit = SQLDatabaseToolkit(db=db, llm=chat_llm)

# Specify the dialect you want to use
dialect = "MySQL"  # Replace with your desired SQL dialect

# Initialize the agent executor
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix=(
        f"You are an agent designed to interact with a {dialect} SQL database.\n"
        "Given an input question, create a syntactically correct query to run,\n "
        "then look at the results of the query and return the answer.\n"
        "Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.\n"
        "You can order the results by a relevant column to return the most interesting examples in the database.\n"
        "Never query for all the columns from a specific table, only ask for the relevant columns given the question.\n"
        "You have access to tools for interacting with the database.\n"
        "Only use the below tools. Only use the information returned by the below tools to construct your final answer.\n"
        "You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n"
        "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n"
        "If the question does not seem related to the database, just return 'I don't know' as the answer."
    ),
    suffix=None,  # Optionally, provide a suffix
    format_instructions=(
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question"
    ),
    input_variables=None,  # Specify input variables if needed
    top_k=10,  # Limit the number of results
    max_iterations=15,  # Set the maximum number of iterations
    max_execution_time=None,  # Specify the maximum execution time if needed
    early_stopping_method="force",  # Choose an early stopping method
    agent_executor_kwargs=None,  # Add any additional keyword arguments
    handle_parsing_errors=True  # Specify how to handle parsing errors
)

st.set_page_config(
    page_title="Database Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Create the Streamlit interface with custom styling
st.title("📊 Database Chatbot: Query your database in natural language")
st.markdown("Welcome to the Database Chatbot. Ask any question about your database! 🚀")

# Sidebar content
st.sidebar.write("ℹ️ About")
st.sidebar.write("This is a chatbot powered by Langchain and Streamlit.")
st.sidebar.write("You can use it to query a SQL database using natural language.")
st.sidebar.markdown("[Streamlit](https://streamlit.io/)")
st.sidebar.markdown("[LangChain](https://python.langchain.com/)")
st.sidebar.markdown("[OpenAI LLM](https://platform.openai.com/docs/models)")
st.sidebar.write("Made by [Sampik Kumar Gupta](https://www.linkedin.com/in/sampik-gupta-41544bb7/)")

# User input and chat history
user_input = st.text_input("🗣️ Ask a question:")

def get_table_columns(engine, selected_table):
    insp = reflection.Inspector.from_engine(engine)
    columns = insp.get_columns(selected_table)
    return [column['name'] for column in columns]


if st.button("🚀 Submit"):
    if user_input:
        response = agent_executor.run(user_input)
        # Append user input and agent response to chat history
        chat_history.append({"user_input": user_input, "agent_response": response})


# Connect to the database
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    
# Get table names
insp = reflection.Inspector.from_engine(engine)
table_names = insp.get_table_names()
selected_table = st.selectbox("Select a Table", options=table_names, index=0)

# Get and display columns for the selected table
if selected_table:
    columns = get_table_columns(engine, selected_table)
    st.write(f"Columns for {selected_table}: {columns}")

# Display chat history
st.markdown("💬 Chat History:")
for chat in chat_history:
    st.write(f"👤 User: {chat['user_input']}")
    st.write(f"🤖 Chatbot: {chat['agent_response']}")

if __name__ == "__main__":
    pass


