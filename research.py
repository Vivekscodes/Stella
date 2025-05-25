```python
import os
import time
import json
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Import required LangChain components
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# Import community tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# Get API keys from environment variables
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")


def save_research_notes(text: str) -> str:
    """Saves research notes to a file."""
    notes_directory: str = "research_notes"  # Define the directory for saving notes

    # Create the directory if it doesn't exist
    os.makedirs(notes_directory, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp: str = time.strftime("%Y%m%d-%H%M%S")
    filename: str = os.path.join(notes_directory, f"notes_{timestamp}.txt")

    # Write the notes to the file
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        return f"Notes successfully saved to {filename}"
    except Exception as e:
        return f"Error saving notes: {e}"


def summarize_text(text: str) -> str:
    """Summarizes the provided text using the LLM."""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        # Define the prompt for summarization
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that summarizes text concisely while preserving key information."),
            ("user",
             "Please summarize the following text in a concise manner, highlighting the most important points:\n\n{text}")
        ])

        # Initialize the LLM here to avoid circular imports
        summarizer = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        # Get the summary from the LLM
        summary_response = summarizer.invoke(summary_prompt.format(text=text))
        return summary_response.content
    except Exception as e:
        return f"Error summarizing text: {e}"


# Initialize an empty list to hold the tools
tools: List[Tool] = []

# Define tools with their names, functions, and descriptions

# Note-taking tool
notes_tool = Tool(
    name="SaveNotes",
    func=save_research_notes,
    description="Save research notes to a file for later reference. Input should be the text to save."
)
tools.append(notes_tool)

# Summarization tool
summarize_tool = Tool(
    name="Summarize",
    func=summarize_text,
    description="Summarize a long piece of text to extract the key points. Input should be the text to summarize."
)
tools.append(summarize_tool)

# Tavily search tool
if TAVILY_API_KEY:
    try:
        tavily_search = TavilySearchResults(
            tavily_api_key=TAVILY_API_KEY,
            max_results=5,  # Increased for more comprehensive research
            include_raw_content=True,  # Get full content when available
            include_domains=["scholar.google.com", "researchgate.net", "academia.edu", "arxiv.org"]
            # Focus on academic sources
        )

        web_search_tool = Tool(
            name="WebSearch",
            description="Search the web for current information. Useful for finding recent research, news, or general information. Input should be a search query.",
            func=tavily_search.invoke
        )
        tools.append(web_search_tool)
    except Exception as e:
        print(f"Error initializing Tavily search: {e}")

# Wikipedia tool
try:
    wikipedia = WikipediaAPIWrapper(top_k_results=3)  # Get more results for comprehensive research
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for retrieving background information and established knowledge from Wikipedia. Input should be a search query."
    )
    tools.append(wikipedia_tool)
except Exception as e:
    print(f"Error initializing Wikipedia tool: {e}")

# ArXiv tool for academic papers
try:
    arxiv = ArxivQueryRun()
    arxiv_tool = Tool(
        name="ArxivSearch",
        func=arxiv.run,
        description="Search arXiv for scientific papers and research. Useful for finding academic papers on specific topics. Input should be a search query."
    )
    tools.append(arxiv_tool)
except Exception as e:
    print(f"Error initializing ArXiv tool: {e}")

# PubMed tool for medical research
try:
    pubmed = PubmedQueryRun()
    pubmed_tool = Tool(
        name="PubMedSearch",
        func=pubmed.run,
        description="Search PubMed for medical and biological research papers. Useful for finding medical studies and health information. Input should be a search query."
    )
    tools.append(pubmed_tool)
except Exception as e:
    print(f"Error initializing PubMed tool: {e}")


def google_scholar_search(query: str) -> str:
    """Searches Google Scholar for academic papers."""
    try:
        from scholarly import scholarly

        search_results: List[str] = []  # Initialize an empty list for results
        search_query = scholarly.search_pubs(query)

        for _ in range(5):  # Limiting search to the first 5 results
            try:
                publication = next(search_query)
                publication_info = f"Title: {publication['bib'].get('title', 'No title')}\n"
                publication_info += f"Authors: {publication['bib'].get('author', 'Unknown')}\n"
                publication_info += f"Year: {publication['bib'].get('pub_year', 'Unknown')}\n"
                publication_info += f"Venue: {publication['bib'].get('venue', 'Unknown')}\n"
                publication_info += f"Citations: {publication.get('num_citations', 0)}\n"
                publication_info += f"Abstract: {publication['bib'].get('abstract', 'No abstract available')[:200]}...\n"
                search_results.append(publication_info)
            except StopIteration:
                break
            except Exception as e:
                search_results.append(f"Error retrieving publication: {str(e)}")

        return "\n\n".join(search_results) if search_results else "No results found on Google Scholar for this query."
    except ImportError:
        return "The scholarly library is not installed. Please install it using: pip install scholarly"
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}"


# Google Scholar tool using scholarly library
scholar_tool = Tool(
    name="GoogleScholarSearch",
    func=google_scholar_search,
    description="Search Google Scholar for academic papers. Input should be a search query."
)
tools.append(scholar_tool)

# Initialize memory with a larger buffer for research context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a research-focused system message
research_system_message = """You are an advanced Research Assistant AI specialized in conducting comprehensive research on any topic.

Your capabilities:
1. Search the web for current information
2. Access Wikipedia for background knowledge
3. Search academic databases like ArXiv and PubMed for scientific papers
4. Take and save research notes
5. Summarize long texts to extract key information

When conducting research:
- Be thorough and explore multiple sources
- Evaluate the credibility of sources
- Cite your sources clearly
- Organize information logically
- Identify gaps in the research
- Suggest areas for further investigation

For complex research topics, break down your approach into steps:
1. Understand the research question
2. Gather background information
3. Search for specific details from specialized sources
4. Synthesize the information
5. Provide a comprehensive answer with citations

Always maintain academic integrity and present balanced viewpoints.
"""

# Initialize LLM with higher temperature for more creative research synthesis
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using the latest model for best research capabilities
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2  # Slightly higher temperature for more creative synthesis while maintaining accuracy
    )

    # Use initialize_agent with ZERO_SHOT_REACT_DESCRIPTION instead of create_structured_chat_agent
    # This avoids the agent_scratchpad issue
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # Allow more iterations for thorough research
        agent_kwargs={
            "prefix": research_system_message
        }
    )


    def conduct_research(query: str) -> Dict[str, Any]:
        """
        Conducts comprehensive research on a given query.

        Args:
            query: The research question or topic

        Returns:
            Dictionary containing the research results and process
        """
        # Add a research prefix to help guide the agent
        research_prompt = f"Research question: {query}\n\nPlease conduct thorough research on this topic, using multiple sources and providing a comprehensive answer with citations."

        # Run the agent
        result = agent_executor.invoke({"input": research_prompt})

        # Format the output
        formatted_result: Dict[str, Any] = {
            "query": query,
            "answer": result["output"],
            "sources": [],
            "research_process": []
        }

        # Extract sources and research process from intermediate steps
        intermediate_steps = agent_executor.intermediate_steps
        for step in intermediate_steps:
            if len(step) >= 2:
                tool_name = step[0].tool
                tool_input = step[0].tool_input
                tool_output = step[1]

                # Add to research process
                formatted_result["research_process"].append({
                    "tool": tool_name,
                    "query": tool_input,
                    "result_summary": summarize_text(str(tool_output)) if len(str(tool_output)) > 500 else str(
                        tool_output)
                })

                # Add to sources if it's a search tool
                if tool_name in ["WebSearch", "ArxivSearch", "PubMedSearch", "GoogleScholarSearch", "Wikipedia"]:
                    formatted_result["sources"].append({
                        "source_type": tool_name,
                        "query": tool_input
                    })

        return formatted_result


    # Function to save research results
    def save_research_results(results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Saves research results to a JSON file."""

        # Generate a filename if one isn't provided
        if not filename:
            timestamp: str = time.strftime("%Y%m%d-%H%M%S")
            filename: str = f"research_results_{timestamp}.json"

        # Define the directory for saving results
        results_directory: str = "research_results"

        # Create research_results directory if it doesn't exist
        os.makedirs(results_directory, exist_ok=True)

        filepath: str = os.path.join(results_directory, filename)

        # Save the research results to the file
        try:
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)

            return f"Research results saved to {filepath}"
        except Exception as e:
            return f"Error saving research results: {e}"


    # Example usage
    if __name__ == "__main__":
        print("🔍 Research Agent Initialized 🔍")
        print("Available tools:", ", ".join([tool.name for tool in tools]))

        # Example research queries
        example_queries: List[str] = [
            "What are the latest advancements in quantum computing?",
            "How does climate change affect marine ecosystems?",
            "What is the current consensus on the benefits and risks of artificial intelligence?"
        ]

        # Ask user to select a query or enter their own
        print("\nExample research topics:")
        for i, query in enumerate(example_queries):
            print(f"{i + 1}. {query}")
        print("4. Enter your own research topic")

        user_choice: str = input("\nSelect an option (1-4): ")

        if user_choice == "4":
            research_topic: str = input("\nEnter your research topic: ")
        elif user_choice in ["1", "2", "3"]:
            research_topic: str = example_queries[int(user_choice) - 1]
        else:
            research_topic: str = "What are the latest advancements in quantum computing?"

        print(f"\n🔍 Researching: {research_topic}")
        print("This may take a few minutes for comprehensive research...\n")

        # Conduct research
        research_results: Dict[str, Any] = conduct_research(research_topic)

        # Display results
        print("\n📊 Research Results 📊")
        print(f"Query: {research_results['query']}")
        print("\nAnswer:")
        print(research_results['answer'])

        print("\nSources used:")
        for source in research_results['sources']:
            print(f"- {source['source_type']}: {source['query']}")

        # Save results
        save_path: str = save_research_results(research_results)
        print(f"\nResearch results saved to {save_path}")

except Exception as e:
    print(f"Error initializing Research Agent: {e}")
    import traceback

    traceback.print_exc()
    print("\nPlease ensure required packages are installed:")
    print(
        "pip install -U langchain langchain-google-genai langchain-community google-generativeai requests python-dotenv wikipedia arxiv pypubmed scholarly")
```