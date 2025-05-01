import os
import requests
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def save_research_notes(text: str) -> str:
    """Save research notes to a file."""
    try:
        # Create a notes directory if it doesn't exist
        if not os.path.exists("research_notes"):
            os.makedirs("research_notes")
        
        # Generate a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"research_notes/notes_{timestamp}.txt"
        
        # Write the notes to the file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        return f"Notes successfully saved to {filename}"
    except Exception as e:
        return f"Error saving notes: {e}"

def summarize_text(text: str) -> str:
    """Summarize the provided text using the LLM."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        # Use the same LLM to summarize the text
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that summarizes text concisely while preserving key information."),
            ("user", "Please summarize the following text in a concise manner, highlighting the most important points:\n\n{text}")
        ])
        
        # We'll initialize the LLM here to avoid circular imports
        summarizer = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        
        summary = summarizer.invoke(summary_prompt.format(text=text))
        return summary.content
    except Exception as e:
        return f"Error summarizing text: {e}"

# Initialize tools
tools = []

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
        tavily_tool = TavilySearchResults(
            tavily_api_key=TAVILY_API_KEY,
            max_results=5,  # Increased for more comprehensive research
            include_raw_content=True,  # Get full content when available
            include_domains=["scholar.google.com", "researchgate.net", "academia.edu", "arxiv.org"]  # Focus on academic sources
        )
        
        search_tool = Tool(
            name="WebSearch",
            description="Search the web for current information. Useful for finding recent research, news, or general information. Input should be a search query.",
            func=tavily_tool.invoke
        )
        tools.append(search_tool)
    except Exception as e:
        print(f"Error initializing Tavily search: {e}")

# Wikipedia tool
try:
    wiki = WikipediaAPIWrapper(top_k_results=3)  # Get more results for comprehensive research
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for retrieving background information and established knowledge from Wikipedia. Input should be a search query."
    )
    tools.append(wikipedia_tool)
except Exception as e:
    print(f"Error initializing Wikipedia tool: {e}")

# ArXiv tool for academic papers
try:
    arxiv_tool = ArxivQueryRun()
    arxiv_search = Tool(
        name="ArxivSearch",
        func=arxiv_tool.run,
        description="Search arXiv for scientific papers and research. Useful for finding academic papers on specific topics. Input should be a search query."
    )
    tools.append(arxiv_search)
except Exception as e:
    print(f"Error initializing ArXiv tool: {e}")

# PubMed tool for medical research
try:
    pubmed_tool = PubmedQueryRun()
    pubmed_search = Tool(
        name="PubMedSearch",
        func=pubmed_tool.run,
        description="Search PubMed for medical and biological research papers. Useful for finding medical studies and health information. Input should be a search query."
    )
    tools.append(pubmed_search)
except Exception as e:
    print(f"Error initializing PubMed tool: {e}")

# Google Scholar tool using scholarly library
from langchain.tools import Tool

def google_scholar_search(query: str) -> str:
    try:
        from scholarly import scholarly
        results = []
        search_query = scholarly.search_pubs(query)
        for i in range(5):
            try:
                pub = next(search_query)
                pub_info = f"Title: {pub['bib'].get('title', 'No title')}\n"
                pub_info += f"Authors: {pub['bib'].get('author', 'Unknown')}\n"
                pub_info += f"Year: {pub['bib'].get('pub_year', 'Unknown')}\n"
                pub_info += f"Venue: {pub['bib'].get('venue', 'Unknown')}\n"
                pub_info += f"Citations: {pub.get('num_citations', 0)}\n"
                pub_info += f"Abstract: {pub['bib'].get('abstract', 'No abstract available')[:200]}...\n"
                results.append(pub_info)
            except StopIteration:
                break
            except Exception as e:
                results.append(f"Error retrieving publication: {str(e)}")
        
        return "\n\n".join(results) if results else "No results found on Google Scholar for this query."
    except ImportError:
        return "The scholarly library is not installed. Please install it using: pip install scholarly"
    except Exception as e:
        return f"Error searching Google Scholar: {str(e)}"

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
    
    # Function to conduct research
    def conduct_research(query: str) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a given query.
        
        Args:
            query: The research question or topic
            
        Returns:
            Dictionary containing the research results and process
        """
        # Add a research prefix to help guide the agent
        research_query = f"Research question: {query}\n\nPlease conduct thorough research on this topic, using multiple sources and providing a comprehensive answer with citations."
        
        # Run the agent
        result = agent_executor.invoke({"input": research_query})
        
        # Format the output
        formatted_result = {
            "query": query,
            "answer": result["output"],
            "sources": [],
            "research_process": []
        }
        
        # Extract sources and research process from intermediate steps
        if hasattr(agent_executor, "intermediate_steps"):
            for step in agent_executor.intermediate_steps:
                if len(step) >= 2:
                    tool = step[0].tool
                    tool_input = step[0].tool_input
                    tool_output = step[1]
                    
                    # Add to research process
                    formatted_result["research_process"].append({
                        "tool": tool,
                        "query": tool_input,
                        "result_summary": summarize_text(str(tool_output)) if len(str(tool_output)) > 500 else str(tool_output)
                    })
                    
                    # Add to sources if it's a search tool
                    if tool in ["WebSearch", "ArxivSearch", "PubMedSearch", "GoogleScholarSearch", "Wikipedia"]:
                        formatted_result["sources"].append({
                            "source_type": tool,
                            "query": tool_input
                        })
        
        return formatted_result
    
    # Function to save research results
    def save_research_results(results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save research results to a JSON file."""
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"research_results_{timestamp}.json"
        
        # Create research_results directory if it doesn't exist
        if not os.path.exists("research_results"):
            os.makedirs("research_results")
        
        filepath = f"research_results/{filename}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return f"Research results saved to {filepath}"

    # Example usage
    if __name__ == "__main__":
        print("🔍 Research Agent Initialized 🔍")
        print("Available tools:", ", ".join([tool.name for tool in tools]))
        
        # Example research queries
        research_queries = [
            "What are the latest advancements in quantum computing?",
            "How does climate change affect marine ecosystems?",
            "What is the current consensus on the benefits and risks of artificial intelligence?"
        ]
        
        # Ask user to select a query or enter their own
        print("\nExample research topics:")
        for i, query in enumerate(research_queries):
            print(f"{i+1}. {query}")
        print("4. Enter your own research topic")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == "4":
            research_topic = input("\nEnter your research topic: ")
        elif choice in ["1", "2", "3"]:
            research_topic = research_queries[int(choice)-1]
        else:
            research_topic = "What are the latest advancements in quantum computing?"
        
        print(f"\n🔍 Researching: {research_topic}")
        print("This may take a few minutes for comprehensive research...\n")
        
        # Conduct research
        results = conduct_research(research_topic)
        
        # Display results
        print("\n📊 Research Results 📊")
        print(f"Query: {results['query']}")
        print("\nAnswer:")
        print(results['answer'])
        
        print("\nSources used:")
        for source in results['sources']:
            print(f"- {source['source_type']}: {source['query']}")
        
        # Save results
        save_path = save_research_results(results)
        print(f"\nResearch results saved to {save_path}")
        
except Exception as e:
    print(f"Error initializing Research Agent: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease ensure required packages are installed:")
    print("pip install -U langchain langchain-google-genai langchain-community google-generativeai requests python-dotenv wikipedia arxiv pypubmed scholarly")