from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# for web search
websearch_agent = Agent(
    name="web search agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# for financial search
financial_agent = Agent(
    name="Finance AI agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    team=[websearch_agent, financial_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["always include sources", "use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("summarize analyst recommendataion and share the latest news for sbin", stream=True, )