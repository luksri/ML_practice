from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
import os
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import phi
from phi.model.groq import Groq

load_dotenv()


phi.api = os.getenv("phi_api_key")


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


app = Playground(agents=[websearch_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)