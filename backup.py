from fastapi import Depends, FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel, Field
import pandas as pd
import os
import httpx
import shutil
from shapely.geometry import Point, Polygon
import io
import sqlite3
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.csv_toolkit import CsvTools
from tavily import TavilyClient
import ast
from typing import List
from typing import Optional, Dict, Any
from collections import Counter
import json
from typing import Union
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from textwrap import dedent
import re
from datetime import datetime
from apify_client import ApifyClient
import pdfplumber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions




app = FastAPI()

tavily_client = TavilyClient(api_key='')
os.environ["OPENAI_API_KEY"]=""
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")

CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
if not CLERK_SECRET_KEY:
    raise RuntimeError("Missing CLERK_SECRET_KEY environment variable.")
sdk = Clerk(bearer_auth=CLERK_SECRET_KEY)

# This should be your frontend's URL
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "https://officialagt.netlify.app")


# --- 2. Create the Authentication Dependency ---
def require_user(request: Request):
    try:
        httpx_req = httpx.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
        )
        state = sdk.authenticate_request(
            httpx_req,
            AuthenticateRequestOptions(authorized_parties=["https://officialagt.netlify.app","http://localhost:3000", "http://127.0.0.1:3000"])
        )
        if not state.is_signed_in:
            # --- THIS IS THE KEY CHANGE ---
            # Log the specific reason for the failure to your terminal
            logger.warning(f"Authentication failed: {state.reason}")
            raise HTTPException(status_code=401, detail=state.reason or "Unauthorized")
        return state
    except Exception as e:
        logger.error(f"An unexpected error occurred during authentication: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")

def get_user_id_from_state(state):
    """Extracts the user ID from the verified token payload."""
    claims = getattr(state, "payload", {}) or {}
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user ID in token")
    return user_id


# --- Database Initialization ---
DB_NAME = "user_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_chats (
            user_id TEXT PRIMARY KEY,
            chat_history TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Call this when your app starts up
init_db()

# --- Dynamic Date Injection ---
# 1. Get the current date and time
current_date = datetime.now()
# 2. Format it into a clear, human-readable string
date_instruction = f"The current date is {current_date.strftime('%A, %B %d, %Y')}. All your answers must be relevant to this date."
APIFY_RESULTS = {} 


# --- Define a static path for the CSV data ---
CSV_DATA_PATH = "data/sales_data.csv"

# Initialize CsvTools to query the sales data directly
csv_tools = CsvTools(csvs=[CSV_DATA_PATH])



CONCURRENT_SEARCHES = 8
_semaphore = asyncio.Semaphore(CONCURRENT_SEARCHES)

async def _web_search_one(q: str) -> Dict[str, Any]:
    """Helper function to run a single search within our concurrency limit."""
    async with _semaphore:
        try:
            # Runs the synchronous tavily_client in a separate thread so it doesn't block.
            resp = await asyncio.to_thread(
                tavily_client.search,
                query=q,
                search_depth="advanced"
            )
            return {"query": q, "results": resp.get("results", [])}
        except Exception as e:
            return {"query": q, "error": str(e), "results": []}

async def batch_web_search(queries: List[str]) -> Dict[str, Any]:
    """
    Performs multiple web searches concurrently for a list of queries. Use this
    when multiple pieces of information are needed to answer a complex question.
    """
    # Create a list of tasks, one for each unique query.
    unique_queries = list(dict.fromkeys([q.strip() for q in queries if q and q.strip()]))
    tasks = [asyncio.create_task(_web_search_one(q)) for q in unique_queries]

    # Wait for all the search tasks to complete.
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return {"queries": unique_queries, "batches": results}

def web_search(query: str) -> str:
    """
    Performs a web search to find real-time information, news, events, local regulations, 
    historical facts, or general knowledge about a specific location. Use this when the
    provided context does not contain the answer.
    """
    print(f"--- 🔎 Performing web search for: {query} ---")
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        return json.dumps(response.get("results", "No results found from web search."))
    except Exception as e:
        print(f"--- ❌ Error during Tavily web search: {e} ---")
        return "Failed to perform web search."
    
def extract_text_from_pdf(file_path: str) -> str:
    """
    Opens a PDF file and extracts all text content from its pages.

    Args:
        file_path: The path to the PDF file (e.g., "abc.pdf").

    Returns:
        A single string containing all the text from the PDF, or an empty string if
        the file is not found or cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"--- ⚠️ Warning: PDF file not found at {file_path} ---")
        return ""

    try:
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n" # Add space between pages
        
        print(f"--- ✅ Successfully extracted text from {file_path} ---")
        return full_text.strip()
    except Exception as e:
        print(f"--- ❌ Error reading PDF file {file_path}: {e} ---")
        return ""



db = SqliteDb(db_file="tmp/agent.db", memory_table="geo_agent_memories_v2")

livability_agent = Agent(
    model=OpenAIChat(id="o4-mini"),
    tools=[web_search],
    instructions=[
        "You are a world-class urban planning and livability analyst specializing in hyperlocal analysis.",
        "Your task is to analyze a given specific locality or neighborhood based on four key indexes: Economic Opportunity, Health and Wellness, Education and Culture, and Infrastructure and Safety.",
        "Use your web search tool to find relevant data for the provided location. Focus your search on the specific locality mentioned (e.g., 'Jubilee Hills' within 'Jubilee Hills, Hyderabad') and avoid city-wide generalizations.",
        "For each of the four indexes, you must provide a score from 1 to 10 and a brief, one-sentence summary of your findings.",
        "You MUST return the final output as a single, valid JSON object. Do not include any conversational text or markdown.",
        "The JSON structure must be: {\"economic\": {\"score\": <int>, \"summary\": \"<string>\"}, \"health\": {\"score\": <int>, \"summary\": \"<string>\"}, \"education\": {\"score\": <int>, \"summary\": \"<string>\"}, \"infrastructure\": {\"score\": <int>, \"summary\": \"<string>\"}}",
    ]
)


# --- Agent #1: POI Data Extraction Agent ---
summarize_agent = Agent(
    # A fast and efficient model is good for this structured task
    model=OpenAIChat(id="gpt-5-mini"), 
    
    # This agent doesn't need external tools
    tools=[], 
    
    markdown=True,
    
    # NEW instructions for extracting each POI
    instructions=[
        "You are a meticulous Data Extraction Agent.",
        "Your sole purpose is to parse a raw JSON data string containing a list of Points of Interest (POIs) and create a structured summary for EACH individual POI.",
        "Iterate through every POI object in the provided JSON data.",
        "For each POI, you must extract its name, category (e.g., 'Restaurant', 'Cafe'), and star rating.",
        "Format the final output as a bulleted list using markdown. Each bullet point must represent one POI.",
        "The format for each line MUST be: '- **[Name of Place]**: [Category] - [Rating] stars.'",
        "If a rating is not available for a POI, use the phrase 'Not rated'.",
        "Do NOT include any introductory or concluding sentences. Your response should begin directly with the first bullet point and end with the last one.",
        "Give the answer in no more than 1000 words while covering all the places of interests"
    ],
    
    # Memory is not needed for this stateless, one-off task
    db=None, 
)



# --- Agent# 2: Geoinsight agent ---
geo_agent = Agent(
    model=OpenAIChat(id="o4-mini"),  # GPT-5 supports parallel_tool_calls by default
    tools=[batch_web_search, web_search],
    markdown=True,
    instructions=[
        date_instruction,
        "You are GeoInsight, an expert geospatial analyst and researcher.",
        # ✅ --- NEW/REVISED LANGUAGE INSTRUCTIONS (Add near the top) ---
        "VERY IMPORTANT: First, detect the language of the user's query.",
        "You MUST generate your entire final response EXCLUSIVELY in the same language as the user's query.",
        "If the web search tool returns results in English, you MUST synthesize and present that information in the detected query language.",
        "Do NOT mix languages in your final response.",
        
        # --- CRITICAL: Tool Usage Strategy ---
        "TOOL USAGE RULES:",
        "1. When you need information from MULTIPLE distinct topics or locations, you MUST use `batch_web_search` with ALL queries at once.",
        "2. Only use `web_search` for a SINGLE follow-up query if batch_web_search results need clarification.",
        "3. NEVER make sequential web_search calls when you can batch them.",
        
        # --- Analysis Requirements ---
        "Your goal is to provide comprehensive, data-driven answers by synthesizing local context with real-time web search results.",
        "First, check your memory for relevant information.",
        "Next, analyze the user's question and the provided 'Local Context' data.",
        
        # --- Formatting (keep your existing formatting rules) ---
        "You MUST format your entire response using rich Markdown.",
        "Structure your answer using headings (`##` for main sections, `###` for subsections).",
        "Use bold text (`**text**`) to highlight key terms and data points.",
        "Use numbered lists for items like competitors, amenities, or data points.",
        "Add blank lines between paragraphs for visual spacing.",
        "Use a horizontal rule (`---`) to separate analysis from the '## Sources' section.",
        
        # --- Final Requirements ---
        "Provide decisive, data-driven answers supported by numbers.",
        "Always include sources at the end.",
        "Be conversational in human language.",
    ],
    db=db,
    user_id="default_user",
    debug_mode=True,
)



# --- NEW: Agent #3 - The Traffic Command Agent ---
# This agent's only job is to return a JSON command when it detects a traffic query.
traffic_agent = Agent(
    model=OpenAIChat(id="o3-mini"),
    role="A specialist for handling all user requests about traffic, road congestion, chokepoints, or travel times.",
    instructions=[
        "You are a traffic system controller.",
        "When a user asks about traffic, your ONLY job is to respond with a specific JSON command object and nothing else.",
        "You MUST return the following JSON structure exactly: ",
        # This is the command the frontend will receive
        '{"type": "command", "content": "Activating the traffic layer for the selected area.", "action": {"name": "TOGGLE_TRAFFIC_LAYER"}}'
    ],
    tools=[], # No tools needed for this agent
)


class GeoAnalysisOutput(BaseModel):
    analysis_content: str = Field(description="The full geospatial analysis in markdown foramt only")
    places: List[str] = Field(description="Array of name of POI places in the analysis")

main_team = Team(
    name="Geospatial Routing Team",
    model=OpenAIChat("gpt-5"),
    members=[
        geo_agent,
        traffic_agent
    ],
    output_schema=GeoAnalysisOutput,
    respond_directly=True,
    determine_input_for_members=True,
    instructions=[
        "You are a master router. Your primary job is to determine which specialist agent is best suited for the user's query and delegate the task to them.",
        "Routing Rules:",
        "- For traffic-related queries only and only if user asks `show me traffic visuals`, route to the `traffic_agent`.",
        "- For all other geospatial or analytical queries, route to the `geo_agent`.",
        "Always include the complete user query and any provided context when delegating."
    ],
    db=db,
    cache_session=True,
    add_history_to_context=True,
    debug_mode=True,
    num_history_runs=2,
    share_member_interactions=True,
)


# --- MODIFIED: This function now sends a formatted string ---
async def extract_locality_from_places(places: list) -> Optional[str]:
    """
    Uses a dedicated Agno agent to extract a common locality name from a formatted string of place information.
    """
    if not places:
        return None

    # Create the formatted string as requested.
    # No need for json.dumps() as this is already a string.
    places_info_string = "\n".join([
        f"- {place.get('name', 'N/A')}: {place.get('address', 'N/A')}, "
        f"Type: {place.get('type', 'N/A')}, "
        f"Rating: {place.get('rating', 'Not rated')}"
        for place in places
    ])

    try:
        # Use the specialized agent for the extraction task
        result = locality_extractor_agent.run(input=places_info_string)
        
        # Use the correct .output attribute and clean the response
        locality = result.content.strip().replace("'", "").replace('"', '')
        print(f"---📍 Extracted Locality: {locality} ---")
        return locality
    except Exception as e:
        print(f"--- ❌ Error extracting locality: {e} ---")
        return "Selected Area"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Our CSV has a column 'Map_Details' with lat/long in the format:
# {'LATITUDE': '17.575957', 'LONGITUDE': '78.471167'}
csv_file_path = "hyderabad.csv"
csv_file_path_gur = "gurgaon_10k.csv"

monitoring_data = pd.read_csv(csv_file_path)
monitoring_data_gur = pd.read_csv(csv_file_path_gur)
gurgaon_data = pd.read_csv("gurugram_data.csv")

class MessageRequest(BaseModel):
    message: str

class Coordinates(BaseModel):
    lat: float
    lng: float   

class MapRequest(BaseModel):
    polygonCoordinates: list

class Location(BaseModel):
    lat: float
    lng: float

class PlaceData(BaseModel):
    name: str
    address: str
    location: Location
    rating: Optional[float] = Field(None)
    type: str

class PlacesRequest(BaseModel):
    places: List[PlaceData]
    search_area_address: Optional[str] = None
    weather_forecast: Optional[Dict[str, Any]] = None # Accepts the full weather JSON object
    POI: Optional[int] = 0 # NEW: Add optional POI field
    coordinates: Optional[Coordinates] = None





class ExtractPlacesRequest(BaseModel):
    message: str

class ComparisonArea(BaseModel):
    places: List[PlaceData]
    search_area_address: Optional[str] = None
    weather_forecast: Optional[Dict[str, Any]] = None

class ComparisonRequest(BaseModel):
    comparison_areas: List[ComparisonArea]


class AreaRequest(BaseModel):
    area_sq_ft: float


class LivabilityRequest(BaseModel):
    location_name: str

class ChatHistoryRequest(BaseModel):
    chat_history: List[Dict[str, Any]]




current_places = []
average_rent = ""
current_search_address = "an unspecified area"
weather_report = None
current_team_dependencies = {}

def run_apify_scraper(coordinates: Coordinates):
    """Runs the Apify Google Maps Scraper using custom geolocation."""
    if not coordinates:
        return None
    try:
        client = ApifyClient("")
        
        # NEW: Use the run_input with customGeolocation
        run_input = {
            "customGeolocation": {
                "type": "Point",
                # NOTE: The order is [longitude, latitude] and they must be strings
                "coordinates": [
                    str(coordinates.lng),
                    str(coordinates.lat)
                ],
                "radiusKm": 0.5
            },
            "includeWebResults": False,
            "language": "en",
            "maxCrawledPlacesPerSearch": 12,
            "maxImages": 0,
            "maximumLeadsEnrichmentRecords": 0,
            "placeMinimumStars": "three",
            "scrapeContacts": False,
            "scrapeDirectories": False,
            "scrapeImageAuthors": False,
            "scrapePlaceDetailPage": False,
            "scrapeReviewsPersonalData": True,
            "scrapeTableReservationProvider": False,
            "searchStringsArray": [
                "restaurant",
                "cafe"
            ],
            "skipClosedPlaces": False
        }
        
        print(f"Starting Apify scraper for coordinates: {coordinates.lat}, {coordinates.lng}")
        run = client.actor("nwua9Gu5YrADL7ZDj").call(run_input=run_input)
        
        results = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
        print(f"Apify scraper finished, found {len(results)} items.")
        print(results)
        return results
    except Exception as e:
        print(f"An error occurred during Apify scraping: {e}")
        return None


def parse_markdown_to_story(markdown_text):
    """
    Parses a markdown string and converts it into a list of ReportLab Platypus objects (a 'story').
    This version uses regex for robust bold text handling.
    """
    styles = getSampleStyleSheet()
    story = []
    
    lines = markdown_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle headings
        if line.startswith('### '):
            story.append(Paragraph(line.lstrip('### '), styles['h3']))
        elif line.startswith('## '):
            story.append(Paragraph(line.lstrip('## '), styles['h2']))
        elif line.startswith('# '):
            story.append(Paragraph(line.lstrip('# '), styles['h1']))
            
        # Handle bullet points
        elif line.startswith('- '):
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line) # Handle bold text inside lists
            story.append(Paragraph(f"• {line.lstrip('- ')}", styles['BodyText']))
            
        # Handle paragraphs with bold text
        else:
            # Use regex to find all occurrences of **text** and replace them with <b>text</b>
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(line, styles['BodyText']))
            
        story.append(Spacer(1, 6))

    return story


@app.post("/process-internal-data")
async def process_internal_data(file: UploadFile = File(...)):
    """
    Receives a client's CSV file and saves it to the static path for the CsvTools to query.
    """
    os.makedirs(os.path.dirname(CSV_DATA_PATH), exist_ok=True)
    
    try:
        # Save the uploaded file to the predefined path, overwriting any existing file.
        with open(CSV_DATA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"--- ✅ Data from {file.filename} saved to {CSV_DATA_PATH} and is ready for querying. ---")
        
        return {"status": "success", "message": f"Data from {file.filename} has been processed and is ready for analysis."}

    except Exception as e:
        print(f"--- ❌ Error saving CSV file: {e} ---")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded file.")
    

@app.post("/generate-report")
async def generate_report(locations: str = Form(...)):
    """
    Generates a comprehensive retail analysis report for the given locations
    and returns it as a downloadable PDF.
    """
    try:
        selected_locations = json.loads(locations)
        if not selected_locations:
            raise HTTPException(status_code=400, detail="No locations provided.")

        sales_analyst = Agent(
            model=OpenAIChat(id="o4-mini"),
            name="Sales Analyst",
            tools=[csv_tools],
            markdown=True,
            instructions=[
                "You are a sales data analyst. Your goal is to analyze internal sales data from a provided CSV file.",
                "Use the `query_csv_file` tool to execute your analysis.",
                # Add these more specific instructions:
                "**IMPORTANT**: When writing queries, use simple, standard SQL syntax.",
                "Always enclose column names in double quotes (\") to avoid syntax errors with reserved keywords.",
                "For example, query for `\"Product Category\"` instead of `Product Category`.",
                "Focus on querying for top-selling items, underperforming items, and sales trends for the given locations."
            ]
        )

        market_researcher = Agent(
            name="Market Researcher",
            model=OpenAIChat(id="o3-mini"),
            tools=[web_search],
            markdown=True,
            exponential_backoff=True, delay_between_retries=2,
            instructions=dedent("""
                You are a hyperlocal competitive intelligence analyst. For Cafe Delhi Heights at each specified location, your task is to find and summarize the most critical external factors impacting it's retail business.

                **IMPORTANT**: You MUST perform separate, concise web searches using web_search tool for each of the 5 areas below. Do NOT combine them into a single search. Keep each search query under 300 characters.

                Your analysis must cover these 5 areas:
                1.  **Competitor Activity**: Identify the top 2-3 direct competitors.
                2.  **Local Events**: Find 1-2 major upcoming local events.
                3.  **Weather Impact**: Find the weather forecast for the upcoming week.
                4.  **Competitor Reviews**: Summarize recent online review sentiment for one key competitor.
                5.  **Search Demand**: Briefly describe local search trends for relevant business terms.

                Return the output as a brief, structured summary for each location.
            """)
        )

        # --- Define the Coordinator Team ---
        retail_analysis_team = Team(
            name="Retail Analysis Team",
            mode="coordinate",
            model=OpenAIChat("gpt-5"),  # Use a powerful model for coordination
            markdown=True,
            members=[sales_analyst, market_researcher],
            instructions=[
                "You are the lead retail strategist. Your goal is to create a comprehensive analysis report for the specified locations.",
                "First, task the Sales Analyst to retrieve internal sales data for each location.",
                "Next, task the Market Researcher to gather external hyperlocal context (competitors, events, demographics) for each location.",
                "Finally, synthesize the findings from both agents into a single, cohesive report.",
                "Provide actionable recommendations based on the combined insights.",
                "The final output should be a well-structured markdown text."
            ]
        )

        # --- Run the Team to Generate the Report Content ---
        report_prompt = f"Generate a comprehensive retail analysis report for the following locations: {json.dumps(selected_locations)}"
        print(f"--- 🚀 Kicking off retail analysis team for: {selected_locations} ---")
        result = retail_analysis_team.run(message=report_prompt)
        report_markdown = result.content 

        # --- Generate PDF ---
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        story = [
            Paragraph("AGTMap Enterprise Retail Analysis", getSampleStyleSheet()['Title']),
            Spacer(1, 12),
            Paragraph(f"Analysis for Locations: {json.dumps(selected_locations)}", getSampleStyleSheet()['h2']),
            Spacer(1, 24),
        ]
        
        parsed_story = parse_markdown_to_story(report_markdown)
        story.extend(parsed_story)
        
        doc.build(story)
        
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': 'attachment; filename="Retail_Analysis_Report.pdf"'}
        )

    except Exception as e:
        print(f"--- ❌ Error generating report: {e} ---")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the report: {e}")

# --- NEW: Endpoint for Livability Analysis ---
@app.post("/livability-analysis")
async def get_livability_analysis(request: LivabilityRequest):
    if not request.location_name or "Unknown" in request.location_name:
        raise HTTPException(status_code=400, detail="A valid location name is required.")
    
    try:
        # The agent's prompt is the location name itself
        result = livability_agent.run(input=request.location_name)
        # The agent is configured to return JSON, so we parse it directly
        livability_data = json.loads(result.content)
        return livability_data
    except Exception as e:
        print(f"--- ❌ Error during livability analysis: {e} ---")
        raise HTTPException(status_code=500, detail="Failed to generate livability analysis.")



# --- UPDATED ENDPOINT: Returns the most recent data for each station ---
@app.get("/groundwater-data")
async def get_gurgaon_data():
    """
    Processes the groundwater data to return only the most recent entry
    for each unique monitoring station.
    """
    if gurgaon_data.empty:
        raise HTTPException(status_code=404, detail="Gurugram data not found.")

    # Check for required columns for filtering
    if 'Data Acquisition Time' not in gurgaon_data.columns or 'Station' not in gurgaon_data.columns:
        # Fallback: If columns for sorting don't exist, return the raw data.
        print("Warning: 'Data Acquisition Time' or 'Station' column not found. Returning unfiltered data.")
        return gurgaon_data.to_dict(orient='records')

    # Create a working copy to avoid modifying the global dataframe
    df = gurgaon_data.copy()

    # Convert 'Data Acquisition Time' to datetime objects for accurate sorting.
    # Errors will be converted to NaT (Not a Time), which can be handled.
    df['Data Acquisition Time'] = pd.to_datetime(df['Data Acquisition Time'], errors='coerce')

    # Drop rows where date conversion failed
    df.dropna(subset=['Data Acquisition Time'], inplace=True)

    # Sort by station and then by time (descending) to get the latest on top for each group
    df.sort_values(by=['Station', 'Data Acquisition Time'], ascending=[True, False], inplace=True)

    # Keep only the first (i.e., the most recent) entry for each station
    latest_data = df.drop_duplicates(subset='Station', keep='first')
    
    print(f"Returning {len(latest_data)} unique, most recent station records.")
    
    return latest_data.to_dict(orient='records')


@app.post("/update-area")
async def update_area(request: AreaRequest):
    """Receives and processes the calculated area of a polygon."""
    area_sq_ft = request.area_sq_ft
    print(f"--- 🏞️ Area received: {area_sq_ft} sq ft ---")
    
    # You can now use this data, for example, by adding it to the agent's context
    area_context_message = f"The area of the user's selected polygon is {area_sq_ft} square feet."
    

    main_team.dependencies["calculated_area"] = area_context_message

    
    return {"status": "success", "message": f"Area of {area_sq_ft} sq ft received."}


@app.post("/update-places")
async def update_places(request: PlacesRequest):
    global current_places
    global current_search_address
    global weather_report
    session_id = "current_user"

    current_places = [place.dict() for place in request.places]
    messages_for_run = []

    if request.search_area_address:
        current_search_address = request.search_area_address
        print(f"--- 📍 Search area updated to: {current_search_address} ---")

    if request.weather_forecast:
        weather_report = request.weather_forecast
        print(f"--- 📍 Weather report for the area: {weather_report} ---")

    # If the POI flag is set, run the scraper with the coordinates
    if request.POI == 1 and request.coordinates:
        scraped_data = run_apify_scraper(request.coordinates) # UPDATED
        if scraped_data:
            APIFY_RESULTS[session_id] = json.dumps(scraped_data, indent=2)

    print(weather_report)

    current_dependencies = {}

    if current_places:
        places_info = "\n".join([
            f"- Name: {place['name']}, Type: {place['type']}, Rating: {place.get('rating', 'N/A')}"
            for place in current_places
        ])

        area_intro = f"The user is analyzing the '{current_search_address}' area."
        local_context_message = f"{area_intro}\nHere are some nearby places:\n{places_info}"
        current_dependencies["places_of_interest"] = local_context_message
    
    global current_team_dependencies
    current_team_dependencies = current_dependencies

    return {"status": "success", "count": len(current_places)}



@app.post("/update-comparison-data")
async def update_comparison_data(request: ComparisonRequest):
    """
    Receives data for multiple areas to be compared and sets the context for the agent.
    """
    if not request.comparison_areas:
        return {"status": "error", "message": "No comparison data provided."}

    comparison_context_parts = []
    for i, area in enumerate(request.comparison_areas):
        # Create a summary for each area
        area_intro = f"--- Data for Comparison Area {i+1}: '{area.search_area_address}' ---"
        
        # Summarize places
        places_summary = "No places found."
        if area.places:
            place_counts = Counter(p.type for p in area.places)
            top_3_places = place_counts.most_common(3)


        # Combine the summary for the area
        area_context = f"{area_intro}"
        comparison_context_parts.append(area_context)
    
    # Create the final context string for the agent
    full_context_string = (
            "You are an expert real estate investment analyst. Your goal is to compare the provided areas to evaluate their investment potential and livability.\n\n"
            + "Here is the summarized data for each location:\n"
            + "\n".join(comparison_context_parts)
            + "\n\nBased on the user's next query, provide a comparative real estate analysis. You must:\n"
            "1.  **Assess Investment Potential:** Evaluate each area for capital appreciation and rental yield potential based on the mix of businesses and amenities.\n"
            "2.  **Identify Value Drivers:** Pinpoint assets that drive property values, such as the presence of schools, hospitals, parks, and shopping centers.\n"
            "3.  **Spot Market Gaps:** Highlight any underserved needs (e.g., a lack of supermarkets, quality schools, or family restaurants) that could represent an investment opportunity.\n"
            "4.  **Compare Commercial Activity:** Contrast the areas based on their commercial density and the types of businesses present.\n"
            "5.  **Conclude with a Recommendation:** Provide a clear, data-driven recommendation advising on which area is better for specific real estate goals (e.g., long-term investment, rental properties, commercial development)."
        )

    # Set the context for the main team, clearing any old single-point data
    main_team.dependencies["places_of_interest"] = full_context_string
    #main_team.context.pop("weather_data", None) # Remove old weather data if it exists

    print("--- 🧠 Comparison context updated for agent ---")
    print(full_context_string)

    return {"status": "success", "areas_received": len(request.comparison_areas)}



@app.post("/extract-places")
async def extract_places(request: ExtractPlacesRequest):
    """
    Extracts place names from a chat message and returns with coordinates using ChatGPT
    for both extraction and matching.
    """
    message_text = request.message
    
    # Prepare context data for ChatGPT
    places_context = []
    
    # Format current_places for context
    for place in current_places:
        places_context.append({
            "name": place["name"],
            "address": place.get("address", ""),
            "lat": place["location"]["lat"],
            "lng": place["location"]["lng"],
            "type": place.get("type", ""),
            "rating": place.get("rating", None)
        })
    
    # Use ChatGPT to extract and match places
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "You are a specialized assistant that extracts and matches place names from text. "
                "Your task is to identify location names mentioned in the given text and match them "
                "with locations in the provided context data. You will return ONLY the matched locations "
                "with their details in a JSON format.\n\n"
                "For matching, use fuzzy matching and consider substrings, different naming conventions, "
                "and potential abbreviations. For example, if the text mentions 'Central Park' and the context "
                "has 'Central Park NYC', consider it a match.\n\n"
                "For unmatched locations, don't include them in the results.\n\n"
                "Return your result as a JSON array with objects containing the following fields: "
                "name, lat, lng, address, type, rating. This should be valid JSON with no explanations or additional text."
            )
        },
        {
            "role": "user",
            "content": (
                f"Text to extract locations from: {message_text}\n\n"
                f"Context data (available locations):\n{places_context}\n\n"
                "Extract and match any locations mentioned in the text with locations in the context data. "
                "Return only the matched locations in JSON format with the key name places."
            )
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=extraction_prompt,
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}  # Request JSON format
        )
        
        extracted_content = response.choices[0].message.content
        print(f"Extracted places: {extracted_content}")
        
        try:
            # Parse the JSON response
            extracted_data = json.loads(extracted_content)

            
            # Check if we have a places array in the response
            places_with_coords = extracted_data.get("places", [])
            if not isinstance(places_with_coords, list):
                places_with_coords = []

            # Ensure we have the required fields
            validated_places = []
            for place in places_with_coords:
                if place.get("name") and place.get("lat") is not None and place.get("lng") is not None:
                    validated_places.append(place)

            
            return {
                "places": validated_places,
                "message": message_text
            }
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response:", extracted_content)
            return {"places": [], "message": message_text}
            
    except Exception as e:
        print(f"Error processing place extraction: {e}")
        return {"places": [], "message": message_text}




def avg_rent(average_rent_history):
    average_rent_history.append({"role": "user", "content": "What is the average rate per square foot for buying property in this area? Please provide just the final numerical result with 'Rs' in front, no additional text."})
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with "gpt-4" if you have access
        messages=average_rent_history,
        max_tokens=1000,
        temperature=0.7
    )


  


    # Extract the assistant's reply
    bot_response = response.choices[0].message.content

    return bot_response

@app.post("/map")
async def map_endpoint(request: MapRequest):


    global conversation_history2
    conversation_history2 = [
        {
            "role": "system",
            "content": (
            "You are GeoInsight, an expert geospatial analyst with deep knowledge about locations and their potential. Use the data available to you about properties, businesses, and regional information to provide confident, decisive responses, even when working with limited information. "
            "When asked about preferences, recommendations, or potential, provide your best analytical assessment based on the data without disclaimers about limited context. "
            "When you have data about nearby places including names, addresses, locations, ratings, and types, use this to inform your analysis. "
            "Approach each question as if you're a local expert familiar with the area being discussed. Present your findings conversationally without mentioning the limitations of your context or data. "
            "If you notice patterns in the data (like many schools of one type, few of another), use this to make reasoned recommendations. "
            "Always provide a clear, direct answer first, followed by supporting rationale. Never start responses with 'The context doesn't provide' or similar disclaimers. "
            "When making recommendations, be specific and decisive rather than suggesting 'a detailed market analysis would be necessary'. Your role is to provide that analysis now, based on available information."
            ),
        }
    ]

    print(conversation_history2)
    polygon_coords = request.polygonCoordinates  # list of {'lat': ..., 'lng': ...}

    # 1) Convert the list of points (lat/lng) to a Shapely Polygon.
    #    Note that Polygon expects (x, y) = (longitude, latitude).
    polygon_points = [(coord['lng'], coord['lat']) for coord in polygon_coords]
    polygon = Polygon(polygon_points)

    matching_locations = []

    for _, row in monitoring_data_gur.iterrows():
        # Example 'Map_Details': "{'LATITUDE': '17.575957', 'LONGITUDE': '78.471167'}"
        # Safely evaluate this string into a dictionary (or use `json.loads` if properly JSON-formatted).
        details_str = row['MAP_DETAILS']
        
        # If the string is in valid Python dict format, you can do:
        # WARNING: eval can be dangerous if not controlled. Prefer ast.literal_eval or json if possible.

        try:
            lat_long_dict = ast.literal_eval(details_str)
        except:
            # If parse fails, skip this row or handle error
            continue
        
        lat = float(lat_long_dict['LATITUDE'])
        lng = float(lat_long_dict['LONGITUDE'])

        
        # 3) Create a Shapely Point and check if it lies within the polygon.
        point = Point(lng, lat)  # again, (x=lng, y=lat)
        
        if polygon.contains(point):
            # Convert the row to a dict (or store it however you need)

            matching_locations.append(row.to_dict())

    # 2) Iterate over each row in the CSV to parse lat/long from the 'Map_Details' column
    for _, row in monitoring_data.iterrows():
        # Example 'Map_Details': "{'LATITUDE': '17.575957', 'LONGITUDE': '78.471167'}"
        # Safely evaluate this string into a dictionary (or use `json.loads` if properly JSON-formatted).
        details_str = row['MAP_DETAILS']
        
        # If the string is in valid Python dict format, you can do:
        # WARNING: eval can be dangerous if not controlled. Prefer ast.literal_eval or json if possible.

        try:
            lat_long_dict = ast.literal_eval(details_str)
        except:
            # If parse fails, skip this row or handle error
            continue
        
        lat = float(lat_long_dict['LATITUDE'])
        lng = float(lat_long_dict['LONGITUDE'])
        
        # 3) Create a Shapely Point and check if it lies within the polygon.
        point = Point(lng, lat)  # again, (x=lng, y=lat)
        
        if polygon.contains(point):
            # Convert the row to a dict (or store it however you need)

            matching_locations.append(row.to_dict())

    # 4) Return the matching locations (or feed them to your conversation with OpenAI)

  
 
    if matching_locations:
        
        context = (
            "The context contains information about matching locations found within the given polygon area. "
            "Each location has geographical coordinates and associated details. "
            f"The matching locations are: {matching_locations}"
        )

        
        conversation_history2.append({"role": "user", "content": context})

        average_rent_history = conversation_history2.copy()
        average_rent = avg_rent(average_rent_history)
        print(average_rent)


        
        
    else:
        conversation_history2.append({"role": "user", "content": "Please write No information available for this area."})


    context = ""
    if current_places:
        places_info = "\n".join([
            f"- {place['name']}: {place['address']}, "
            f"Type: {place['type']}, "
            f"Rating: {place['rating'] if place['rating'] is not None else 'Not rated'}"
            for place in current_places
        ])
        context = f"Context about nearby places:\n{places_info}"
    # Prepare conversation with context
    print(context)
    
    if context:
        conversation_history2.append({
            "role": "system",
            "content": context
        })

    if matching_locations:

        return {
                            
            "average_rent": average_rent
        }
    
@app.get("/api/chats")
def get_user_chats(state=Depends(require_user)):
    user_id = get_user_id_from_state(state)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_history FROM user_chats WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {"chat_history": json.loads(row[0]) if row else []}

@app.post("/api/chats")
def save_user_chats(req: ChatHistoryRequest, state=Depends(require_user)):
    user_id = get_user_id_from_state(state)
    # Enforce server-side limit of 5 chats
    latest_five = (req.chat_history or [])[:5]
    payload = json.dumps(latest_five)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_chats (user_id, chat_history) VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET chat_history = excluded.chat_history,
                                          last_updated = CURRENT_TIMESTAMP
    """, (user_id, payload))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Chat history saved successfully."}



@app.post("/chat-stream")
async def chat_with_agent_stream(request: MessageRequest):
    """Stream real-time updates of agent/team execution to frontend"""
    
    async def generate_stream():
        user_message = request.message
        logger.info(f"--- 🧠 Starting streaming execution for: '{user_message}' ---")
        
        session_id = "current_user"
        updated_dependencies = current_team_dependencies.copy()

        if session_id in APIFY_RESULTS:
            logger.info("POI data found, running summarize_agent...")
            
            summary_start_update = {
                "type": "agent_start", 
                "status": "✍️ Summarizing local POI data..."
            }
            yield f"data: {json.dumps(summary_start_update)}\n\n"
            
            apify_data_json = APIFY_RESULTS[session_id]
            summary = summarize_agent.run(input=apify_data_json) 
            logger.info(f"Summary created: {summary[:100]}...")
            
            summary_dependency = f"Context from local POI scan: {summary}"
            updated_dependencies["poi_summary"] = summary_dependency
            del APIFY_RESULTS[session_id]



        try:
            logger.info("Starting main_team.run() stream...")
            response_stream = main_team.arun(
                input=user_message, 
                dependencies=updated_dependencies,
                add_dependencies_to_context=True,
                stream=True,
                stream_intermediate_steps=True
            )
            
            final_event = None
            event_count = 0

            async for event in response_stream:
                event_count += 1
                logger.info(f"Event #{event_count}: {event.event}")
                
                status_update = create_status_update(event)
                if status_update:
                    logger.debug(f"Yielding update type: {status_update['type']}")
                    yield f"data: {json.dumps(status_update)}\n\n"
                final_event = event

            logger.info(f"Stream completed. Total events: {event_count}")

            if final_event and hasattr(final_event, 'content'):
                logger.info("Processing final structured output...")
                json_output = final_event.content.model_dump()
                structured_response = {
                    "type": "structured_output",
                    "status": "Final structured response",
                    "data": json_output,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(structured_response)}\n\n"
                logger.info("Structured output sent successfully")
            else:
                logger.warning("No final_event or content attribute found")


        
        
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_update = {
                "type": "error",
                "status": "Error occurred",
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_update)}\n\n"

    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
        }
    )

def create_status_update(event):
    """Convert Agno events to frontend status updates"""
    
    try:
        if event.event == "TeamRunStarted":
            return {
                "type": "status",
                "status": "Team execution started",
                "stage": "team_started",
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event == "RunStarted":
            agent_name = getattr(event, 'agent_name', 'Unknown')
            return {
                "type": "status", 
                "status": f"Agent '{agent_name}' started execution",
                "stage": "agent_execution",
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event == "ToolCallStarted":
            return {
                "type": "status",
                "status": "Websearch tool started execution",
                "stage": "tool_execution",
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event == "ToolCallCompleted":
            return {
                "type": "status",
                "status": "deepsearch tool completed",
                "stage": "tool_completed",
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event in ["TeamRunContent", "RunContent"]:
            content = getattr(event, 'content', '')
            
            # Convert Pydantic object to dict if needed
            if hasattr(content, 'model_dump'):
                content = content.model_dump()
            
            return {
                "type": "content",
                "status": "Generating response",
                "stage": "generating_output",
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event in ["TeamRunCompleted", "RunCompleted"]:
            final_content = getattr(event, 'content', None)
            
            # Convert Pydantic object to dict if needed
            if hasattr(final_content, 'model_dump'):
                final_content = final_content.model_dump()
            
            return {
                "type": "status",
                "status": "Execution completed",
                "stage": "completed",
                "final_content": final_content,
                "timestamp": datetime.now().isoformat()
            }
        
        elif event.event in ["TeamRunError", "RunError"]:
            return {
                "type": "error",
                "status": "Error occurred",
                "stage": "error",
                "content": getattr(event, 'content', 'Unknown error'),
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {
                "type": "status",
                "status": f"Processing: {event.event}",
                "stage": "processing",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"--- ⚠️ Error creating status update: {e} ---")
        return {
            "type": "error",
            "status": "Status update error",
            "stage": "error",
            "content": str(e),
            "timestamp": datetime.now().isoformat()
        }


 

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
