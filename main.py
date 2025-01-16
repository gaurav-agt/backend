from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import openai
import os
from shapely.geometry import Point, Polygon
import ast

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY", "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")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["comfy-croissant-6c68a3.netlify.app"],  # Replace with your Netlify URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Our CSV has a column 'Map_Details' with lat/long in the format:
# {'LATITUDE': '17.575957', 'LONGITUDE': '78.471167'}
csv_file_path = "hyderabad.csv"
monitoring_data = pd.read_csv(csv_file_path)

class MessageRequest(BaseModel):
    message: str
    

class MapRequest(BaseModel):
    polygonCoordinates: list


conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. You must answer only based on the provided context. "
            "The context contains information about real estate properties and their details. "
            "Do not use any external information."
            "Just answer the query in a concise way and don't show extra information."
        ),
    }  # Starting system message
]

conversation_histor = [
    {
      "role": "system",
      "content": "You must respond with exactly this phrase: 'Here are the gyms' - no other text, no punctuation, no additional information."
    }
  ]






  # Starting system message]

@app.post("/map")
async def map_endpoint(request: MapRequest):
    polygon_coords = request.polygonCoordinates  # list of {'lat': ..., 'lng': ...}

    # 1) Convert the list of points (lat/lng) to a Shapely Polygon.
    #    Note that Polygon expects (x, y) = (longitude, latitude).
    polygon_points = [(coord['lng'], coord['lat']) for coord in polygon_coords]
    polygon = Polygon(polygon_points)

    matching_locations = []

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

        print(context)
        conversation_history.append({"role": "user", "content": context})
        
        
    else:
        conversation_history.append({"role": "user", "content": "Please write No information available for this area."})
    

@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    user_message = request.message
    
    conversation_history.append({"role": "user", "content": user_message})
    # Call OpenAI's ChatCompletion API with the updated conversation history




    conversation_histor.append({"role": "user", "content": user_message})
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Replace with "gpt-4" if you have access
        messages=conversation_history,
        max_tokens=1000,
        temperature=0.7
    )


  


    # Extract the assistant's reply
    bot_response = response.choices[0].message.content



    # Append the assistant's response to conversation history
    conversation_history.append({"role": "assistant", "content": bot_response})

    # Return the assistant's response
    return {"response": bot_response}

 

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
