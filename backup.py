from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import openai
import os
from math import radians, cos, sin, asin, sqrt


# Initialize FastAPI app
app = FastAPI()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# Enable CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


csv_file_path = "Comprehensive_Monitoring_Locations_Filled.csv"
monitoring_data = pd.read_csv(csv_file_path)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c  # Distance in kilometers


# Define request model
class MessageRequest(BaseModel):
    message: str
    polygonCoordinates: listr

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}  # Starting system message
]

# Endpoint to handle user chat messages
@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    user_message = request.message
    polygon_coords = request.polygonCoordinates

    

    try:
        # Append the user's message to conversation history
        
        matching_locations = []

        for _, row in monitoring_data.iterrows():
            location_lat, location_lng = row['Latitude'], row['Longitude']

            # Check proximity to any point in the polygon
            for point in polygon_coords:
                polygon_lat, polygon_lng = point['lat'], point['lng']
                distance = haversine(location_lat, location_lng, polygon_lat, polygon_lng)

                if distance <= 3:  # 2 km radius
                    
                    location_data = row.to_dict()
                    location_data['Distance_km'] = round(distance, 2)
                    matching_locations.append(location_data)
                    break

        
        if matching_locations:
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "user", "content": str(matching_locations)})
            
            
        else:
            conversation_history.append({"role": "user", "content": "Please write No information available for this area."})
           

        # Call OpenAI's ChatCompletion API with the updated conversation history
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Replace with "gpt-4" if you have access
            messages=conversation_history,
            max_tokens=150,
            temperature=0.7
        )

        # Extract the assistant's reply
        bot_response = response.choices[0].message.content

        # Append the assistant's response to conversation history
        conversation_history.append({"role": "assistant", "content": bot_response})

        # Return the assistant's response
        return {"response": bot_response}

    except Exception as e:
        # Handle errors gracefully
        return {"response": f"Error: {str(e)}"}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
