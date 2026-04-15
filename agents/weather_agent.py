import requests, os

API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_weather(city: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"

def run(query: str, context: str) -> str:
    """Simple Weather Agent: Extracts city from query and fetches weather.
    """
    # Naive city extraction (could be improved with NER)
    words = query.split()
    city = None
    for word in words:
        if word.istitle():  # Assume city names are capitalized
            city = word
            break

    if not city:
        return "❌ Could not extract city from query."

    try:
        weather_info = get_weather(city)
        return weather_info
    except Exception as e:
        return f"❌ Weather API Error: {str(e)}"
    