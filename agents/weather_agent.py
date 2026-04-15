import requests

def get_weather(city: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"