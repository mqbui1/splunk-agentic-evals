"""Mock tools for the travel planner agent."""


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search for available flights between two cities on a given date."""
    return {
        "flights": [
            {"airline": "United", "flight": "UA123", "price": 289, "depart": "06:00", "arrive": "14:32"},
            {"airline": "Delta",  "flight": "DL456", "price": 341, "depart": "09:15", "arrive": "17:48"},
            {"airline": "JetBlue","flight": "B6789", "price": 265, "depart": "14:00", "arrive": "22:19"},
        ],
        "origin": origin,
        "destination": destination,
        "date": date,
    }


def search_hotels(city: str, check_in: str, check_out: str) -> dict:
    """Search for available hotels in a city."""
    return {
        "hotels": [
            {"name": "Marriott Midtown",  "stars": 4, "price_per_night": 189, "available": True},
            {"name": "Hilton Times Square","stars": 4, "price_per_night": 215, "available": True},
            {"name": "Pod 51",             "stars": 3, "price_per_night": 119, "available": True},
        ],
        "city": city,
        "check_in": check_in,
        "check_out": check_out,
    }


def get_weather(city: str, date: str) -> dict:
    """Get weather forecast for a city on a given date."""
    return {
        "city": city,
        "date": date,
        "forecast": "Partly cloudy",
        "high_f": 72,
        "low_f": 58,
        "precipitation_pct": 20,
    }
