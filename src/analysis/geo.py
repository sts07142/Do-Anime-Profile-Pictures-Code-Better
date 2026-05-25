"""Coarse country / region detection from free-text GitHub location strings.

GitHub stores `location` as a free-text field — anything from "Tokyo" to
"Earth" to "remote 🌍". This module produces a best-effort ISO 3166-1 alpha-2
country code and continental region for filtering / aggregation. Anything
unmatched returns None and is treated as "Unknown" downstream.

Designed for the Streamlit dashboard's country/region filter, not for any
geolocation precision claim. We deliberately match on country names and a
handful of major cities per country; misses are accepted in exchange for
keeping the lookup dependency-free.
"""
from __future__ import annotations

import re
from typing import Optional

# country code -> list of lower-case match patterns. Order matters within the
# list (more specific first), and ALL keys/patterns are lowercase. The matcher
# below uses substring containment with word-boundary regex so "us" won't
# match "russia" or "australia".
_COUNTRY_PATTERNS: dict[str, list[str]] = {
    "JP": ["japan", "tokyo", "osaka", "kyoto", "yokohama", "fukuoka",
           "nagoya", "sapporo", "日本", "東京"],
    "KR": ["korea", "seoul", "busan", "republic of korea", "south korea",
           "한국", "서울"],
    "CN": ["china", "beijing", "shanghai", "shenzhen", "hangzhou", "chengdu",
           "guangzhou", "wuhan", "中国", "北京"],
    "TW": ["taiwan", "taipei", "kaohsiung", "台灣", "台北"],
    "HK": ["hong kong", "香港"],
    "SG": ["singapore"],
    "IN": ["india", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad",
           "chennai", "kolkata", "pune", "noida", "gurgaon", "gurugram"],
    "PK": ["pakistan", "lahore", "karachi", "islamabad"],
    "BD": ["bangladesh", "dhaka", "chittagong"],
    "ID": ["indonesia", "jakarta", "surabaya", "bandung"],
    "VN": ["vietnam", "ho chi minh", "hanoi", "saigon"],
    "TH": ["thailand", "bangkok"],
    "PH": ["philippines", "manila", "cebu"],
    "MY": ["malaysia", "kuala lumpur", "penang"],
    "TR": ["turkey", "türkiye", "istanbul", "ankara", "izmir"],
    "IR": ["iran", "tehran"],
    "IL": ["israel", "tel aviv", "jerusalem"],
    "SA": ["saudi arabia", "riyadh", "jeddah"],
    "AE": ["united arab emirates", "uae", "dubai", "abu dhabi"],
    "EG": ["egypt", "cairo", "alexandria"],
    "NG": ["nigeria", "lagos", "abuja"],
    "KE": ["kenya", "nairobi"],
    "ZA": ["south africa", "johannesburg", "cape town"],
    "MA": ["morocco", "casablanca"],

    "US": ["united states", "u.s.", "u.s.a", "usa", "america", "san francisco",
           "new york", "seattle", "boston", "los angeles", "chicago", "austin",
           "atlanta", "denver", "portland", "san diego", "san jose",
           "silicon valley", "bay area", "nyc", "sf", "la, ca",
           "mountain view", "palo alto"],
    "CA": ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary"],
    "MX": ["mexico", "ciudad de méxico", "mexico city"],
    "BR": ["brazil", "brasil", "são paulo", "sao paulo", "rio de janeiro"],
    "AR": ["argentina", "buenos aires"],
    "CL": ["chile", "santiago"],
    "CO": ["colombia", "bogota", "medellin"],
    "PE": ["peru", "lima"],
    "VE": ["venezuela", "caracas"],

    "GB": ["united kingdom", "uk", "england", "london", "manchester",
           "edinburgh", "scotland", "wales", "ireland-uk"],
    "IE": ["ireland", "dublin"],
    "DE": ["germany", "berlin", "munich", "hamburg", "frankfurt", "deutschland"],
    "FR": ["france", "paris", "lyon", "marseille"],
    "ES": ["spain", "madrid", "barcelona", "españa"],
    "IT": ["italy", "rome", "milan", "italia"],
    "NL": ["netherlands", "amsterdam", "rotterdam", "the hague"],
    "BE": ["belgium", "brussels", "antwerp"],
    "CH": ["switzerland", "zurich", "geneva"],
    "AT": ["austria", "vienna"],
    "SE": ["sweden", "stockholm", "gothenburg"],
    "NO": ["norway", "oslo"],
    "DK": ["denmark", "copenhagen"],
    "FI": ["finland", "helsinki"],
    "PL": ["poland", "warsaw", "kraków", "krakow"],
    "CZ": ["czech", "prague"],
    "HU": ["hungary", "budapest"],
    "RO": ["romania", "bucharest"],
    "PT": ["portugal", "lisbon", "porto"],
    "GR": ["greece", "athens"],
    "RU": ["russia", "moscow", "saint petersburg"],
    "UA": ["ukraine", "kyiv", "kiev"],
    "BY": ["belarus", "minsk"],

    "AU": ["australia", "sydney", "melbourne", "brisbane", "perth"],
    "NZ": ["new zealand", "auckland", "wellington"],
}

_REGION: dict[str, str] = {
    # East Asia
    "JP": "East Asia", "KR": "East Asia", "CN": "East Asia",
    "TW": "East Asia", "HK": "East Asia",
    # SE Asia
    "SG": "Southeast Asia", "ID": "Southeast Asia", "VN": "Southeast Asia",
    "TH": "Southeast Asia", "PH": "Southeast Asia", "MY": "Southeast Asia",
    # South Asia
    "IN": "South Asia", "PK": "South Asia", "BD": "South Asia",
    # Middle East
    "TR": "Middle East", "IR": "Middle East", "IL": "Middle East",
    "SA": "Middle East", "AE": "Middle East",
    # Africa
    "EG": "Africa", "NG": "Africa", "KE": "Africa", "ZA": "Africa", "MA": "Africa",
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    # Latin America
    "BR": "Latin America", "AR": "Latin America", "CL": "Latin America",
    "CO": "Latin America", "PE": "Latin America", "VE": "Latin America",
    # Europe
    "GB": "Europe", "IE": "Europe", "DE": "Europe", "FR": "Europe",
    "ES": "Europe", "IT": "Europe", "NL": "Europe", "BE": "Europe",
    "CH": "Europe", "AT": "Europe", "SE": "Europe", "NO": "Europe",
    "DK": "Europe", "FI": "Europe", "PL": "Europe", "CZ": "Europe",
    "HU": "Europe", "RO": "Europe", "PT": "Europe", "GR": "Europe",
    "RU": "Europe", "UA": "Europe", "BY": "Europe",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania",
}

_COUNTRY_NAMES: dict[str, str] = {
    "JP": "Japan", "KR": "South Korea", "CN": "China", "TW": "Taiwan",
    "HK": "Hong Kong", "SG": "Singapore", "IN": "India", "PK": "Pakistan",
    "BD": "Bangladesh", "ID": "Indonesia", "VN": "Vietnam", "TH": "Thailand",
    "PH": "Philippines", "MY": "Malaysia", "TR": "Turkey", "IR": "Iran",
    "IL": "Israel", "SA": "Saudi Arabia", "AE": "UAE", "EG": "Egypt",
    "NG": "Nigeria", "KE": "Kenya", "ZA": "South Africa", "MA": "Morocco",
    "US": "United States", "CA": "Canada", "MX": "Mexico",
    "BR": "Brazil", "AR": "Argentina", "CL": "Chile", "CO": "Colombia",
    "PE": "Peru", "VE": "Venezuela",
    "GB": "United Kingdom", "IE": "Ireland", "DE": "Germany", "FR": "France",
    "ES": "Spain", "IT": "Italy", "NL": "Netherlands", "BE": "Belgium",
    "CH": "Switzerland", "AT": "Austria", "SE": "Sweden", "NO": "Norway",
    "DK": "Denmark", "FI": "Finland", "PL": "Poland", "CZ": "Czech Republic",
    "HU": "Hungary", "RO": "Romania", "PT": "Portugal", "GR": "Greece",
    "RU": "Russia", "UA": "Ukraine", "BY": "Belarus",
    "AU": "Australia", "NZ": "New Zealand",
}


def detect_country(text: Optional[str]) -> Optional[str]:
    """Return ISO-2 country code, or None if no match."""
    if not text or not isinstance(text, str):
        return None
    s = text.lower()
    for code, patterns in _COUNTRY_PATTERNS.items():
        for p in patterns:
            if re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", s):
                return code
    return None


def country_to_region(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return _REGION.get(code)


def country_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return _COUNTRY_NAMES.get(code, code)


def all_regions() -> list[str]:
    return sorted(set(_REGION.values()))


def all_countries() -> list[tuple[str, str]]:
    """[(code, name), ...] sorted by name."""
    return sorted(((c, _COUNTRY_NAMES.get(c, c)) for c in _COUNTRY_NAMES),
                  key=lambda x: x[1])
