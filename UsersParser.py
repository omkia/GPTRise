import re
import html
import pymysql
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
import unicodedata

# ---------------------------------------------
# MariaDB connection
# ---------------------------------------------
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="stack",
    charset="utf8mb4",
    autocommit=False
)
cursor = conn.cursor()

# ---------------------------------------------
# Country to Continent Mapping
# ---------------------------------------------
CONTINENT_MAP = {
    # Africa
    "algeria": "Africa", "angola": "Africa", "benin": "Africa", "botswana": "Africa",
    "burkina faso": "Africa", "burundi": "Africa", "cabo verde": "Africa", "cameroon": "Africa",
    "central african republic": "Africa", "chad": "Africa", "comoros": "Africa", "congo (congo-brazzaville)": "Africa",
    "djibouti": "Africa", "egypt": "Africa", "equatorial guinea": "Africa", "eritrea": "Africa",
    "eswatini": "Africa", "ethiopia": "Africa", "gabon": "Africa", "gambia": "Africa",
    "ghana": "Africa", "guinea": "Africa", "guinea-bissau": "Africa", "kenya": "Africa",
    "lesotho": "Africa", "liberia": "Africa", "libya": "Africa", "madagascar": "Africa",
    "malawi": "Africa", "mali": "Africa", "mauritania": "Africa", "mauritius": "Africa",
    "morocco": "Africa", "mozambique": "Africa", "namibia": "Africa", "niger": "Africa",
    "nigeria": "Africa", "rwanda": "Africa", "sao tome and principe": "Africa", "senegal": "Africa",
    "seychelles": "Africa", "sierra leone": "Africa", "somalia": "Africa", "south africa": "Africa",
    "south sudan": "Africa", "sudan": "Africa", "tanzania": "Africa", "togo": "Africa",
    "tunisia": "Africa", "uganda": "Africa", "zambia": "Africa", "zimbabwe": "Africa",

    # Asia
    "afghanistan": "Asia", "armenia": "Asia", "azerbaijan": "Asia", "bahrain": "Asia",
    "bangladesh": "Asia", "bhutan": "Asia", "brunei": "Asia", "cambodia": "Asia",
    "china": "Asia", "georgia": "Asia", "india": "Asia", "indonesia": "Asia",
    "iran": "Asia", "iraq": "Asia", "israel": "Asia", "japan": "Asia",
    "jordan": "Asia", "kazakhstan": "Asia", "kuwait": "Asia", "kyrgyzstan": "Asia",
    "laos": "Asia", "lebanon": "Asia", "malaysia": "Asia", "maldives": "Asia",
    "mongolia": "Asia", "myanmar": "Asia", "nepal": "Asia", "north korea": "Asia",
    "oman": "Asia", "pakistan": "Asia", "palestine": "Asia", "philippines": "Asia",
    "qatar": "Asia", "saudi arabia": "Asia", "singapore": "Asia", "south korea": "Asia",
    "sri lanka": "Asia", "syria": "Asia", "taiwan": "Asia", "tajikistan": "Asia",
    "thailand": "Asia", "timor-leste": "Asia", "turkey": "Asia", "turkmenistan": "Asia",
    "united arab emirates": "Asia", "uzbekistan": "Asia", "vietnam": "Asia", "yemen": "Asia",

    # Europe
    "albania": "Europe", "andorra": "Europe", "austria": "Europe", "belarus": "Europe",
    "belgium": "Europe", "bosnia and herzegovina": "Europe", "bulgaria": "Europe", "croatia": "Europe",
    "cyprus": "Europe", "czechia": "Europe", "denmark": "Europe", "estonia": "Europe",
    "finland": "Europe", "france": "Europe", "germany": "Europe", "greece": "Europe",
    "hungary": "Europe", "iceland": "Europe", "ireland": "Europe", "italy": "Europe",
    "latvia": "Europe", "liechtenstein": "Europe", "lithuania": "Europe", "luxembourg": "Europe",
    "malta": "Europe", "moldova": "Europe", "monaco": "Europe", "montenegro": "Europe",
    "netherlands": "Europe", "north macedonia": "Europe", "norway": "Europe", "poland": "Europe",
    "portugal": "Europe", "romania": "Europe", "russia": "Europe", "san marino": "Europe",
    "serbia": "Europe", "slovakia": "Europe", "slovenia": "Europe", "spain": "Europe",
    "sweden": "Europe", "switzerland": "Europe", "ukraine": "Europe", "united kingdom": "Europe",
    "vatican city": "Europe",

    # North America
    "antigua and barbuda": "North America", "bahamas": "North America", "barbados": "North America",
    "belize": "North America", "canada": "North America", "costa rica": "North America",
    "cuba": "North America", "dominica": "North America", "dominican republic": "North America",
    "el salvador": "North America", "grenada": "North America", "guatemala": "North America",
    "haiti": "North America", "honduras": "North America", "jamaica": "North America",
    "mexico": "North America", "nicaragua": "North America", "panama": "North America",
    "saint kitts and nevis": "North America", "saint lucia": "North America",
    "saint vincent and the grenadines": "North America", "trinidad and tobago": "North America",
    "usa": "North America",

    # South America
    "argentina": "South America", "bolivia": "South America", "brazil": "South America",
    "chile": "South America", "colombia": "South America", "ecuador": "South America",
    "guyana": "South America", "paraguay": "South America", "peru": "South America",
    "suriname": "South America", "uruguay": "South America", "venezuela": "South America",

    # Oceania
    "australia": "Oceania", "fiji": "Oceania", "kiribati": "Oceania", "marshall islands": "Oceania",
    "micronesia": "Oceania", "nauru": "Oceania", "new zealand": "Oceania", "palau": "Oceania",
    "papua new guinea": "Oceania", "samoa": "Oceania", "solomon islands": "Oceania",
    "tonga": "Oceania", "tuvalu": "Oceania", "vanuatu": "Oceania",
}

# ---------------------------------------------
# Full Country Keywords (from previous merge)
# ---------------------------------------------
COUNTRIES = {
    "afghanistan": ["afghanistan", "kabul", "herat", "kandahar"],
    "albania": ["albania", "tirana"],
    "algeria": ["algeria", "algiers", "oran"],
    "andorra": ["andorra", "andorra la vella"],
    "angola": ["angola", "luanda"],
    "antigua and barbuda": ["antigua and barbuda", "st. john's"],
    "argentina": ["argentina", "buenos aires", "cordoba", "rosario"],
    "armenia": ["armenia", "yerevan"],
    "australia": ["australia", "sydney", "melbourne", "adelaide", "brisbane", "perth"],
    "austria": ["austria", "vienna", "salzburg", "innsbruck"],
    "azerbaijan": ["azerbaijan", "baku"],
    "bahamas": ["bahamas", "nassau"],
    "bahrain": ["bahrain", "manama"],
    "bangladesh": ["bangladesh", "dhaka", "chittagong", "khulna", "rajshahi"],
    "barbados": ["barbados", "bridgetown"],
    "belarus": ["belarus", "minsk"],
    "belgium": ["belgium", "brussels", "antwerp"],
    "belize": ["belize", "belize city"],
    "benin": ["benin", "porto-novo", "cotonou"],
    "bhutan": ["bhutan", "thimphu"],
    "bolivia": ["bolivia", "la paz", "sucre", "santa cruz"],
    "bosnia and herzegovina": ["bosnia and herzegovina", "sarajevo", "banja luka"],
    "botswana": ["botswana", "gaborone"],
    "brazil": ["brazil", "brasil", "rio", "rio de janeiro", "sao paulo"],
    "brunei": ["brunei", "bandar seri begawan"],
    "bulgaria": ["bulgaria", "sofia", "plovdiv"],
    "burkina faso": ["burkina faso", "ouagadougou"],
    "burundi": ["burundi", "gitega", "bujumbura"],
    "cabo verde": ["cabo verde", "praia"],
    "cambodia": ["cambodia", "phnom penh", "siem reap"],
    "cameroon": ["cameroon", "yaounde", "douala"],
    "canada": ["canada", "toronto", "vancouver", "montreal", "ottawa"],
    "central african republic": ["central african republic", "bangui"],
    "chad": ["chad", "n'djamena"],
    "chile": ["chile", "santiago", "valparaiso"],
    "china": ["china", "prc", "beijing", "shanghai", "guangzhou", "shenzhen"],
    "colombia": ["colombia", "bogota", "medellin", "cali"],
    "comoros": ["comoros", "moroni"],
    "congo (congo-brazzaville)": ["congo", "brazzaville"],
    "costa rica": ["costa rica", "san jose"],
    "croatia": ["croatia", "zagreb", "split"],
    "cuba": ["cuba", "havana"],
    "cyprus": ["cyprus", "nicosia"],
    "czechia": ["czechia", "czech republic", "prague", "brno"],
    "denmark": ["denmark", "copenhagen"],
    "djibouti": ["djibouti"],
    "dominica": ["dominica", "roseau"],
    "dominican republic": ["dominican republic", "santo domingo"],
    "ecuador": ["ecuador", "quito", "guayaquil"],
    "egypt": ["egypt", "cairo", "alexandria", "giza"],
    "el salvador": ["el salvador", "san salvador"],
    "equatorial guinea": ["equatorial guinea", "malabo"],
    "eritrea": ["eritrea", "asmara"],
    "estonia": ["estonia", "tallinn"],
    "eswatini": ["eswatini", "mbabane"],
    "ethiopia": ["ethiopia", "addis ababa"],
    "fiji": ["fiji", "suva"],
    "finland": ["finland", "helsinki", "espoo"],
    "france": ["france", "paris", "lyon", "marseille"],
    "gabon": ["gabon", "libreville"],
    "gambia": ["gambia", "banjul"],
    "georgia": ["georgia", "tbilisi", "batumi"],
    "georgia (usa)": ["georgia usa", "georgia, usa", "atlanta", "savannah"],
    "germany": ["germany", "deutschland", "berlin", "munich", "hamburg"],
    "ghana": ["ghana", "accra", "kumasi"],
    "greece": ["greece", "athens", "thessaloniki"],
    "grenada": ["grenada", "st. george's"],
    "guatemala": ["guatemala", "guatemala city"],
    "guinea": ["guinea", "conakry"],
    "guinea-bissau": ["guinea-bissau", "bissau"],
    "guyana": ["guyana", "georgetown"],
    "haiti": ["haiti", "port-au-prince"],
    "honduras": ["honduras", "tegucigalpa"],
    "hungary": ["hungary", "budapest"],
    "iceland": ["iceland", "reykjavik"],
    "india": ["india", "delhi", "new delhi", "bangalore", "mumbai"],
    "indonesia": ["indonesia", "jakarta", "surabaya", "bandung"],
    "iran": ["iran", "tehran", "isfahan", "shiraz", "mashhad", "tabriz"],
    "iraq": ["iraq", "baghdad", "basra", "erbil"],
    "ireland": ["ireland", "dublin"],
    "israel": ["israel", "tel aviv", "jerusalem", "haifa"],
    "italy": ["italy", "rome", "milan", "naples", "turin"],
    "jamaica": ["jamaica", "kingston"],
    "japan": ["japan", "tokyo", "osaka", "yokohama"],
    "jordan": ["jordan", "amman"],
    "kazakhstan": ["kazakhstan", "astana", "almaty"],
    "kenya": ["kenya", "nairobi", "mombasa"],
    "kiribati": ["kiribati", "tarawa"],
    "kuwait": ["kuwait", "kuwait city"],
    "kyrgyzstan": ["kyrgyzstan", "bishkek"],
    "laos": ["laos", "vientiane"],
    "latvia": ["latvia", "riga"],
    "lebanon": ["lebanon", "beirut"],
    "lesotho": ["lesotho", "maseru"],
    "liberia": ["liberia", "monrovia"],
    "libya": ["libya", "tripoli", "benghazi"],
    "liechtenstein": ["liechtenstein", "vaduz"],
    "lithuania": ["lithuania", "vilnius"],
    "luxembourg": ["luxembourg"],
    "madagascar": ["madagascar", "antananarivo"],
    "malawi": ["malawi", "lilongwe"],
    "malaysia": ["malaysia", "kuala lumpur", "george town"],
    "maldives": ["maldives", "male"],
    "mali": ["mali", "bamako"],
    "malta": ["malta", "valletta"],
    "marshall islands": ["marshall islands", "majuro"],
    "mauritania": ["mauritania", "nouakchott"],
    "mauritius": ["mauritius", "port louis"],
    "mexico": ["mexico", "mexico city", "guadalajara", "monterrey"],
    "micronesia": ["micronesia", "palikir"],
    "moldova": ["moldova", "chisinau"],
    "monaco": ["monaco"],
    "mongolia": ["mongolia", "ulaanbaatar"],
    "montenegro": ["montenegro", "podgorica"],
    "morocco": ["morocco", "rabat", "casablanca", "marrakech"],
    "mozambique": ["mozambique", "maputo"],
    "myanmar": ["myanmar", "burma", "naypyidaw", "yangon"],
    "namibia": ["namibia", "windhoek"],
    "nauru": ["nauru"],
    "nepal": ["nepal", "kathmandu"],
    "netherlands": ["netherlands", "holland", "groningen", "amsterdam", "rotterdam", "utrecht"],
    "new zealand": ["new zealand", "auckland", "wellington"],
    "nicaragua": ["nicaragua", "managua"],
    "niger": ["niger", "niamey"],
    "nigeria": ["nigeria", "abuja", "lagos", "kano"],
    "north korea": ["north korea", "dprk", "pyongyang"],
    "north macedonia": ["north macedonia", "skopje"],
    "norway": ["norway", "oslo", "bergen"],
    "oman": ["oman", "muscat"],
    "pakistan": ["pakistan", "islamabad", "karachi", "lahore"],
    "palau": ["palau", "ngerulmud"],
    "palestine": ["palestine", "palestinian territories", "ramallah", "gaza", "west bank"],
    "panama": ["panama", "panama city"],
    "papua new guinea": ["papua new guinea", "port moresby"],
    "paraguay": ["paraguay", "asuncion"],
    "peru": ["peru", "lima", "arequipa"],
    "philippines": ["philippines", "manila", "quezon city", "davao"],
    "poland": ["poland", "warsaw", "krakow", "wroclaw"],
    "portugal": ["portugal", "lisbon", "porto"],
    "qatar": ["qatar", "doha"],
    "romania": ["romania", "bucharest", "cluj-napoca"],
    "russia": ["russia", "russian federation", "moscow", "saint petersburg"],
    "rwanda": ["rwanda", "kigali"],
    "saint kitts and nevis": ["saint kitts and nevis", "basseterre"],
    "saint lucia": ["saint lucia", "castries"],
    "saint vincent and the grenadines": ["saint vincent and the grenadines", "kingstown"],
    "samoa": ["samoa", "apia"],
    "san marino": ["san marino"],
    "sao tome and principe": ["sao tome and principe", "sao tome"],
    "saudi arabia": ["saudi arabia", "riyadh", "jeddah", "mecca"],
    "senegal": ["senegal", "dakar"],
    "serbia": ["serbia", "belgrade", "novi sad"],
    "seychelles": ["seychelles", "victoria"],
    "sierra leone": ["sierra leone", "freetown"],
    "singapore": ["singapore"],
    "slovakia": ["slovakia", "bratislava"],
    "slovenia": ["slovenia", "ljubljana"],
    "solomon islands": ["solomon islands", "honiara"],
    "somalia": ["somalia", "mogadishu"],
    "south africa": ["south africa", "cape town", "johannesburg", "pretoria"],
    "south korea": ["south korea", "korea", "republic of korea", "anyang", "seoul", "busan", "incheon", "daejeon", "ulsan"],
    "south sudan": ["south sudan", "juba"],
    "spain": ["spain", "españa", "madrid", "barcelona", "valencia"],
    "sri lanka": ["sri lanka", "colombo", "kandy"],
    "sudan": ["sudan", "khartoum"],
    "suriname": ["suriname", "paramaribo"],
    "sweden": ["sweden", "stockholm", "gothenburg"],
    "switzerland": ["switzerland", "zurich", "geneva", "bern"],
    "syria": ["syria", "damascus", "aleppo"],
    "taiwan": ["taiwan", "taipei"],
    "tajikistan": ["tajikistan", "dushanbe"],
    "tanzania": ["tanzania", "dar es salaam", "dodoma"],
    "thailand": ["thailand", "bangkok", "chiang mai"],
    "timor-leste": ["timor-leste", "dili"],
    "togo": ["togo", "lome"],
    "tonga": ["tonga", "nuku'alofa"],
    "trinidad and tobago": ["trinidad and tobago", "port of spain"],
    "tunisia": ["tunisia", "tunis"],
    "turkey": ["turkey", "türkiye", "istanbul", "ankara", "izmir", "antalya"],
    "turkmenistan": ["turkmenistan", "ashgabat"],
    "tuvalu": ["tuvalu", "funafuti"],
    "uganda": ["uganda", "kampala"],
    "ukraine": ["ukraine", "kyiv", "lviv", "kharkiv"],
    "united arab emirates": ["uae", "united arab emirates", "dubai", "abu dhabi", "sharjah"],
    "united kingdom": ["united kingdom", "england", "britain", "uk", "london", "manchester", "birmingham", "liverpool"],
    "usa": ["usa", "united states", "united states of america", "america", "u.s.", "new york", "los angeles", "san francisco", "chicago", "houston", "austin", "philadelphia", "washington", "winchester", "indianapolis", "virginia", "va", "ca"],
    "uruguay": ["uruguay", "montevideo"],
    "uzbekistan": ["uzbekistan", "tashkent"],
    "vanuatu": ["vanuatu", "port vila"],
    "vatican city": ["vatican city", "vatican"],
    "venezuela": ["venezuela", "caracas"],
    "vietnam": ["vietnam", "viet nam", "hanoi", "ho chi minh", "saigon", "da nang"],
    "yemen": ["yemen", "sana'a", "aden"],
    "zambia": ["zambia", "lusaka"],
    "zimbabwe": ["zimbabwe", "harare"]
}

# US States
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming"
}

STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
}

# ---------------------------------------------
# Academic vs Practical classifier
# ---------------------------------------------
academic_keywords = [
    "professor", "phd", "msc", "bsc", "doctorate", "researcher",
    "postdoc", "lecturer", "institute", "university", "faculty",
    "academic", "scientist", "lab", "scholar"
]

practical_keywords = [
    "developer", "engineer", "programmer", "software engineer",
    "full stack", "backend", "frontend", "freelancer", "consultant",
    "architect", "founder", "entrepreneur"
]

degree_patterns = {
    "PhD": r"\b(phd|doctorate|dr\.?)\b",
    "MSc": r"\b(msc|m\.sc|master|graduate student)\b",
    "BSc": r"\b(bsc|b\.sc|bachelor)\b",
    "Professor": r"\b(professor|prof\.)\b",
    "Postdoc": r"\b(postdoc|post-doctoral)\b",
    "Student": r"\b(student|undergrad)\b",
    "Researcher": r"\bresearch(er)?\b",
}

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def normalize_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"\s+", " ", text.strip().lower())

def detect_degree(bio: str):
    if not bio:
        return None
    text = bio.lower()
    for degree, pattern in degree_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return degree
    return None

def classify_user_type(bio: str, website: str):
    text = (bio or "").lower() + " " + (website or "").lower()
    academic_score = sum(k in text for k in academic_keywords)
    practical_score = sum(k in text for k in practical_keywords)
    if academic_score > practical_score:
        return "Academic"
    if practical_score > academic_score:
        return "Practical Developer"
    return "Unknown"

def detect_country(location: str, bio: str = "") -> tuple:
    text = normalize_text(location + " " + bio)

    if not text:
        return "unknown", "Unknown"

    # 1. Georgia (USA) explicit
    if re.search(r"\bgeorgia[, ]+usa\b", text):
        return "usa", "North America"

    # 2. US state abbreviation: "City, ST"
    m = re.search(r",\s*([A-Za-z]{2})\b", text)
    if m and m.group(1).upper() in STATE_ABBR:
        return "usa", "North America"

    # 3. City, State pattern
    m = re.search(r"\b([\w\s]+),\s*([\w\s]+)\b", text)
    if m:
        state_part = m.group(2).strip()
        if state_part in US_STATES:
            return "usa", "North America"

    # 4. US states standalone
    for state in US_STATES:
        if state in text:
            return "usa", "North America"

    # 5. Georgia (country) with cities
    if any(city in text for city in ["tbilisi", "batumi"]):
        return "georgia", "Asia"

    # 6. Exact "georgia" alone → country
    if text.strip() == "georgia":
        return "georgia", "Asia"

    # 7. Keyword match
    for country, keywords in COUNTRIES.items():
        if any(kw in text for kw in keywords):
            continent = CONTINENT_MAP.get(country, "Unknown")
            return country, continent

    return "unknown", "Unknown"

# ---------------------------------------------
# SQL Insert (Updated with Continent)
# ---------------------------------------------
insert_sql = """
INSERT INTO users_processed
(Id, Reputation, DisplayName, Location, AboutMe, 
 UserType, Degree, Country, Continent, AccountId, CreationDate, LastAccessDate)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

batch = []
batch_size = 3000

# ---------------------------------------------
# Stream XML efficiently
# ---------------------------------------------
def parse_users(xml_path):
    global batch

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        # Extract fields
        Id = elem.attrib.get("Id")
        Reputation = elem.attrib.get("Reputation")
        DisplayName = elem.attrib.get("DisplayName")
        Location = elem.attrib.get("Location", "")
        AboutMe_raw = elem.attrib.get("AboutMe", "")
        WebsiteUrl = elem.attrib.get("WebsiteUrl", "")
        AccountId = elem.attrib.get("AccountId")
        CreationDate = elem.attrib.get("CreationDate")
        LastAccessDate = elem.attrib.get("LastAccessDate")

        # Clean AboutMe
        AboutMe = normalize_text(AboutMe_raw)

        # Classifications
        user_type = classify_user_type(AboutMe, WebsiteUrl)
        degree = detect_degree(AboutMe)
        country, continent = detect_country(Location)

        batch.append((
            Id, Reputation, DisplayName, Location, AboutMe,
            user_type, degree, country, continent, AccountId, CreationDate, LastAccessDate
        ))

        if len(batch) >= batch_size:
            cursor.executemany(insert_sql, batch)
            conn.commit()
            batch.clear()

        elem.clear()

    # Final batch
    if batch:
        cursor.executemany(insert_sql, batch)
        conn.commit()
        batch.clear()

# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    parse_users("users.xml")
    cursor.close()
    conn.close()
    print("Processing complete.")
