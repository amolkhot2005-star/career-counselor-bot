import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "careers.json"
LIST_FIELDS = ("skills", "interests", "topics", "projects")
ROADMAP_KEYS = ("fundamentals", "core_skills", "projects", "portfolio", "interview")


class CareerDataError(Exception):
    pass


def _clean_string_list(value):
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def validate_career(raw_career):
    if not isinstance(raw_career, dict):
        return None
    title = raw_career.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    description = raw_career.get("description")
    career = {
        "title": title.strip(),
        "description": description.strip() if isinstance(description, str) else "",
    }
    for field_name in LIST_FIELDS:
        career[field_name] = _clean_string_list(raw_career.get(field_name))
    raw_roadmap = raw_career.get("roadmap")
    raw_roadmap = raw_roadmap if isinstance(raw_roadmap, dict) else {}
    career["roadmap"] = {key: _clean_string_list(raw_roadmap.get(key)) for key in ROADMAP_KEYS}
    has_matching_data = career["skills"] or career["interests"] or career["topics"]
    return career if has_matching_data else None


def load_careers(path=None):
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    try:
        with open(data_path, "r", encoding="utf-8") as data_file:
            payload = json.load(data_file)
    except FileNotFoundError as error:
        raise CareerDataError(f"Career data file not found: {data_path.name}") from error
    except (json.JSONDecodeError, OSError) as error:
        raise CareerDataError(f"Career data file could not be read: {error}") from error
    raw_careers = payload.get("careers") if isinstance(payload, dict) else payload
    if not isinstance(raw_careers, list):
        raise CareerDataError("Career data must contain a list of careers.")
    careers = []
    seen_titles = set()
    for raw_career in raw_careers:
        career = validate_career(raw_career)
        if career and career["title"].lower() not in seen_titles:
            seen_titles.add(career["title"].lower())
            careers.append(career)
    if not careers:
        raise CareerDataError("No valid careers were found in the career data file.")
    return careers


def careers_to_dataframe(careers):
    rows = [
        {
            "Career": career["title"],
            "Description": career["description"],
            "Required skills": ", ".join(career["skills"]),
            "Relevant interests": ", ".join(career["interests"]),
            "Beginner projects": "; ".join(career["projects"]),
        }
        for career in careers
    ]
    return pd.DataFrame(rows)
