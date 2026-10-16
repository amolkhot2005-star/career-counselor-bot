from dataclasses import dataclass

from services.nlp_processor import build_vocabulary, phrase_matches, phrase_stems, process_text

SKILL_WEIGHT = 2.0
INTEREST_WEIGHT = 1.5
TOPIC_WEIGHT = 1.0
TITLE_WEIGHT = 3.0
PROFILE_FACTOR = 0.5

CATEGORY_WEIGHTS = (
    ("skills", SKILL_WEIGHT),
    ("interests", INTEREST_WEIGHT),
    ("topics", TOPIC_WEIGHT),
)


@dataclass(frozen=True)
class CareerMatch:
    title: str
    score: float
    matched_skills: tuple
    matched_interests: tuple
    matched_topics: tuple
    matched_profile: tuple
    title_matched: bool


@dataclass(frozen=True)
class SkillGap:
    career_title: str
    have: tuple
    missing: tuple
    coverage_percent: int
    projects: tuple


def career_vocabulary(careers):
    phrases = []
    for career in careers:
        phrases.append(career["title"])
        for field_name, _ in CATEGORY_WEIGHTS:
            phrases.extend(career.get(field_name, []))
    return build_vocabulary(phrases)


def score_career(career, stems, profile_stems=frozenset()):
    score = 0.0
    seen_phrases = set()
    matched = {}
    for field_name, weight in CATEGORY_WEIGHTS:
        hits = []
        for phrase in career.get(field_name, []):
            key = phrase.lower()
            if key not in seen_phrases and phrase_matches(phrase, stems):
                hits.append(phrase)
                seen_phrases.add(key)
                score += weight
        matched[field_name] = tuple(hits)
    profile_hits = []
    if profile_stems:
        for field_name, weight in CATEGORY_WEIGHTS:
            for phrase in career.get(field_name, []):
                key = phrase.lower()
                if key not in seen_phrases and phrase_matches(phrase, profile_stems):
                    profile_hits.append(phrase)
                    seen_phrases.add(key)
                    score += weight * PROFILE_FACTOR
    title_stems = phrase_stems(career["title"])
    title_matched = bool(title_stems) and title_stems <= stems
    if title_matched:
        score += TITLE_WEIGHT
    return CareerMatch(
        title=career["title"],
        score=round(score, 2),
        matched_skills=matched["skills"],
        matched_interests=matched["interests"],
        matched_topics=matched["topics"],
        matched_profile=tuple(profile_hits),
        title_matched=title_matched,
    )


def recommend_careers(careers, text, profile_text="", field_text="", top_n=3):
    if not careers:
        return []
    vocabulary = career_vocabulary(careers)
    combined = process_text(f"{text or ''} {profile_text or ''}", vocabulary)
    secondary = process_text(field_text, vocabulary)
    if combined.is_empty and secondary.is_empty:
        return []
    matches = [score_career(career, combined.stems, secondary.stems) for career in careers]
    matches = [match for match in matches if match.score > 0]
    matches.sort(key=lambda match: (-match.score, match.title))
    return matches if top_n is None else matches[:top_n]


def explain_match(match):
    reasons = []
    if match.title_matched:
        reasons.append("you named this career directly")
    if match.matched_skills:
        reasons.append("your skills in " + ", ".join(match.matched_skills))
    if match.matched_interests:
        reasons.append("your interest in " + ", ".join(match.matched_interests))
    if match.matched_topics:
        reasons.append("related topics such as " + ", ".join(match.matched_topics))
    if match.matched_profile:
        reasons.append("your study background (" + ", ".join(match.matched_profile) + ")")
    if not reasons:
        return "No direct keyword overlap was found."
    return "Recommended because of " + "; ".join(reasons) + "."


def find_career(careers, title):
    for career in careers:
        if career["title"] == title:
            return career
    return None


def find_mentioned_career(careers, text):
    stems = process_text(text).stems
    candidates = []
    for career in careers:
        title_stems = phrase_stems(career["title"])
        if title_stems and title_stems <= stems:
            candidates.append((len(title_stems), career))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]["title"]))
    return candidates[0][1]


def analyze_skill_gap(career, skills_text, careers=None):
    vocabulary = career_vocabulary(careers) if careers else career_vocabulary([career])
    processed = process_text(skills_text, vocabulary)
    required = career.get("skills", [])
    have = tuple(skill for skill in required if phrase_matches(skill, processed.stems))
    missing = tuple(skill for skill in required if skill not in have)
    coverage = round(100 * len(have) / len(required)) if required else 0
    return SkillGap(
        career_title=career["title"],
        have=have,
        missing=missing,
        coverage_percent=coverage,
        projects=tuple(career.get("projects", [])),
    )
