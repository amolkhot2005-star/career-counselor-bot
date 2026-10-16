from dataclasses import dataclass, field

from services.nlp_processor import tokenize
from services.recommendation_engine import (
    explain_match,
    find_career,
    find_mentioned_career,
    recommend_careers,
)
from services.roadmap_generator import generate_roadmap

MAX_CONTEXT_MESSAGES = 5
TOP_RECOMMENDATIONS = 3

GREETING_WORDS = frozenset({"hi", "hii", "hello", "hey", "greetings", "namaste", "thanks", "thank", "you", "ok", "okay"})
GREETING_FILLER = frozenset({"good", "morning", "afternoon", "evening", "there", "all", "everyone"})
ROADMAP_WORDS = frozenset({"roadmap", "roadmaps", "pathway", "become"})
SKILL_REQUEST_WORDS = frozenset({"need", "needed", "required", "require", "missing", "gap"})

EXAMPLE_QUESTIONS = (
    "I like programming and solving problems.",
    "I am interested in artificial intelligence.",
    "I enjoy working with data and mathematics.",
    "I want a career in cybersecurity.",
    "I like designing websites.",
    "I want a job in cloud computing.",
)

DISCLAIMER = (
    "These suggestions are guidance based only on the details you entered. "
    "They are not a scientific measure of your ideal career."
)

FOLLOW_UPS = (
    "Which technical skills do you already have?",
    "Do you prefer working with data, building products, or designing experiences?",
    "Would you like to see the learning roadmap for {title}?",
    "Which of these careers would you like a skill gap analysis for?",
)


@dataclass
class ConversationContext:
    career_messages: list = field(default_factory=list)
    focus_career: str = None
    turns: int = 0

    def combined_text(self):
        return " ".join(self.career_messages[-MAX_CONTEXT_MESSAGES:])


@dataclass
class BotReply:
    text: str
    matches: list = field(default_factory=list)
    focus_career: str = None
    follow_up: str = ""


def split_profile_text(profile):
    profile = profile or {}
    primary = " ".join(
        str(profile.get(key, "")) for key in ("interests", "skills", "work_area", "goal")
    )
    return primary.strip(), str(profile.get("field_of_study", "")).strip()


def compute_matches(careers, context, profile=None, top_n=TOP_RECOMMENDATIONS):
    profile_text, field_text = split_profile_text(profile)
    return recommend_careers(careers, context.combined_text(), profile_text, field_text, top_n=top_n)


def _example_list():
    return "\n".join(f"- {question}" for question in EXAMPLE_QUESTIONS)


def _is_greeting(tokens):
    words = set(tokens)
    return bool(words) and words <= (GREETING_WORDS | GREETING_FILLER) and bool(words & GREETING_WORDS)


def _asks_for_help(tokens):
    words = set(tokens)
    return "help" in words or {"what", "can", "you", "do"} <= words or {"how", "does", "this", "work"} <= words


def _pick_follow_up(context, profile, title):
    profile = profile or {}
    if not str(profile.get("skills", "")).strip() and context.turns <= 1:
        return FOLLOW_UPS[0]
    template = FOLLOW_UPS[(context.turns + 1) % len(FOLLOW_UPS)]
    return template.format(title=title)


def _roadmap_reply(career):
    stages = generate_roadmap(career)
    lines = [f"Here is a quick overview of the **{career['title']}** roadmap:", ""]
    lines.extend(f"- **{stage.title}**: {stage.steps[0]}" for stage in stages)
    lines += ["", "Open the **Roadmap** tab for every step, or the **Skill Gap** tab to see what you still need to learn."]
    return "\n".join(lines)


def _skills_reply(career):
    skills = ", ".join(career["skills"])
    return (
        f"The **{career['title']}** profile lists these skills, roughly in learning order: {skills}.\n\n"
        "Open the **Skill Gap** tab and enter the skills you already have to see what is missing."
    )


def generate_reply(message, context, careers, profile=None):
    cleaned = message.strip() if isinstance(message, str) else ""
    if not cleaned:
        return BotReply(
            "Your message was empty. Tell me what you enjoy, what you are good at, or which field interests you.\n\n"
            "For example:\n" + _example_list()
        )
    context.turns += 1
    tokens = tokenize(cleaned)
    mentioned = find_mentioned_career(careers, cleaned)
    if mentioned:
        context.focus_career = mentioned["title"]

    if _asks_for_help(tokens):
        text = (
            "I am a career guidance chatbot. Tell me your interests, skills or goals and I will suggest careers, "
            "show your skill gaps and build a learning roadmap. Try one of these:\n\n" + _example_list()
        )
        return BotReply(text, focus_career=context.focus_career)

    if _is_greeting(tokens):
        text = (
            "Hello! I can help you explore careers based on your interests and skills. "
            "Tell me what you enjoy or what you want to learn. For example:\n\n" + _example_list()
        )
        return BotReply(text, focus_career=context.focus_career)

    wants_roadmap = bool(set(tokens) & ROADMAP_WORDS)
    wants_skills = bool({"skill", "skills"} & set(tokens)) and bool(set(tokens) & SKILL_REQUEST_WORDS)
    if wants_roadmap or wants_skills:
        target_title = context.focus_career
        if target_title is None:
            matches = compute_matches(careers, context, profile, top_n=1)
            target_title = matches[0].title if matches else None
        target = find_career(careers, target_title) if target_title else None
        if target is None:
            return BotReply(
                "Which career would you like a roadmap or skill list for? You can name one, such as Data Analyst, "
                "or first tell me about your interests."
            )
        context.focus_career = target["title"]
        text = _roadmap_reply(target) if wants_roadmap else _skills_reply(target)
        return BotReply(text, focus_career=target["title"])

    new_matches = recommend_careers(careers, cleaned, top_n=1)
    if not new_matches:
        text = (
            "I could not find career-related interests or skills in that message. I am designed to help students "
            "explore careers in technology and related fields. You could ask things like:\n\n" + _example_list()
        )
        return BotReply(text, focus_career=context.focus_career)

    context.career_messages.append(cleaned)
    matches = compute_matches(careers, context, profile)
    context.focus_career = mentioned["title"] if mentioned else matches[0].title
    lines = ["Based on what you have told me so far, these careers look like a good starting point:", ""]
    for rank, match in enumerate(matches, start=1):
        lines.append(f"{rank}. **{match.title}** (score {match.score:.1f}): {explain_match(match)}")
    follow_up = _pick_follow_up(context, profile, matches[0].title)
    lines += ["", DISCLAIMER, "", f"**Next question:** {follow_up}"]
    return BotReply("\n".join(lines), matches=matches, focus_career=context.focus_career, follow_up=follow_up)
