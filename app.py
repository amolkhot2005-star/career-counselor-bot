import html
import json
from datetime import datetime

import streamlit as st

from services.chatbot import EXAMPLE_QUESTIONS, ConversationContext, compute_matches, generate_reply
from services.recommendation_engine import analyze_skill_gap, explain_match, find_career
from services.roadmap_generator import generate_roadmap
from utils.data_loader import CareerDataError, careers_to_dataframe, load_careers
from services.llm_service import LLMConfig, LLMError, load_local_env, stream_chat

EDUCATION_LEVELS = (
    "Not specified",
    "High school",
    "Diploma",
    "Undergraduate (B.E. / B.Tech / B.Sc)",
    "Postgraduate",
    "Other",
)

CSS = """
.block-container{padding-top:1.5rem;max-width:1200px}
.hero{background:linear-gradient(135deg,#1e3a5f 0%,#0f766e 100%);color:#fff;padding:2rem 2.2rem;border-radius:16px;margin-bottom:1.2rem}
.hero h1{color:#fff;font-size:2.1rem;margin:0 0 .4rem 0;padding:0}
.hero p{color:#e6f4f1;margin:0;font-size:1.05rem;max-width:780px;line-height:1.5}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem;margin-bottom:.4rem;box-shadow:0 1px 2px rgba(15,23,42,.06)}
.card-head{display:flex;align-items:center;gap:.7rem;margin-bottom:.4rem}
.rank{background:#0f766e;color:#fff;border-radius:999px;padding:.1rem .6rem;font-weight:600;font-size:.85rem}
.card-title{font-size:1.15rem;font-weight:700;color:#1e3a5f}
.score{margin-left:auto;color:#475569;font-size:.9rem}
.bar{background:#e2e8f0;border-radius:999px;height:8px;margin:.3rem 0 .7rem 0}
.bar-fill{background:#0f766e;height:8px;border-radius:999px}
.muted{color:#475569;font-size:.95rem;margin:.2rem 0}
.chip{display:inline-block;background:#e0f2f1;color:#0f5f58;border-radius:999px;padding:.1rem .65rem;margin:.15rem .25rem .15rem 0;font-size:.85rem}
.chip-missing{background:#fef3c7;color:#92400e}
.table-wrap{overflow-x:auto}
.career-table{border-collapse:collapse;width:100%;font-size:.9rem}
.career-table th{background:#1e3a5f;color:#fff;text-align:left;padding:.5rem .6rem}
.career-table td{border-bottom:1px solid #e2e8f0;padding:.5rem .6rem;vertical-align:top}
.footer{color:#64748b;font-size:.85rem;margin-top:1.5rem}
"""

st.set_page_config(page_title="AI Virtual Career Counsellor", page_icon="🎓", layout="wide")
load_local_env()
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_careers():
    return load_careers()


def escape(value):
    return html.escape(str(value))


def chips(items, css_class="chip"):
    return "".join(f'<span class="{css_class}">{escape(item)}</span>' for item in items)


def init_state(titles):
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("context", ConversationContext())
    st.session_state.setdefault("profile", {})
    st.session_state.setdefault("gap_skills", "")
    st.session_state.setdefault("selected_career", titles[0])
    st.session_state.setdefault("pending_message", None)
    st.session_state.setdefault("response_style", "Balanced")
    st.session_state.setdefault("last_user_message", "")


def queue_message(text):
    st.session_state.pending_message = text


def clear_chat():
    st.session_state.messages = []
    st.session_state.context = ConversationContext()
    st.session_state.last_user_message = ""


def reset_profile():
    st.session_state.profile = {}
    st.session_state.gap_skills = ""


def select_career(title):
    st.session_state.selected_career = title


def process_pending_message(careers):
    text = st.session_state.pending_message
    st.session_state.pending_message = None
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.last_user_message = text
    career_reply = generate_reply(text, st.session_state.context, careers, st.session_state.profile)
    titles = [career["title"] for career in careers]
    if career_reply.focus_career in titles:
        st.session_state.selected_career = career_reply.focus_career
    with st.chat_message("assistant"):
        config = LLMConfig.from_env()
        if config.configured:
            try:
                answer = st.write_stream(stream_chat(
                    st.session_state.messages, config, st.session_state.response_style
                ))
                if not answer:
                    answer = "The model returned an empty response. Please try again."
            except LLMError as error:
                answer = f"{error}"
                st.error(answer)
        else:
            if not career_reply.matches and "empty" not in career_reply.text.lower() and "hello" not in career_reply.text.lower():
                answer = ("The AI model is not configured, so I can only answer career guidance questions offline. "
                          "Set LLM_API_KEY in your environment to enable general questions.")
                st.info(answer)
            else:
                answer = career_reply.text
                st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_sidebar():
    profile = st.session_state.profile
    with st.sidebar:
        st.header("Assistant")
        config = LLMConfig.from_env()
        if config.configured:
            st.success(f"Connected · {config.model}")
        else:
            st.warning("Offline career mode · AI model not configured")
        st.selectbox("Response style", ("Balanced", "Concise", "Detailed", "Beginner-friendly", "Technical"), key="response_style")
        st.caption("Configure LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in your environment, then restart Streamlit.")
        st.divider()
        st.header("Student profile")
        st.caption("Optional. These details improve your recommendations.")
        education_default = profile.get("education_level", EDUCATION_LEVELS[0])
        education_index = EDUCATION_LEVELS.index(education_default) if education_default in EDUCATION_LEVELS else 0
        with st.form("profile_form"):
            education = st.selectbox("Education level", EDUCATION_LEVELS, index=education_index)
            field_of_study = st.text_input("Degree or field of study", value=profile.get("field_of_study", ""))
            interests = st.text_area(
                "Interests", value=profile.get("interests", ""), height=80, placeholder="e.g. data, AI, designing websites"
            )
            skills = st.text_area(
                "Existing skills", value=profile.get("skills", ""), height=80, placeholder="e.g. Python, SQL, Git"
            )
            work_area = st.text_input("Preferred work area", value=profile.get("work_area", ""))
            goal = st.text_input("Career goal", value=profile.get("goal", ""))
            submitted = st.form_submit_button("Save profile", width="stretch")
        if submitted:
            st.session_state.profile = {
                "education_level": education,
                "field_of_study": field_of_study.strip(),
                "interests": interests.strip(),
                "skills": skills.strip(),
                "work_area": work_area.strip(),
                "goal": goal.strip(),
            }
            st.session_state.gap_skills = skills.strip()
            st.success("Profile saved.")
        st.button("Reset profile", on_click=reset_profile, width="stretch")
        st.divider()
        st.caption("Recommendations are guidance based on the information you enter, not a scientific assessment.")


def render_hero():
    st.markdown(
        '<div class="hero"><h1>AI Virtual Career Counsellor</h1>'
        "<p>Tell the chatbot what you enjoy and what you can do. It suggests suitable careers, shows the skills you "
        "still need and builds a step-by-step learning roadmap. Everything runs locally with rule-based NLP, "
        "with no paid APIs.</p></div>",
        unsafe_allow_html=True,
    )


def render_chat():
    st.subheader("Chat with your assistant")
    left, right = st.columns([1, 1])
    left.button("＋ New chat / clear history", on_click=clear_chat, width="stretch")
    payload = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
    right.download_button("Export conversation", payload, file_name=f"conversation-{datetime.now():%Y%m%d-%H%M}.json", mime="application/json", width="stretch", disabled=not st.session_state.messages)
    st.markdown("##### Try a sample question")
    columns = st.columns(3)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        columns[index % 3].button(
            question, key=f"sample_{index}", on_click=queue_message, args=(question,), width="stretch"
        )
    st.button("Clear chat", on_click=clear_chat)
    if not st.session_state.messages:
        st.info(
            "No messages yet. Describe your interests, skills or goals below, or click a sample question above."
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.last_user_message:
        if st.button("Regenerate last answer", key="regenerate"):
            st.session_state.messages = st.session_state.messages[:-2]
            queue_message(st.session_state.last_user_message)
            st.rerun()
    prompt = st.chat_input("Ask anything, or describe your interests and career goals...")
    if prompt:
        queue_message(prompt)
        st.rerun()
    if st.session_state.pending_message:
        process_pending_message(get_careers())
        st.rerun()


def render_recommendations(careers):
    st.subheader("Career recommendations")
    st.caption(f"Currently selected for skill gap and roadmap: **{st.session_state.selected_career}**")
    matches = compute_matches(careers, st.session_state.context, st.session_state.profile, top_n=5)
    if not matches:
        st.info("No recommendations yet. Chat about your interests or save your student profile to see matching careers.")
        return
    top_score = matches[0].score
    for rank, match in enumerate(matches, start=1):
        career = find_career(careers, match.title)
        matched_terms = (
            list(match.matched_skills) + list(match.matched_interests) + list(match.matched_topics) + list(match.matched_profile)
        )
        width = round(100 * match.score / top_score) if top_score else 0
        st.markdown(
            f'<div class="card"><div class="card-head"><span class="rank">#{rank}</span>'
            f'<span class="card-title">{escape(match.title)}</span>'
            f'<span class="score">Score {match.score:.1f}</span></div>'
            f'<div class="bar"><div class="bar-fill" style="width:{width}%"></div></div>'
            f'<p class="muted">{escape(career["description"])}</p>'
            f'<p class="muted">{escape(explain_match(match))}</p>'
            f'<div>{chips(matched_terms)}</div></div>',
            unsafe_allow_html=True,
        )
        st.button(
            f"Use {match.title} for skill gap and roadmap",
            key=f"select_{rank}",
            on_click=select_career,
            args=(match.title,),
        )


def render_skill_gap(careers):
    st.subheader("Skill gap analysis")
    titles = [career["title"] for career in careers]
    st.selectbox("Career to analyse", titles, key="selected_career")
    st.text_area(
        "Skills you already have (comma separated)",
        key="gap_skills",
        height=90,
        placeholder="e.g. Python, SQL, Excel",
    )
    career = find_career(careers, st.session_state.selected_career)
    if not st.session_state.gap_skills.strip():
        st.info("Enter your skills above or in the sidebar profile to compare them with this career.")
        st.markdown("**Skills required for this career, in suggested learning order:**")
        st.markdown(chips(career["skills"], "chip chip-missing"), unsafe_allow_html=True)
        return
    gap = analyze_skill_gap(career, st.session_state.gap_skills, careers)
    st.progress(gap.coverage_percent / 100, text=f"You cover {gap.coverage_percent}% of the listed skills")
    left, right = st.columns(2)
    with left:
        st.markdown("**Skills you already have**")
        if gap.have:
            st.markdown(chips(gap.have), unsafe_allow_html=True)
        else:
            st.caption("None of the listed skills were detected yet.")
    with right:
        st.markdown("**Skills to learn, in suggested order**")
        if gap.missing:
            for position, skill in enumerate(gap.missing, start=1):
                st.markdown(f"{position}. {skill}")
        else:
            st.success("You already cover every listed skill.")
    st.markdown("**Beginner project ideas for practice**")
    for project in gap.projects:
        st.markdown(f"- {project}")


def render_roadmap(careers):
    career = find_career(careers, st.session_state.selected_career)
    st.subheader(f"Learning roadmap: {career['title']}")
    st.caption("Change the career in the Skill Gap tab or pick one from the recommendations.")
    gap = None
    if st.session_state.gap_skills.strip():
        gap = analyze_skill_gap(career, st.session_state.gap_skills, careers)
    for stage in generate_roadmap(career, gap):
        with st.container(border=True):
            st.markdown(f"**{stage.title}**")
            for step in stage.steps:
                st.markdown(f"- {step}")


def render_library(careers):
    st.subheader("Career library")
    st.caption("This table is loaded from data/careers.json. Edit that file to add or change careers.")
    table_html = careers_to_dataframe(careers).to_html(index=False, escape=True, classes="career-table", border=0)
    st.markdown(f'<div class="table-wrap">{table_html}</div>', unsafe_allow_html=True)


def main():
    try:
        careers = get_careers()
    except CareerDataError as error:
        st.error(f"The career data could not be loaded. {error}")
        st.stop()
    init_state([career["title"] for career in careers])
    render_sidebar()
    render_hero()
    chat_tab, recommendations_tab, gap_tab, roadmap_tab, library_tab = st.tabs(
        ["Chat", "Recommendations", "Skill Gap", "Roadmap", "Career Library"]
    )
    with chat_tab:
        render_chat()
    with recommendations_tab:
        render_recommendations(careers)
    with gap_tab:
        render_skill_gap(careers)
    with roadmap_tab:
        render_roadmap(careers)
    with library_tab:
        render_library(careers)
    st.markdown(
        '<p class="footer">Guidance only. Results depend on the information you enter and do not guarantee any job outcome.</p>',
        unsafe_allow_html=True,
    )
main()
