from dataclasses import dataclass

STAGES = (
    ("fundamentals", "Stage 1: Fundamentals"),
    ("core_skills", "Stage 2: Core technical skills"),
    ("projects", "Stage 3: Practical projects"),
    ("portfolio", "Stage 4: Portfolio development"),
    ("interview", "Stage 5: Interview preparation"),
)
PRIORITY_SKILL_COUNT = 4


@dataclass(frozen=True)
class RoadmapStage:
    title: str
    steps: tuple


def _personalise_core_steps(steps, skill_gap):
    if skill_gap.missing:
        priority = ", ".join(skill_gap.missing[:PRIORITY_SKILL_COUNT])
        return [f"Learn these missing skills first, in this order: {priority}"] + steps
    return ["You already cover every listed skill, so focus on depth and real projects"] + steps


def generate_roadmap(career, skill_gap=None):
    roadmap = career.get("roadmap", {})
    stages = []
    for key, title in STAGES:
        steps = list(roadmap.get(key, []))
        if key == "core_skills" and skill_gap is not None:
            steps = _personalise_core_steps(steps, skill_gap)
        if key == "projects":
            steps += [f"Practice project idea: {project}" for project in career.get("projects", [])[:2]]
        if not steps:
            topics = career.get("topics", [])[:3]
            steps = ["Study: " + ", ".join(topics)] if topics else ["No steps are defined for this stage yet."]
        stages.append(RoadmapStage(title=title, steps=tuple(steps)))
    return stages


def roadmap_to_markdown(career, stages):
    lines = [f"# {career['title']} learning roadmap", ""]
    for stage in stages:
        lines.append(f"## {stage.title}")
        lines.extend(f"- {step}" for step in stage.steps)
        lines.append("")
    return "\n".join(lines)
