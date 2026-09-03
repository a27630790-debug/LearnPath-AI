
from google import genai
from google.genai import types

GEMINI_MODEL_NAME = "gemini-3.6-flash"
SUPPORTED_LANGUAGES = ["English", "Urdu", "Roman Urdu"]
MIN_DURATION_WEEKS, MAX_DURATION_WEEKS = 2, 52
MIN_DAILY_MINUTES, MAX_DAILY_MINUTES = 30, 480

print("Configuration loaded successfully.")

# ------------------------------------------------------------
# STEP 2: Load and clean secrets
# ------------------------------------------------------------
# IMPORTANT:
# Secret NAMES must be exactly: GEMINI_API_KEY and GROK_AUTH_TOKEN
# NEVER put the actual key values directly in this code.
# ------------------------------------------------------------

def clean_key(raw_key: str) -> str:
    """Remove whitespace/control characters from the API key."""
    if not raw_key:
        return ""
    return re.sub(r"[\s\r\n\t]+", "", raw_key)


GEMINI_SECRET_NAME = "GEMINI_API_KEY"


def load_gemini_key():
    """Load the Gemini key from Streamlit secrets or environment variables."""
    try:
        if GEMINI_SECRET_NAME in st.secrets:
            return clean_key(st.secrets[GEMINI_SECRET_NAME])
    except Exception:
        pass

    return clean_key(os.environ.get(GEMINI_SECRET_NAME, ""))


API_KEY = load_gemini_key()

if API_KEY:
    print("Gemini API key loaded successfully.")
else:
    print(f"Gemini API key not found. Add '{GEMINI_SECRET_NAME}' in Streamlit Secrets.")
# ------------------------------------------------------------
# STEP 3: Configure the Gemini client + a wrapper for generate_content
# ------------------------------------------------------------

client = None
model = None


class GeminiModelWrapper:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def generate_content(self, contents, generation_config=None):
        cfg = generation_config or {}
        config = types.GenerateContentConfig(
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            max_output_tokens=cfg.get("max_output_tokens", 8192),
            response_mime_type=cfg.get("response_mime_type", "application/json"),
        )
        return self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )


if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        model = GeminiModelWrapper(client, GEMINI_MODEL_NAME)
        print(f"✅ Gemini model '{GEMINI_MODEL_NAME}' configured successfully.")
    except Exception as e:
        print(f"⚠️ Failed to configure Gemini client: {e}")
else:
    print("⚠️ Skipping model configuration — API key missing.")

# ------------------------------------------------------------
# STEP 4: Prompt construction functions
# ------------------------------------------------------------

ROADMAP_JSON_SCHEMA_DESCRIPTION = """
Return ONLY a valid JSON object (no markdown, no extra text) with this exact structure:

{
  "goal_analysis": {
    "main_goal": string,
    "career_direction": string,
    "personalized_focus": string,
    "why_suitable": string
  },
  "skill_analysis": {
    "existing_skills": [string],
    "estimated_proficiency": string,
    "strengths": [string],
    "weak_areas": [string],
    "missing_prerequisites": [string]
  },
  "skill_gap_analysis": [
    {"skill": string, "why_it_matters": string}
  ],
  "time_warning": string or null,
  "phases": [
    {
      "phase_name": string,
      "objective": string,
      "topics": [string],
      "estimated_time": string,
      "prerequisites": [string],
      "practice_tasks": [string],
      "resources": [
        {"name": string, "type": string, "why_useful": string, "difficulty": string, "url": string}
      ],
      "projects": [
        {
          "name": string, "difficulty": string, "objective": string,
          "skills_used": [string], "requirements": [string],
          "steps": [string], "expected_outcome": string, "portfolio_value": string
        }
      ]
    }
  ],
  "weekly_plan": [
    {"period_label": string, "focus": string, "tasks": [string], "estimated_hours": string}
  ],
  "tools_and_technologies": [
    {"name": string, "why_useful": string}
  ],
  "certifications": [
    {"name": string, "provider": string, "is_free": boolean, "why_useful": string}
  ],
  "job_ready_skills": {
    "technical_skills": [string],
    "practical_skills": [string],
    "tools": [string],
    "soft_skills": [string]
  },
  "milestones": [string],
  "ai_mentor_advice": {
    "focus_first": string,
    "learning_advice": string,
    "common_mistakes": [string],
    "recommendations": [string],
    "next_action": string,
    "project_advice": string,
    "motivation": string
  }
}
"""


def build_roadmap_prompt(user_data: dict) -> str:
    """Builds the full prompt for generating a fresh personalized roadmap."""
    prompt = f"""
You are an expert AI career mentor and curriculum designer.

Analyze the following user information carefully and design a personalized
learning roadmap. Do NOT create a generic roadmap — base every part of it
on this specific user's goal, current skills, interests, level, time
availability, duration, and preferred format.

USER GOAL:
{user_data['goal']}

CURRENT SKILLS (self-described):
{user_data['current_skills']}

INTERESTS:
Selected: {', '.join(user_data['selected_interests']) if user_data['selected_interests'] else 'None specified'}
Custom: {user_data['custom_interest'] or 'None'}

PREFERRED LEARNING LEVEL: {user_data['level']}
ROADMAP DURATION: {user_data['duration_weeks']} weeks
DAILY STUDY TIME: {user_data['daily_minutes']} minutes/day
LEARNING FORMAT: {user_data['learning_format']}
OUTPUT LANGUAGE: {user_data['language']}

INSTRUCTIONS:
- Write ALL text content (analysis, roadmap, projects, resources descriptions,
  mentor advice, everything) in {user_data['language']}.
- If Roman Urdu is selected, write using Roman script (Urdu words in English letters),
  not the Urdu script.
- Only recommend genuinely FREE resources. Do not invent URLs — use resources
  you are confident exist (official docs, well-known free platforms, GitHub, etc.).
- If the available time (duration x daily study time) is not realistic for the
  goal, explain this clearly in "time_warning". Otherwise set "time_warning" to null.
- Create as many "phases" as make sense for this specific goal (do not force a fixed count).
- Do not repeat concepts the user says they already know.
- Do not add unnecessary technologies or certifications just to fill space.

{ROADMAP_JSON_SCHEMA_DESCRIPTION}
"""
    return textwrap.dedent(prompt).strip()


def build_modification_prompt(user_data: dict, existing_roadmap: dict, modification_request: str) -> str:
    """Builds a prompt to modify an existing roadmap based on user feedback."""
    prompt = f"""
You are the same AI mentor who created the roadmap below. The user now wants
to MODIFY it. Use the ORIGINAL USER INFO and the EXISTING ROADMAP as context,
and apply the requested change. Do NOT generate an unrelated roadmap from scratch —
keep everything that is still relevant, and only change what the request implies.

ORIGINAL USER INFO:
Goal: {user_data['goal']}
Current Skills: {user_data['current_skills']}
Interests: {', '.join(user_data['selected_interests'])} {user_data['custom_interest']}
Level: {user_data['level']}
Duration: {user_data['duration_weeks']} weeks
Daily Study Time: {user_data['daily_minutes']} minutes/day
Learning Format: {user_data['learning_format']}
Language: {user_data['language']}

EXISTING ROADMAP (JSON):
{json.dumps(existing_roadmap, ensure_ascii=False)}

MODIFICATION REQUEST FROM USER:
"{modification_request}"

Apply this modification and return the FULL updated roadmap in the exact same
JSON structure as before. Write everything in {user_data['language']}.

{ROADMAP_JSON_SCHEMA_DESCRIPTION}
"""
    return textwrap.dedent(prompt).strip()


print("Prompt builder functions ready.")

# ------------------------------------------------------------
# STEP 5: Gemini API call function (with error handling)
# ------------------------------------------------------------

class GeminiCallError(Exception):
    """Custom exception for user-friendly Gemini error messages."""
    pass


def call_gemini(prompt: str, max_retries: int = 2):
    """
    Calls the Gemini API and returns the raw text response.
    Raises GeminiCallError with a friendly message on failure.
    """
    if model is None:
        raise GeminiCallError(
            "Gemini model is not configured. Please check your Gemini API key in Streamlit Secrets."
        )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)

            if not response or not getattr(response, "text", None):
                raise GeminiCallError("Gemini returned an empty response. Please try again.")

            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if "api key" in error_str or "permission" in error_str or "unauthorized" in error_str:
                raise GeminiCallError(
                    "Invalid or unauthorized API key. Please check your Gemini API key in Streamlit Secrets."
                )
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise GeminiCallError(
                    "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
                )
            if "timeout" in error_str or "network" in error_str or "connection" in error_str:
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise GeminiCallError(
                    "Network/timeout error while contacting Gemini. Please check your connection and try again."
                )

            if attempt < max_retries:
                time.sleep(1)
                continue

    raise GeminiCallError(f"Unexpected error while calling Gemini: {last_error}")


print("Gemini API call function ready.")

# ------------------------------------------------------------
# STEP 6: Response validation and JSON parsing
# ------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = [
    "goal_analysis", "skill_analysis", "skill_gap_analysis", "phases",
    "weekly_plan", "tools_and_technologies", "certifications",
    "job_ready_skills", "milestones", "ai_mentor_advice"
]


def extract_json_from_text(raw_text: str):
    """Extracts a JSON object from raw model text, stripping markdown fences if present."""
    text = raw_text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_roadmap(data: dict):
    """Validates that the parsed roadmap JSON has the expected structure."""
    if not isinstance(data, dict):
        return False, "Response was not a valid JSON object."

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing:
        return False, f"Response is missing required sections: {', '.join(missing)}"

    if not isinstance(data.get("phases"), list) or len(data["phases"]) == 0:
        return False, "Roadmap has no learning phases."

    return True, "OK"


def parse_gemini_roadmap(raw_text: str):
    """
    Full pipeline: extract JSON -> validate -> return (roadmap_dict, error_message).
    On failure, roadmap_dict is None and error_message explains what went wrong.
    """
    data = extract_json_from_text(raw_text)
    if data is None:
        return None, "Could not parse Gemini's response as valid JSON. Please try generating again."

    is_valid, message = validate_roadmap(data)
    if not is_valid:
        return None, f"Gemini's response was incomplete: {message}"

    return data, None


print("Response validation/parsing functions ready.")

def roadmap_to_markdown(data: dict, user_data: dict) -> str:
    """Converts the roadmap JSON into a well-structured Markdown document."""
    md = []
    md.append(f"# LearnPath AI — Personalized Roadmap\n")
    md.append(f"**Goal:** {user_data['goal']}\n")

    ga = data.get("goal_analysis", {})
    md.append("## Goal Analysis")
    md.append(f"- **Main Goal:** {ga.get('main_goal','')}")
    md.append(f"- **Career Direction:** {ga.get('career_direction','')}")
    md.append(f"- **Focus:** {ga.get('personalized_focus','')}")
    md.append(f"- **Why This Path:** {ga.get('why_suitable','')}\n")

    sa = data.get("skill_analysis", {})
    md.append("## Current Skill Analysis")
    md.append(f"- **Existing Skills:** {', '.join(sa.get('existing_skills', []))}")
    md.append(f"- **Proficiency:** {sa.get('estimated_proficiency','')}")
    md.append(f"- **Strengths:** {', '.join(sa.get('strengths', []))}")
    md.append(f"- **Weak Areas:** {', '.join(sa.get('weak_areas', []))}\n")

    if data.get("time_warning"):
        md.append(f"> ⚠️ **Time Warning:** {data['time_warning']}\n")

    md.append("## Roadmap Phases")
    for i, phase in enumerate(data.get("phases", []), 1):
        md.append(f"\n### Phase {i}: {phase.get('phase_name','')}")
        md.append(f"**Objective:** {phase.get('objective','')}")
        md.append(f"**Estimated Time:** {phase.get('estimated_time','')}")
        md.append("**Topics:** " + ", ".join(phase.get("topics", [])))
        md.append("\n**Resources:**")
        for r in phase.get("resources", []):
            md.append(f"- [{r.get('name','')}]({r.get('url','')}) — {r.get('why_useful','')}")
        md.append("\n**Projects:**")
        for p in phase.get("projects", []):
            md.append(f"- **{p.get('name','')}** ({p.get('difficulty','')}): {p.get('objective','')}")

    md.append("\n## Weekly Plan")
    for w in data.get("weekly_plan", []):
        md.append(f"- **{w.get('period_label','')}:** {w.get('focus','')} ({w.get('estimated_hours','')})")

    md.append("\n## Milestones")
    for m in data.get("milestones", []):
        md.append(f"- {m}")

    mentor = data.get("ai_mentor_advice", {})
    md.append("\n## AI Mentor Advice")
    md.append(f"- **Focus First:** {mentor.get('focus_first','')}")
    md.append(f"- **Next Action:** {mentor.get('next_action','')}")
    md.append(f"- **Motivation:** {mentor.get('motivation','')}")

    return "\n".join(md)


def roadmap_to_txt(data: dict, user_data: dict) -> str:
    """Converts the roadmap to a plain-text version (strips markdown symbols)."""
    md_text = roadmap_to_markdown(data, user_data)
    txt = re.sub(r"[#*_>\[\]()]", "", md_text)
    return txt

import streamlit as st
import json
import os

st.set_page_config(
    page_title="LearnPath AI",
    page_icon="🚀",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🚀 LearnPath AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Your Personalized AI Learning Roadmap"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

INTEREST_OPTIONS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "NLP",
    "Computer Vision",
    "Data Science",
    "Data Analytics",
    "AI Agents",
    "Robotics",
    "Python",
    "Automation",
    "Software Development",
    "Other"
]

with st.container():

    st.subheader("🎯 Your Goal")

    goal = st.text_area(
        "Describe your career or learning goal",
        height=100,
        placeholder=(
            "e.g. I want to become an AI Engineer "
            "and specialize in AI agents."
        )
    )

    st.subheader("🧠 Current Skills")

    current_skills = st.text_area(
        "Describe what you already know",
        height=100,
        placeholder="e.g. I know basic Python and some Pandas."
    )

    st.subheader("❤️ Your Interests")

    col1, col2 = st.columns(2)

    with col1:
        selected_interests = st.multiselect(
            "Select your interests",
            INTEREST_OPTIONS
        )

    with col2:
        custom_interest = st.text_input(
            "Anything else you're interested in?"
        )

    st.subheader("📚 Learning Level")

    level = st.radio(
        "Choose your preferred explanation level",
        ["Beginner", "Intermediate", "Expert"],
        horizontal=True
    )

    st.subheader("📅 Time & Duration")

    col3, col4 = st.columns(2)

    with col3:
        duration_weeks = st.slider("Roadmap duration (weeks)", 2, 52, 12)

    with col4:
        daily_minutes = st.slider("Daily study time (minutes)", 30, 480, 60, step=15)

    st.subheader("📖 Learning Format")

    learning_format = st.selectbox(
        "Preferred learning format",
        ["Mixed", "Reading / Text", "Hands-on Coding", "Project-Based"]
    )

    st.subheader("🌐 Language")

    language = st.selectbox(
        "Output language",
        ["English", "Urdu", "Roman Urdu"]
    )

    generate_clicked = st.button(
        "✨ Generate Roadmap",
        type="primary",
        use_container_width=True
    )

if "roadmap" not in st.session_state:
    st.session_state.roadmap = None

if "user_data" not in st.session_state:
    st.session_state.user_data = None


def validate_inputs(goal, current_skills):
    if not goal or len(goal.strip()) < 5:
        return False, "Please describe your goal in a bit more detail."
    if not current_skills or len(current_skills.strip()) < 3:
        return False, "Please describe your current skills, even briefly."
    return True, ""


if generate_clicked:

    is_valid, msg = validate_inputs(goal, current_skills)

    if not is_valid:
        st.warning(f"⚠️ {msg}")
    else:
        user_data = {
            "goal": goal.strip(),
            "current_skills": current_skills.strip(),
            "selected_interests": selected_interests,
            "custom_interest": custom_interest.strip(),
            "level": level,
            "duration_weeks": duration_weeks,
            "daily_minutes": daily_minutes,
            "learning_format": learning_format,
            "language": language,
        }

        with st.spinner("Analyzing your profile and building your roadmap..."):
            try:
                prompt = build_roadmap_prompt(user_data)
                raw = call_gemini(prompt)
                roadmap, error = parse_gemini_roadmap(raw)

                if error:
                    st.error(f"⚠️ {error}")
                else:
                    st.session_state.roadmap = roadmap
                    st.session_state.user_data = user_data
                    st.success("✅ Your personalized roadmap is ready!")

            except GeminiCallError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")


if st.session_state.roadmap:

    data = st.session_state.roadmap
    user_data = st.session_state.user_data

    st.divider()
    st.header("🗺️ Personalized Roadmap")

    if data.get("time_warning"):
        st.warning(f"⚠️ {data['time_warning']}")

    with st.expander("📌 Goal Analysis", expanded=True):
        ga = data.get("goal_analysis", {})
        st.write(f"**Career Direction:** {ga.get('career_direction', '')}")
        st.write(f"**Focus:** {ga.get('personalized_focus', '')}")
        st.write(ga.get("why_suitable", ""))

    with st.expander("🧠 Skill Analysis"):
        sa = data.get("skill_analysis", {})
        st.write(f"**Strengths:** {', '.join(sa.get('strengths', []))}")
        st.write(f"**Weak Areas:** {', '.join(sa.get('weak_areas', []))}")

    with st.expander("📊 Skill Gap Analysis"):
        skill_gaps = data.get("skill_gap_analysis", [])
        if skill_gaps:
            for gap in skill_gaps:
                if isinstance(gap, dict):
                    st.markdown(f"- **{gap.get('skill', '')}** — {gap.get('why_it_matters', '')}")
                else:
                    st.markdown(f"- {gap}")
        else:
            st.write("No major skill gaps were identified.")

    for i, phase in enumerate(data.get("phases", []), 1):
        with st.expander(f"📂 Phase {i}: {phase.get('phase_name', '')}"):
            st.write(f"**Objective:** {phase.get('objective', '')}")
            st.write(f"**Estimated Time:** {phase.get('estimated_time', '')}")

            topics = phase.get("topics", [])
            if topics:
                st.write("**Topics:** " + ", ".join(topics))

            st.write("**Resources:**")
            for r in phase.get("resources", []):
                name = r.get("name", "Resource")
                url = r.get("url", "")
                why_useful = r.get("why_useful", "")
                if url:
                    st.markdown(f"- [{name}]({url}) — {why_useful}")
                else:
                    st.markdown(f"- **{name}** — {why_useful}")

            st.write("**Projects:**")
            for p in phase.get("projects", []):
                st.markdown(f"- **{p.get('name', '')}** — {p.get('objective', '')}")

            practice_tasks = phase.get("practice_tasks", [])
            if practice_tasks:
                st.write("**Practice / Exercises:**")
                for task in practice_tasks:
                    st.markdown(f"- {task}")

    with st.expander("📅 Weekly Plan"):
        for w in data.get("weekly_plan", []):
            st.write(f"- **{w.get('period_label', '')}:** {w.get('focus', '')}")

    with st.expander("🛠️ Tools & Technologies"):
        for tool in data.get("tools_and_technologies", []):
            if isinstance(tool, dict):
                st.markdown(f"- **{tool.get('name', '')}** — {tool.get('why_useful', '')}")
            else:
                st.markdown(f"- {tool}")

    with st.expander("💼 Job-Ready Skills"):
        jrs = data.get("job_ready_skills", {})
        st.write("**Technical Skills:** " + ", ".join(jrs.get("technical_skills", [])))
        st.write("**Practical Skills:** " + ", ".join(jrs.get("practical_skills", [])))
        st.write("**Tools:** " + ", ".join(jrs.get("tools", [])))
        st.write("**Soft Skills:** " + ", ".join(jrs.get("soft_skills", [])))

    with st.expander("🏆 Certifications"):
        for cert in data.get("certifications", []):
            if isinstance(cert, dict):
                name = cert.get("name", "Certification")
                free_tag = " (Free)" if cert.get("is_free") else ""
                st.markdown(f"- **{name}**{free_tag} — {cert.get('why_useful', '')}")
            else:
                st.markdown(f"- {cert}")

    with st.expander("🏆 Milestones"):
        for milestone in data.get("milestones", []):
            st.write(f"- {milestone}")

    mentor = data.get("ai_mentor_advice", {})
    st.subheader("🤖 AI Mentor Advice")
    st.info(
        f"**Focus First:** {mentor.get('focus_first', '')}\\n\\n"
        f"**Next Action:** {mentor.get('next_action', '')}\\n\\n"
        f"**Motivation:** {mentor.get('motivation', '')}"
    )

    st.divider()
    st.subheader("🔄 Modify Roadmap")

    modification = st.text_area(
        "How would you like to modify your roadmap?",
        placeholder="e.g. Extend it from 3 months to 6 months."
    )

    if st.button("Apply Modification"):
        if modification.strip():
            with st.spinner("Updating your roadmap..."):
                try:
                    prompt = build_modification_prompt(user_data, data, modification.strip())
                    raw = call_gemini(prompt)
                    new_roadmap, error = parse_gemini_roadmap(raw)

                    if error:
                        st.error(f"⚠️ {error}")
                    else:
                        st.session_state.roadmap = new_roadmap
                        st.rerun()

                except GeminiCallError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"⚠️ Something went wrong: {e}")
        else:
            st.warning("⚠️ Please describe how you'd like to modify the roadmap.")

    st.divider()
    st.subheader("📥 Download")

    colA, colB = st.columns(2)

    with colA:
        st.download_button(
            "Download Markdown",
            roadmap_to_markdown(data, user_data),
            file_name="learnpath_roadmap.md",
            mime="text/markdown"
        )

    with colB:
        st.download_button(
            "Download TXT",
            roadmap_to_txt(data, user_data),
            file_name="learnpath_roadmap.txt",
            mime="text/plain"
        )
