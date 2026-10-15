"""CareerPilot AI — Streamlit front end for the multi-agent career mentor."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from llm import PROVIDERS, LLMError, build_llm
from orchestrator import Orchestrator
from tools.resume_parser import extract_resume_text

st.set_page_config(page_title="CareerPilot AI", page_icon="🧭", layout="wide")

ROOT = Path(__file__).parent
BASELINE_ROLES = [r for r in json.loads((ROOT / "data" / "role_baseline.json").read_text()) if not r.startswith("_")]
AGENTS = [
    ("Profile Analyzer", "Reads your resume / skills and rates each skill with evidence"),
    ("Job-Market Agent", "Finds what employers ask for in your target role"),
    ("Skill-Gap Agent", "Matches you against the market and scores readiness"),
    ("Roadmap Agent", "Plans week-by-week learning that fits your hours"),
    ("Project Agent", "Suggests portfolio projects that close your gaps"),
    ("Interview Coach", "Asks role-specific questions and grades your answers"),
    ("Progress Agent", "Reviews finished weeks and triggers re-planning"),
]


def get_secrets():
    try:
        len(st.secrets)  # raises if no secrets file exists
        return st.secrets
    except Exception:
        return None


SECRETS = get_secrets()
ss = st.session_state

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    llm = build_llm(SECRETS)
    if llm is None:
        st.info("No LLM key is configured on the server. Paste a free API key to run the agents. "
                "It is kept only in this browser session.")
        prov = st.selectbox("LLM provider", list(PROVIDERS), format_func=lambda p: PROVIDERS[p]["label"], key="prov")
        key = st.text_input("API key", type="password", key="user_key",
                            help="Gemini: aistudio.google.com/apikey · Groq: console.groq.com/keys")
        if key:
            llm = build_llm(SECRETS, override_provider=prov, override_key=key)
    if llm is not None:
        # keep one client per session so the call counter survives reruns
        sig = (getattr(llm, "provider", "mock"), llm.model, hash(getattr(llm, "api_key", "")))
        if ss.get("llm_sig") == sig:
            llm = ss["llm"]
        else:
            ss["llm"], ss["llm_sig"] = llm, sig
        st.success(f"LLM: {llm.label}")
    adz_id, adz_key = (SECRETS.get("ADZUNA_APP_ID") if SECRETS else None), (SECRETS.get("ADZUNA_APP_KEY") if SECRETS else None)
    st.caption("Live job postings: " + ("**on** (Adzuna API)" if adz_id and adz_key else
                                         "off — market agent uses LLM knowledge + curated baseline"))
    st.divider()
    st.subheader("Agents")
    for name, desc in AGENTS:
        st.markdown(f"**{name}**  \n<small>{desc}</small>", unsafe_allow_html=True)

# ------------------------------------------------------------------ header + input form
st.title("🧭 CareerPilot AI")
st.caption("Multi-agent AI career mentor · SDG 8: Decent Work & Economic Growth · "
           "find what the job market wants, see your gaps, and close them.")

with st.form("profile_form"):
    c1, c2 = st.columns([1, 1])
    with c1:
        resume = st.file_uploader("Upload your resume (PDF or TXT, optional)", type=["pdf", "txt"])
        skills_text = st.text_area("…and/or type your skills, projects, internships",
                                   placeholder="e.g. C++, DSA (300 LeetCode), Java, SQL, Git. Project: hostel booking app in React + Node.",
                                   height=120)
    with c2:
        role_choice = st.selectbox("Target role", BASELINE_ROLES + ["Other (type below)"])
        custom_role = st.text_input("Other role", placeholder="e.g. Android Developer",
                                    disabled=False)
        a, b = st.columns(2)
        year = a.selectbox("Current year", ["1st year", "2nd year", "3rd year", "4th year", "Graduate"], index=2)
        hours = b.slider("Study hours / week", 4, 30, 10)
        location = st.text_input("Preferred job location (optional)", placeholder="e.g. Bengaluru")
    submitted = st.form_submit_button("Run the agents", type="primary", use_container_width=True)

if submitted:
    role = custom_role.strip() if role_choice.startswith("Other") else role_choice
    if llm is None:
        st.error("Add an API key in the sidebar first.")
    elif not role:
        st.error("Type the role you are targeting.")
    else:
        try:
            resume_text = extract_resume_text(resume.getvalue(), resume.name) if resume else ""
            inputs = {"resume_text": resume_text, "skills_text": skills_text.strip(), "target_role": role,
                      "year": year, "hours_per_week": hours, "location": location.strip()}
            if not resume_text and not inputs["skills_text"]:
                raise ValueError("Upload a resume or type your skills first.")
            orch = Orchestrator(llm, adz_id, adz_key)
            state = Orchestrator.new_state(inputs, keep=ss.get("state"))
            with st.status("Agents are working…", expanded=True) as status:
                def on_step(agent_name):
                    status.update(label=f"{agent_name} is working…")
                    status.write(f"▶ {agent_name}")
                orch.run_analysis(state, on_step)
                status.update(label=f"Done · {llm.calls} LLM calls", state="complete", expanded=False)
            ss["state"] = state
        except (LLMError, ValueError) as e:
            st.error(str(e))

state = ss.get("state")
if not state:
    st.info("Fill in your profile and press **Run the agents**. Seven agents will analyse your profile, "
            "research the job market, find your skill gaps, plan your learning, suggest projects and prepare "
            "interview questions.")
    st.stop()

orch = Orchestrator(llm, adz_id, adz_key) if llm else None
gap, market, prof = state["gap"], state["market"], state["profile"]

# ------------------------------------------------------------------ summary
hist = state.get("readiness_history", [gap["readiness"]])
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Job readiness", f"{gap['readiness']}/100",
          delta=(hist[-1] - hist[-2]) if len(hist) > 1 else None)
m2.metric("Skills covered", sum(r["status"] == "have" for r in gap["rows"]))
m3.metric("Partial", sum(r["status"] == "partial" for r in gap["rows"]))
m4.metric("Missing", sum(r["status"] == "missing" for r in gap["rows"]))
m5.metric("Roadmap", f"{len(state.get('roadmap', {}).get('weeks', []))} weeks")
st.markdown(f"**{state['inputs']['target_role']}** — {gap['verdict']}")
if gap.get("advice"):
    st.info(gap["advice"])

VIEWS = ["Skill gap", "Job market", "Roadmap", "Projects", "Interview practice", "Progress", "Agent trace"]
view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
st.divider()

# ---- skill gap
if view == "Skill gap":
    df = pd.DataFrame(gap["rows"])
    df["status"] = df["status"].map({"have": "✅ have", "partial": "🟡 partial", "missing": "❌ missing"})
    st.dataframe(df[["skill", "importance", "status", "covered_by", "your_level", "priority", "reason"]],
                 hide_index=True, use_container_width=True,
                 column_config={"importance": st.column_config.ProgressColumn("importance", min_value=0, max_value=100, format="%d")})
    with st.expander("Your profile as the Profile Analyzer read it"):
        st.write(prof.get("summary", ""))
        st.dataframe(pd.DataFrame(prof["skills"]), hide_index=True, use_container_width=True)
        if prof.get("projects"):
            st.write("**Projects:** " + "; ".join(p.get("name", "") for p in prof["projects"]))

# ---- market
if view == "Job market":
    st.caption(f"Source: {market['source']}")
    if market.get("role_summary"):
        st.write(market["role_summary"])
    mdf = pd.DataFrame(market["skills"])
    st.bar_chart(mdf.set_index("name")["importance"], horizontal=True, height=420)
    st.dataframe(mdf[["name", "importance", "category", "why"]], hide_index=True, use_container_width=True)
    if market.get("postings"):
        st.subheader("Sample of the live postings analysed")
        for p in market["postings"]:
            st.markdown(f"- [{p['title']}]({p['url']}) — {p['company']}, {p['location']}")

# ---- roadmap
if view == "Roadmap":
    rm = state.get("roadmap", {})
    if rm.get("note"):
        st.caption(rm["note"])
    for w in rm.get("weeks", []):
        with st.expander(f"Week {w['week']} · {w['focus']}  (~{w['hours']} h)", expanded=w["week"] <= 2):
            st.markdown("**Skills:** " + ", ".join(w["skills"]))
            st.markdown("**Topics:**\n" + "\n".join(f"- {t}" for t in w["topics"]))
            if w["resources"]:
                st.markdown("**Free resources:** " + "; ".join(w["resources"]))
            st.markdown(f"**Deliverable:** {w['deliverable']}")
    if rm.get("weeks"):
        md = f"# CareerPilot roadmap — {state['inputs']['target_role']}\n\n" + "\n\n".join(
            f"## Week {w['week']}: {w['focus']} (~{w['hours']} h)\n" + "\n".join(f"- {t}" for t in w["topics"])
            + f"\n\nResources: {', '.join(w['resources'])}\n\nDeliverable: {w['deliverable']}" for w in rm["weeks"])
        st.download_button("Download roadmap (.md)", md, file_name="careerpilot_roadmap.md")

# ---- projects
if view == "Projects":
    for p in state.get("projects", {}).get("projects", []):
        with st.container(border=True):
            st.subheader(p.get("title", ""))
            st.caption(f"{p.get('difficulty', '')} · ~{p.get('duration_weeks', '?')} weeks · closes "
                       f"{len(p['gaps_closed'])} gaps: {', '.join(p['gaps_closed']) or '—'}")
            st.write(p.get("description", ""))
            st.markdown("\n".join(f"{i+1}. {s}" for i, s in enumerate(p.get("steps", []))))
            st.markdown(f"**Resume bullet:** _{p.get('resume_bullet', '')}_")

# ---- interview
if view == "Interview practice":
    qs = state.get("interview", {}).get("questions", [])
    if qs:
        labels = {q["id"]: f"[{q['type']}] {q['question']}" for q in qs}
        qid = st.selectbox("Pick a question", list(labels), format_func=lambda k: labels[k])
        q = next(x for x in qs if x["id"] == qid)
        st.markdown(f"### {q['question']}")
        st.caption(f"Skill: {q['skill']}")
        with st.expander("Hint: what a strong answer covers"):
            st.markdown("\n".join(f"- {m}" for m in q["must_cover"]))
        ans = st.text_area("Your answer", key=f"ans_{qid}", height=160)
        if st.button("Get feedback from the Interview Coach", disabled=orch is None):
            try:
                with st.spinner("Interview Coach is grading…"):
                    orch.answer_question(state, qid, ans)
            except (LLMError, ValueError) as e:
                st.error(str(e))
        for att in reversed(state["interview"].get("attempts", {}).get(qid, [])):
            ev = att["eval"]
            with st.container(border=True):
                st.markdown(f"**Score: {ev['score']}/10** — {ev.get('verdict', '')}")
                c1, c2 = st.columns(2)
                c1.markdown("**Covered**\n" + "\n".join(f"- {x}" for x in ev.get("covered", [])))
                c2.markdown("**Missing**\n" + "\n".join(f"- {x}" for x in ev.get("missing", [])))
                if ev.get("mistakes"):
                    st.markdown("**Mistakes:** " + "; ".join(ev["mistakes"]))
                st.markdown("**A stronger answer would:**\n" + "\n".join(f"- {x}" for x in ev.get("better_answer", [])))

# ---- progress
if view == "Progress":
    weeks = state.get("roadmap", {}).get("weeks", [])
    if not weeks:
        st.write("No roadmap weeks to track.")
    else:
        st.write("Tick the weeks you finished and rate how confident you feel. The Progress Agent will "
                 "update your profile and re-plan if needed.")
        done, conf = [], {}
        for w in weeks:
            c1, c2 = st.columns([3, 2])
            if c1.checkbox(f"Week {w['week']}: {w['focus']}", key=f"done_{w['week']}_{w['focus']}"):
                done.append(w["week"])
                conf[w["week"]] = c2.slider("Confidence", 1, 5, 3, key=f"conf_{w['week']}_{w['focus']}")
        notes = st.text_input("Notes for the Progress Agent (optional)", placeholder="e.g. SQL went fast, OS was hard")
        if st.button("Submit progress", type="primary", disabled=orch is None):
            try:
                with st.status("Progress Agent is reviewing…", expanded=True) as status:
                    review = orch.report_progress(state, done, conf, notes,
                                                  on_step=lambda n: status.write(f"▶ {n}"))
                    status.update(label="Progress reviewed", state="complete", expanded=False)
                st.rerun()
            except (LLMError, ValueError) as e:
                st.error(str(e))
    reviews = state.get("progress", {}).get("reviews", [])
    if reviews:
        r = reviews[-1]
        st.subheader("Latest review")
        st.write(r.get("summary", ""))
        st.markdown(f"**Acquired:** {', '.join(r.get('acquired_skills', [])) or '—'}  \n"
                    f"**Still weak:** {', '.join(r.get('still_weak', [])) or '—'}  \n"
                    f"**Pace:** {r.get('pace', '')} · **Next focus:** {r.get('next_focus', '')}")
    if len(hist) > 1:
        st.line_chart(pd.DataFrame({"readiness": hist}), height=220)

# ---- trace
if view == "Agent trace":
    st.caption("Every step the agents took in this session.")
    for d in state.get("decisions", []):
        st.markdown(f"🔀 **Orchestrator:** {d}")
    tdf = pd.DataFrame(state["trace"])
    st.dataframe(tdf[["agent", "action", "tool", "llm_calls", "seconds", "status", "detail"]],
                 hide_index=True, use_container_width=True)
    if llm is not None:
        st.caption(f"LLM calls this session: {llm.calls}" + (f" · tokens: {llm.total_tokens}" if llm.total_tokens else ""))
