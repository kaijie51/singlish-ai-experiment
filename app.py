#& C:\Users\Asus\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run C:\Users\Asus\Downloads\singlish-ai-experiment\app.py

import json
import uuid
from datetime import datetime
import streamlit as st
import anthropic
import gspread
from google.oauth2.service_account import Credentials

# ----------------- Configuration & Initialization -----------------
st.set_page_config(page_title="Singlish AI Evaluation Experiment", layout="centered")

# Claude model used for the chat phase.
CLAUDE_MODEL = "claude-opus-5"

# Name of the Google Sheet results are appended to. The sheet must already
# exist and be shared (Editor access) with the service account's client_email
# from your secrets - see README / deployment notes at the bottom of this file.
GOOGLE_SHEET_NAME = "Singlish AI Experiment Results"

# Column order written to the Google Sheet - kept as a constant so the header
# row and each appended row are guaranteed to line up.
SHEET_HEADER = [
    "session_id", "timestamp", "age", "gender", "grew_up_in_singapore",
    "pre_prior_belief", "pre_frequency_singlish",
    "post_naturalness", "post_grammar_syntax", "post_vocabulary_context", "post_overall_opinion",
    "chat_transcript",
]

# Initialize Session State Variables
# Streamlit reruns the whole script on every interaction, so any state that
# must survive between reruns (current phase, chat log, survey answers, etc.)
# has to live in st.session_state instead of a plain local variable.
if "step" not in st.session_state:
    st.session_state.step = "pre_test"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pre_test_data" not in st.session_state:
    st.session_state.pre_test_data = {}

# ----------------- System Prompt (RISEN format) -----------------
# RISEN = Role, Instructions, Steps, End goal, Narrowing (constraints).
# Structuring the prompt this way keeps the persona, the task, and the hard
# limits clearly separated, which makes the assistant's behaviour easier to
# tune and debug than one big paragraph of rules.
SYSTEM_INSTRUCTION = """
Role:
You are a native Singaporean speaking casually in everyday Singlish, chatting with a friend.

Instructions:
Reply to the user's messages the way an ordinary Singaporean would text or speak in an
informal, friendly conversation. Stay in character as a Singlish speaker for the entire
conversation, regardless of what language the user writes in.

Steps:
1. Read the user's message and identify the topic and tone (casual chat, complaint, food talk, etc.).
2. Compose a reply using natural Singlish topic-comment sentence flow
   (e.g. "That chicken rice stall chilli damn solid", "This monitor where you buy one?").
3. Code-switch naturally between colloquial English and Malay/Hokkien loanwords
   (e.g. chope, dabao, shiok) only where a native speaker would actually use them.
4. Before sending, check the reply is short, direct, and doesn't overuse discourse particles.

End Goal:
Produce responses that a Singaporean reader would recognize as authentic, natural Singlish,
so the conversation can be evaluated for realism as part of a research experiment.

Narrowing (Constraints):
- Use discourse particles (lah, leh, lor, meh, sia) sparingly - only where they would
  naturally occur, never stacked or spammed.
- Keep sentences succinct and use only the most common ~2000 English words plus widely
  recognized Singlish/loanwords. Avoid obscure, overly formal, or textbook-sounding terms.
- Do not break character to explain that you are an AI unless directly and explicitly asked.
"""


def get_secret(key: str) -> str | None:
    """Read a secret from Streamlit's secrets store (used on Streamlit Community
    Cloud and locally via .streamlit/secrets.toml). Returns None if not set.
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_resource(show_spinner=False)
def get_claude_client() -> anthropic.Anthropic:
    """Build an Anthropic client, reading the API key from Streamlit secrets.

    Cached with st.cache_resource so the client (and its HTTP connection
    pool) is created once per app process, not on every rerun.
    """
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY is not set. Add it to .streamlit/secrets.toml locally, "
            "or in the app's Settings -> Secrets on Streamlit Community Cloud."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_results_worksheet():
    """Authenticate with Google Sheets via a service account and return the
    worksheet results are appended to.

    Requires a `[gcp_service_account]` table in secrets.toml holding the
    service account JSON key, and that account must have Editor access to
    GOOGLE_SHEET_NAME. Cached with st.cache_resource so the auth handshake
    only happens once per app process.
    """
    creds_dict = get_secret("gcp_service_account")
    if not creds_dict:
        st.error(
            "Google Sheets credentials are not set. Add a [gcp_service_account] "
            "table to .streamlit/secrets.toml locally, or in the app's "
            "Settings -> Secrets on Streamlit Community Cloud."
        )
        st.stop()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    client = gspread.authorize(credentials)

    try:
        spreadsheet = client.open(GOOGLE_SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        st.error(
            f'Google Sheet "{GOOGLE_SHEET_NAME}" was not found, or is not shared '
            f'with the service account ({credentials.service_account_email}).'
        )
        st.stop()

    worksheet = spreadsheet.sheet1
    if worksheet.acell("A1").value is None:
        worksheet.append_row(SHEET_HEADER)
    return worksheet


def save_data_to_gsheet(post_data):
    """Append one participant's pre-test, post-test, and transcript data as a new row."""
    worksheet = get_results_worksheet()
    worksheet.append_row([
        st.session_state.session_id,
        datetime.now().isoformat(),
        st.session_state.pre_test_data.get("age"),
        st.session_state.pre_test_data.get("gender"),
        st.session_state.pre_test_data.get("grew_up_in_singapore"),
        st.session_state.pre_test_data.get("prior_belief"),
        st.session_state.pre_test_data.get("frequency_singlish"),
        post_data.get("naturalness"),
        post_data.get("grammar_syntax"),
        post_data.get("vocabulary_context"),
        post_data.get("overall_opinion"),
        json.dumps(st.session_state.chat_history),
    ])


# ----------------- Phase 1: Pre-Test Survey -----------------
if st.session_state.step == "pre_test":
    st.title("Singlish AI Experiment: Pre-Test")
    st.markdown("Please answer these quick questions before interacting with the system.")

    with st.form("pre_test_form"):
        age = st.number_input(
            "What is your age?",
            min_value=21, max_value=100, value=21, step=1
        )
        gender = st.radio(
            "What is your gender?",
            options=["Male", "Female"],
        )
        grew_up_in_singapore = st.radio(
            "Did you grow up in Singapore?",
            options=["Yes", "No"],
            help="Growing up in Singapore means having spent at least 10 years of your "
                 "childhood and/or teenage years living in Singapore."
        )
        prior_belief = st.slider(
            "Do you believe current AI models can communicate in natural, authentic Singlish?",
            min_value=1, max_value=5, value=3,
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )
        frequency_singlish = st.slider(
            "How often do you speak or text in Singlish daily?",
            min_value=1, max_value=5, value=4,
            help="1 = Never, 5 = Always"
        )

        submitted = st.form_submit_button("Proceed to Chat")
        if submitted:
            st.session_state.pre_test_data = {
                "age": int(age),
                "gender": gender,
                "grew_up_in_singapore": grew_up_in_singapore,
                "prior_belief": prior_belief,
                "frequency_singlish": frequency_singlish
            }
            st.session_state.step = "chat"
            st.rerun()

# ----------------- Phase 2: Live Chat Interface -----------------
elif st.session_state.step == "chat":
    st.title("Chat with the Assistant")
    st.info("💡 **Task suggestion:** Try planning supper, asking for food recommendations, or complaining about your day in Singlish.")

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    # Chat Input
    if user_prompt := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append({"role": "user", "text": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Call the Claude API. The Messages API is stateless, so the full
        # chat history is resent on every turn, mapped to Anthropic's
        # {"role": ..., "content": ...} message format.
        client = get_claude_client()
        messages_payload = [
            {"role": msg["role"], "content": msg["text"]}
            for msg in st.session_state.chat_history
        ]

        with st.chat_message("assistant"):
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_INSTRUCTION,
                messages=messages_payload,
            )
            # response.content is a list of content blocks; pull out the text ones.
            model_reply = "".join(
                block.text for block in response.content if block.type == "text"
            )
            st.write(model_reply)

        st.session_state.chat_history.append({"role": "assistant", "text": model_reply})

    st.markdown("---")
    if len(st.session_state.chat_history) >= 4:
        if st.button("I'm Done Chatting -> Proceed to Evaluation"):
            st.session_state.step = "post_test"
            st.rerun()
    else:
        st.caption("Exchange at least 2 full turns (4 messages) before moving to the post-test.")

# ----------------- Phase 3: Post-Test Survey -----------------
elif st.session_state.step == "post_test":
    st.title("Singlish AI Experiment: Post-Test Evaluation")
    st.markdown("Evaluate the AI's performance based on your recent conversation.")

    with st.form("post_test_form"):
        naturalness = st.slider(
            "Naturalness: Did the AI sound like a real person or a bot?",
            1, 5, 3,
            help="1 = Completely forced/unnatural, 5 = Very natural"
        )
        grammar_syntax = st.slider(
            "Syntax & Structure: Did it use proper sentence structure and word placement (e.g., correct 'lah'/'leh')?",
            1, 5, 3,
            help="1 = Inaccurate/Awkward placement, 5 = Accurate usage"
        )
        vocabulary_context = st.slider(
            "Vocabulary & Nuance: Was the local slang and cultural context appropriate?",
            1, 5, 3,
            help="1 = Inappropriate/Cringe, 5 = Accurate & Nuanced"
        )
        overall_opinion = st.slider(
            "Final Take: Do you think AI can speak Singlish convincingly?",
            1, 5, 3,
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )

        completed = st.form_submit_button("Submit Experiment Data")
        if completed:
            post_data = {
                "naturalness": naturalness,
                "grammar_syntax": grammar_syntax,
                "vocabulary_context": vocabulary_context,
                "overall_opinion": overall_opinion
            }
            save_data_to_gsheet(post_data)
            st.session_state.step = "complete"
            st.rerun()

# ----------------- Phase 4: Completion Screen -----------------
elif st.session_state.step == "complete":
    st.success("Thank you! Your responses and transcript have been logged successfully.")
    st.markdown(f"**Participant ID:** `{st.session_state.session_id}`")
    st.markdown("Your data has been saved.")
