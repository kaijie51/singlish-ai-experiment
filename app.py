#& C:\Users\Asus\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run C:\Users\Asus\Downloads\singlish-ai-experiment\app.py

import io
import json
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import streamlit as st
import anthropic
import gspread
import numpy as np
import pypdfium2 as pdfium
from google.oauth2.service_account import Credentials
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from streamlit_drawable_canvas import st_canvas

# ----------------- Configuration & Initialization -----------------
st.set_page_config(page_title="Singlish AI Evaluation Experiment", layout="centered")

# Claude model used for the chat phase.
CLAUDE_MODEL = "claude-opus-5"

# Name of the Google Sheet results are appended to. The sheet must already
# exist and be shared (Editor access) with the service account's client_email
# from your secrets - see README / deployment notes at the bottom of this file.
GOOGLE_SHEET_NAME = "Singlish AI Experiment Results"

# Blank IRB-approved consent form that each participant signs before starting.
CONSENT_TEMPLATE = Path(__file__).parent / "IRB Forms" / "IRB-Tan_Kai_Jie_Template.pdf"

# Signed copies that could not be emailed land here so a consent record is
# never lost just because the network or the mail server was unavailable.
LOCAL_CONSENT_FALLBACK = Path(__file__).parent / "signed_consents"

# Every timestamp the study records - the date signed onto the consent form,
# the results-sheet timestamps - is Singapore time. A bare datetime.now() follows
# the host clock, which is UTC on Streamlit Community Cloud and would date a
# form signed after 8am SGT to the previous day. A fixed offset is exact here:
# Singapore has been UTC+8 with no daylight saving since 1982.
SINGAPORE_TIME = timezone(timedelta(hours=8), "SGT")

# Signed consent forms are emailed to the researcher as PDF attachments.
# Gmail's SMTP takes an App Password, which needs no OAuth consent screen and
# does not expire - unlike an OAuth refresh token, which Google invalidates
# every 7 days for an unpublished app.
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# Geometry of the signature row on page 4 of the consent form, in PDF points
# from the bottom-left of the A4 page. Measured off the rendered template: the
# printed rule sits at y=315.6, so signed content baselines just above it.
CONSENT_SIGNATURE_PAGE = 3
SIGNATURE_ROW_BASELINE = 319.0
NAME_FIELD_X = 40.0
DATE_FIELD_X = 385.0
SIGNATURE_FIELD_BOX = (192.0, 366.0)  # left, right bounds of the "Signature" rule
SIGNATURE_MAX_HEIGHT = 26.0           # keeps a tall signature clear of the paragraph above

# The template ships empty AcroForm fields on the signature row. They are
# dropped from the signed copy so nobody can type over a signature that has
# already been given.
PARTICIPANT_FORM_FIELDS = {"Name of Participant", "Signature", "Date"}

# Column order written to the Google Sheet - kept as a constant so the header
# row and each appended row are guaranteed to line up.
#
# The participant's name is deliberately absent. Under the approved ICF the
# signed consent PDF is the only non-anonymous record; session_id is the sole
# key linking it to these responses, which is what makes a withdrawal request
# actionable without putting names in the results set.
SHEET_HEADER = [
    "session_id", "timestamp", "consent_signed_at", "consent_record",
    "age", "gender", "grew_up_in_singapore",
    "pre_prior_belief", "pre_frequency_singlish",
    "post_naturalness", "post_grammar_syntax", "post_vocabulary_context", "post_overall_opinion",
    "chat_transcript",
]

# Initialize Session State Variables
# Streamlit reruns the whole script on every interaction, so any state that
# must survive between reruns (current phase, chat log, survey answers, etc.)
# has to live in st.session_state instead of a plain local variable.
if "step" not in st.session_state:
    st.session_state.step = "consent"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pre_test_data" not in st.session_state:
    st.session_state.pre_test_data = {}
if "consent" not in st.session_state:
    st.session_state.consent = {}

# ----------------- System Prompt (RISEN format) -----------------
# RISEN = Role, Instructions, Steps, End goal, Narrowing (constraints).
# Structuring the prompt this way keeps the persona, the task, and the hard
# limits clearly separated, which makes the assistant's behaviour easier to
# tune and debug than one big paragraph of rules.
SYSTEM_INSTRUCTION = """
Role:
You are a native Singaporean speaking casually in everyday Singlish, chatting with a friend. I
want you to be friendly, as sometimes the use of discourse particles like "lah", "leh", "lor",
"meh", and "sia" can make you sound aggressive or sarcastic, so use them sparingly and only when 
appropriate.

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
  naturally occur, never stacked or spammed. You don't need to use them in every message.
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


@st.cache_data(show_spinner=False)
def render_consent_preview(scale: float = 1.8) -> list[bytes]:
    """Render the blank consent form to PNGs so participants can read it in the
    browser.

    Streamlit has no native PDF viewer, and Chrome blocks PDFs embedded from
    `data:` iframes, so rasterising the pages is the only display path that
    works for every participant regardless of browser.
    """
    document = pdfium.PdfDocument(str(CONSENT_TEMPLATE))
    try:
        # Without init_forms the filled AcroForm values - the investigator's
        # name and the ticked task boxes - render as blanks.
        document.init_forms()
        pages = []
        for page in document:
            buffer = io.BytesIO()
            page.render(scale=scale).to_pil().convert("RGB").save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        return pages
    finally:
        document.close()


def signature_to_png(image_data) -> bytes | None:
    """Crop the drawable-canvas output to the ink and return it as a PNG.

    Ink is a pixel that is both opaque and darker than the paper. Testing both
    is what makes this independent of the canvas background: on a transparent
    canvas the empty area fails the opacity test, and on the white canvas used
    here it fails the darkness test. Returns None for an untouched canvas.

    The returned PNG is transparent outside the strokes so that stamping it
    onto the consent form does not paint a box over the printed signature rule.
    """
    if image_data is None:
        return None

    pixels = np.asarray(image_data, dtype=np.uint8)
    ink = (pixels[..., 3] > 0) & (pixels[..., :3].mean(axis=2) < 200)
    if not ink.any():
        return None

    rows, columns = np.nonzero(ink)
    top, bottom = rows.min(), rows.max() + 1
    left, right = columns.min(), columns.max() + 1

    cropped = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    cropped[..., :3] = pixels[top:bottom, left:right, :3]
    cropped[..., 3] = np.where(ink[top:bottom, left:right], 255, 0)

    buffer = io.BytesIO()
    Image.fromarray(cropped, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _drop_participant_form_fields(writer: PdfWriter, page) -> None:
    """Remove the empty Name/Signature/Date widgets from the signed page, and
    their now-orphaned entries in the document-wide AcroForm field list.
    """
    def is_participant_field(ref) -> bool:
        return ref.get_object().get("/T") in PARTICIPANT_FORM_FIELDS

    annotations = page.get("/Annots")
    if annotations:
        page[NameObject("/Annots")] = ArrayObject(
            [ref for ref in annotations.get_object() if not is_participant_field(ref)]
        )

    acroform = writer.root_object.get("/AcroForm")
    if acroform is not None:
        acroform = acroform.get_object()
        if "/Fields" in acroform:
            acroform[NameObject("/Fields")] = ArrayObject(
                [ref for ref in acroform["/Fields"].get_object()
                 if not is_participant_field(ref)]
            )


def flatten_pdf(pdf_bytes: bytes) -> bytes:
    """Bake form field values into page content, leaving no interactive fields.

    The template's filled fields - the investigator's name, the ticked task
    boxes - live as AcroForm values that only form-aware viewers draw. Flattening
    turns them into ordinary page content so the archived consent record renders
    identically everywhere and can no longer be edited.
    """
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        document.init_forms()
        for index in range(len(document)):
            # Re-fetch each page by index: flattening invalidates page handles.
            document[index].flatten()
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    finally:
        document.close()


def build_signed_consent_pdf(participant_name: str, signed_on: datetime,
                             signature_png: bytes) -> bytes:
    """Stamp the participant's name, signature, and date onto page 4 of the
    IRB consent template and return the signed PDF as bytes.

    The signature is drawn into the page's content stream rather than added as
    a form field, so the signed record cannot be edited afterwards in a viewer.
    """
    template = PdfReader(str(CONSENT_TEMPLATE))
    page_box = template.pages[CONSENT_SIGNATURE_PAGE].mediabox
    page_size = (float(page_box.width), float(page_box.height))

    overlay_buffer = io.BytesIO()
    overlay = pdf_canvas.Canvas(overlay_buffer, pagesize=page_size)

    overlay.setFont("Helvetica", 10)
    overlay.drawString(NAME_FIELD_X, SIGNATURE_ROW_BASELINE, participant_name)
    overlay.drawString(DATE_FIELD_X, SIGNATURE_ROW_BASELINE, signed_on.strftime("%d %b %Y"))

    signature = ImageReader(io.BytesIO(signature_png))
    source_width, source_height = signature.getSize()
    box_left, box_right = SIGNATURE_FIELD_BOX
    box_width = box_right - box_left
    scale = min(box_width / source_width, SIGNATURE_MAX_HEIGHT / source_height)
    width, height = source_width * scale, source_height * scale
    overlay.drawImage(
        signature,
        box_left + (box_width - width) / 2,
        SIGNATURE_ROW_BASELINE,
        width=width,
        height=height,
        mask="auto",
    )

    overlay.showPage()
    overlay.save()
    overlay_buffer.seek(0)

    writer = PdfWriter(clone_from=template)
    page = writer.pages[CONSENT_SIGNATURE_PAGE]
    page.merge_page(PdfReader(overlay_buffer).pages[0])
    _drop_participant_form_fields(writer, page)

    signed_buffer = io.BytesIO()
    writer.write(signed_buffer)
    return flatten_pdf(signed_buffer.getvalue())


def archive_consent_pdf(pdf_bytes: bytes, filename: str, session_id: str) -> str:
    """Email the signed consent form to the researcher as a PDF attachment.

    Returns a short record of where the form ended up, which is written to the
    results sheet so a failed send is visible in the data rather than silent.
    Falls back to local disk on any failure - a participant must never be
    blocked from taking part, and a consent form must never be discarded.
    """
    try:
        config = get_secret("email")
        if not config:
            raise RuntimeError("[email] is not configured in secrets")

        message = EmailMessage()
        message["Subject"] = f"Signed consent form - participant {session_id}"
        message["From"] = config["sender"]
        message["To"] = config["recipient"]
        message.set_content(
            f"Participant {session_id} signed the consent form on "
            f"{datetime.now(SINGAPORE_TIME).strftime('%d %b %Y at %H:%M')}.\n\n"
            "The signed form is attached."
        )
        message.add_attachment(
            pdf_bytes, maintype="application", subtype="pdf", filename=filename
        )

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(config["sender"], config["app_password"])
            smtp.send_message(message)
        return f"emailed:{config['recipient']}"
    except Exception as exc:  # noqa: BLE001 - any failure must fall back to disk
        failure = f"{type(exc).__name__}: {exc}"[:200]

    LOCAL_CONSENT_FALLBACK.mkdir(exist_ok=True)
    (LOCAL_CONSENT_FALLBACK / filename).write_bytes(pdf_bytes)
    return f"local:{filename} ({failure})"


def save_data_to_gsheet(post_data):
    """Append one participant's pre-test, post-test, and transcript data as a new row."""
    worksheet = get_results_worksheet()
    worksheet.append_row([
        st.session_state.session_id,
        datetime.now(SINGAPORE_TIME).isoformat(),
        st.session_state.consent.get("signed_at"),
        st.session_state.consent.get("record"),
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


# ----------------- Phase 0: Informed Consent -----------------
if st.session_state.step == "consent":
    st.title("Informed Consent")
    st.markdown(
        "Before taking part, please read the study information sheet below and "
        "sign the consent form. Your participation is voluntary and you may stop "
        "at any time."
    )

    st.download_button(
        "Download a copy of the information sheet (PDF)",
        data=CONSENT_TEMPLATE.read_bytes(),
        file_name=CONSENT_TEMPLATE.name,
        mime="application/pdf",
    )

    with st.expander("Read the study information sheet", expanded=True):
        for page_png in render_consent_preview():
            st.image(page_png, width="stretch")

    st.markdown("---")
    st.subheader("Sign to take part")

    participant_name = st.text_input("Your full name (as you would sign it)")
    st.caption(f"Date: {datetime.now(SINGAPORE_TIME).strftime('%d %b %Y')}")

    st.markdown("**Draw your signature in the box below**")
    # An opaque white canvas, not the transparent default: participants whose
    # browser is in dark mode would otherwise be drawing near-black ink onto a
    # dark page and see nothing.
    signature_canvas = st_canvas(
        stroke_width=3,
        stroke_color="#111111",
        background_color="#FFFFFF",
        height=150,
        width=600,
        drawing_mode="freedraw",
        return_image_data=True,
        key="signature_canvas",
    )
    st.caption("Use your mouse, or your finger on a touchscreen. Use the toolbar above the box to undo or clear.")

    agreed = st.checkbox(
        "I have read and understand the information and procedures in the study "
        "information sheet. My questions have been answered to my satisfaction, and "
        "I am participating in this study of my own free will."
    )

    if st.button("Sign and begin the study", type="primary"):
        signature_png = signature_to_png(
            signature_canvas.image_data if signature_canvas is not None else None
        )
        if not participant_name.strip():
            st.warning("Please enter your full name.")
        elif signature_png is None:
            st.warning("Please draw your signature in the box above.")
        elif not agreed:
            st.warning("Please confirm that you agree to take part.")
        else:
            signed_at = datetime.now(SINGAPORE_TIME)
            with st.spinner("Saving your consent form..."):
                signed_pdf = build_signed_consent_pdf(
                    participant_name.strip(), signed_at, signature_png
                )
                record = archive_consent_pdf(
                    signed_pdf,
                    f"consent_{st.session_state.session_id}_"
                    f"{signed_at.strftime('%Y%m%d-%H%M%S')}.pdf",
                    st.session_state.session_id,
                )
            st.session_state.consent = {
                "signed_at": signed_at.isoformat(),
                "record": record,
                "pdf": signed_pdf,
            }
            st.session_state.step = "pre_test"
            st.rerun()

# ----------------- Phase 1: Pre-Test Survey -----------------
elif st.session_state.step == "pre_test":
    st.title("Singlish AI Experiment: Pre-Test")
    if st.session_state.consent.get("pdf"):
        st.success("Thank you - your consent form has been recorded.")
        st.download_button(
            "Download your signed consent form",
            data=st.session_state.consent["pdf"],
            file_name=f"consent_{st.session_state.session_id}.pdf",
            mime="application/pdf",
        )
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
