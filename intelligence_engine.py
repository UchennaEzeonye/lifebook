import json
import re
from datetime import datetime

# System notification patterns to ignore in exports
SYSTEM_IGNORE_PATTERNS = [
    r"Messages and calls are end-to-end encrypted",
    r"<Media omitted>",
    r"image omitted",
    r"video omitted",
    r"audio omitted",
    r"sticker omitted",
    r"GIF omitted",
    r"You deleted this message",
    r"This message was deleted",
    r"changed the subject to",
    r"changed this group's icon",
    r"created group",
    r"added",
    r"left"
]

def get_valid_text_models(client):
    """
    Queries Groq's live API to find models explicitly available on your API key,
    excluding whisper, safeguard, vision, and embedding models.
    """
    priority_order = [
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    try:
        models_page = client.models.list()
        # Filter out audio, safeguard, vision, and embedding models
        available_ids = [
            m.id for m in models_page.data 
            if not any(tag in m.id.lower() for tag in ["safeguard", "guard", "whisper", "vision", "embed", "orpheus"])
        ]
        
        # Sort available models by priority
        sorted_models = [m for m in priority_order if m in available_ids]
        for m in available_ids:
            if m not in sorted_models:
                sorted_models.append(m)
                
        if sorted_models:
            return sorted_models
    except Exception as e:
        print(f"Model discovery warning: {e}")
    
    # Safe fallback default standard model
    return ["openai/gpt-oss-20b"]


def parse_whatsapp_transcript(file_content: str):
    """Parses WhatsApp export files across iOS, Android, 12h/24h formats."""
    lines = file_content.splitlines()
    structured_data = []
    
    patterns = [
        re.compile(r'^\[(\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s*([^:]+):\s*(.*)$'),
        re.compile(r'^(\d{1,4}[/.-]\d{1,4}[/.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*([^:]+):\s*(.*)$')
    ]

    current_msg = None

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        matched = False
        for pattern in patterns:
            match = pattern.match(line_str)
            if match:
                timestamp_str, sender, message = match.groups()
                sender = sender.strip()
                message = message.strip()

                if any(re.search(pat, message, re.IGNORECASE) for pat in SYSTEM_IGNORE_PATTERNS):
                    matched = True
                    break

                if ',' in timestamp_str:
                    date_part, time_part = timestamp_str.split(',', 1)
                else:
                    date_part, time_part = timestamp_str, ""

                current_msg = {
                    "date": date_part.strip(),
                    "time": time_part.strip(),
                    "sender": sender,
                    "message": message
                }
                structured_data.append(current_msg)
                matched = True
                break

        if not matched and current_msg is not None:
            current_msg["message"] += f"\n{line_str}"

    return structured_data


def compute_transcript_stats(structured_data):
    """Calculates factual timeline metrics."""
    dates = []
    for m in structured_data:
        date_str = m['date'].strip('[]')
        for fmt in ('%m/%d/%y', '%m/%d/%Y', '%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%y', '%d.%m.%Y'):
            try:
                dates.append(datetime.strptime(date_str, fmt))
                break
            except ValueError:
                continue

    total_messages = len(structured_data)
    if dates:
        dates.sort()
        first_msg = dates[0].strftime('%B %d, %Y')
        last_msg = dates[-1].strftime('%B %d, %Y')
        total_span_days = (dates[-1] - dates[0]).days
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        max_gap_days = max(gaps) if gaps else 0
    else:
        first_msg = last_msg = "Unknown"
        total_span_days = max_gap_days = 0

    return {
        "total_messages": total_messages,
        "first_message_date": first_msg,
        "last_message_date": last_msg,
        "span_days": total_span_days,
        "longest_communication_gap_days": max_gap_days
    }


def analyze_relationship(client, chat_transcript, computed_stats):
    """Generates structured JSON profile with rate-limit safety and dynamic fallback."""
    system_prompt = """
    You are the Intelligence Engine for an evidence-based biographical memoir generator.
    Analyze the chat transcript and stats, and return a strict JSON object describing the relationship.
    
    STRICT RULES:
    - Base all findings purely on observable evidence in the transcript.
    - Do not invent non-existent emotions, motivations, or relationship labels.
    - Output MUST be valid JSON matching the exact schema below.

    JSON SCHEMA:
    {
        "connection_description": "Factual description of the overall communication style and pattern",
        "observed_activity": "Description of timeline activity spanning start to end date",
        "key_verbatim_quotes": ["3 to 5 distinct, verbatim quotes from the text"],
        "specific_topics_and_events": ["Concrete events or recurring topics discussed"],
        "turning_points": ["Specific timeline shifts, long gaps, or key moments"],
        "communication_changes": ["Observable changes in pacing or response structure"]
    }
    """

    models = get_valid_text_models(client)
    transcript_sample = chat_transcript[:7000]

    for model in models:
        try:
            print(f"[Intelligence Engine] Trying Groq Model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Computed Stats:\n{json.dumps(computed_stats, indent=2)}\n\nTranscript Snippet:\n{transcript_sample}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
            return json.loads(raw_content)
        except Exception as err:
            print(f"Model {model} failed ({err}), trying next valid model...")

    raise RuntimeError("None of the available Groq models were able to complete the intelligence analysis request.")