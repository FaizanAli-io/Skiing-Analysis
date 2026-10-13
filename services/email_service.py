"""SMTP email sending (Gmail App Password by default).

Configure with SMTP_USERNAME and SMTP_PASSWORD (a 16-character Google App
Password). Until those are set, ``is_configured()`` is False and features
that need email report themselves as unavailable instead of failing.
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from html import escape

logger = logging.getLogger(__name__)


def _settings():
    username = os.getenv("SMTP_USERNAME", "").strip()
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com").strip(),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": username,
        "password": os.getenv("SMTP_PASSWORD", "").replace(" ", ""),
        "from_email": os.getenv("SMTP_FROM_EMAIL", username).strip(),
        "from_name": os.getenv("SMTP_FROM_NAME", "Bluerun").strip(),
    }


def is_configured() -> bool:
    settings = _settings()
    return bool(settings["host"] and settings["username"] and settings["password"] and settings["from_email"])


def send_email(to_email: str, subject: str, text_body: str, html_body: str) -> None:
    """Send one email. Raises RuntimeError when sending fails."""
    settings = _settings()
    if not is_configured():
        raise RuntimeError("Email is not configured")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f'{settings["from_name"]} <{settings["from_email"]}>'
    message["To"] = to_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        if settings["port"] == 465:
            with smtplib.SMTP_SSL(settings["host"], settings["port"], context=ssl.create_default_context(), timeout=20) as server:
                server.login(settings["username"], settings["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(settings["host"], settings["port"], timeout=20) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(settings["username"], settings["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to_email, exc)
        raise RuntimeError("Email could not be sent") from exc


def send_code_email(to_email: str, code: str, purpose: str, minutes_valid: int) -> None:
    if purpose == "signup":
        subject = "Your Bluerun verification code"
        intro = "Use this code to verify your email and finish creating your Bluerun account."
    else:
        subject = "Your Bluerun password reset code"
        intro = "Use this code to reset your Bluerun password."
    outro = (
        f"The code expires in {minutes_valid} minutes. "
        "If you did not request it, you can ignore this email."
    )
    text_body = f"{intro}\n\n{code}\n\n{outro}\n\nBluerun"
    html_body = f"""\
<div style="font-family:Arial,Helvetica,sans-serif;background:#151515;padding:32px;color:#ffffff">
  <div style="max-width:460px;margin:0 auto;background:#222222;border:1px solid #383838;border-radius:14px;padding:28px">
    <p style="margin:0 0 6px;color:#8BC0D5;font-weight:700;letter-spacing:.08em;font-size:12px">BLUERUN</p>
    <p style="margin:0 0 20px;font-size:15px;line-height:1.5">{escape(intro)}</p>
    <p style="margin:0 0 20px;font-size:34px;font-weight:800;letter-spacing:.3em;color:#ffffff">{escape(code)}</p>
    <p style="margin:0;color:#8c8c8c;font-size:13px;line-height:1.5">{escape(outro)}</p>
  </div>
</div>"""
    send_email(to_email, subject, text_body, html_body)
