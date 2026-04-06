from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def _add_workspace_venv_site_packages() -> None:
    site_packages = Path(__file__).resolve().parent / ".venv_uv" / "Lib" / "site-packages"
    if not site_packages.exists():
        return

    site_packages_str = str(site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)


_add_workspace_venv_site_packages()

from reports.generate_report import generate_monthly_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the monthly VILC report and send it using Microsoft Graph."
    )
    parser.add_argument("month", help="Month name or number, for example March or 3")
    parser.add_argument("year", help="Year, for example 2026")
    return parser.parse_args()


def _split_recipients(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def _load_settings() -> dict:
    load_dotenv()

    settings = {
        "tenant_id": os.getenv("MS_TENANT_ID", "").strip(),
        "client_id": os.getenv("MS_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("MS_CLIENT_SECRET", "").strip(),
        "sender_email": os.getenv("MS_SENDER_EMAIL", "").strip(),
        "mail_to": os.getenv("OUTLOOK_MAIL_TO", "").strip(),
        "mail_cc": os.getenv("OUTLOOK_MAIL_CC", "").strip(),
        "mail_subject": os.getenv("OUTLOOK_MAIL_SUBJECT", "").strip(),
        "mail_body": os.getenv("OUTLOOK_MAIL_BODY", "").strip(),
    }

    required = {
        "tenant_id": "MS_TENANT_ID",
        "client_id": "MS_CLIENT_ID",
        "client_secret": "MS_CLIENT_SECRET",
        "sender_email": "MS_SENDER_EMAIL",
        "mail_to": "OUTLOOK_MAIL_TO",
    }

    missing = [env_name for key, env_name in required.items() if not settings[key]]
    if missing:
        raise RuntimeError(f"Missing required .env values: {', '.join(missing)}")

    return settings


def _get_graph_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }

    response = requests.post(token_url, data=data, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Token request failed: {response.status_code} {response.text}")

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("No access token returned from Microsoft identity platform.")

    return token


def _build_inline_image_attachment(image_path: str | Path, content_id: str) -> dict:
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    content_type = mime_type or "application/octet-stream"

    with open(image_path, "rb") as f:
        content_bytes = base64.b64encode(f.read()).decode("utf-8")

    return {
        "@odata.type": "#microsoft.graph.fileAttachment",
        "name": image_path.name,
        "contentType": content_type,
        "contentBytes": content_bytes,
        "isInline": True,
        "contentId": content_id,
    }


def _text_to_html_paragraphs(text: str) -> str:
    safe_text = html.escape(text.strip())
    if not safe_text:
        return ""
    paragraphs = [part.strip() for part in safe_text.split("\n\n") if part.strip()]
    return "".join(
        f"<p style='margin:0 0 12px 0; line-height:1.5;'>{p.replace(chr(10), '<br>')}</p>"
        for p in paragraphs
    )


def _build_html_body(month: str, year: str, body_text: str, content_id: str) -> str:
    intro_html = _text_to_html_paragraphs(body_text)

    if not intro_html:
        intro_html = (
            f"<p style='margin:0 0 12px 0; line-height:1.5;'>Hi Team,</p>"
            f"<p style='margin:0 0 12px 0; line-height:1.5;'>"
            f"Please find below the monthly VILC report for {html.escape(str(month))} {html.escape(str(year))}."
            f"</p>"
        )

    return f"""
    <html>
      <body style="margin:0; padding:24px; font-family:Arial, sans-serif; font-size:14px; color:#222222; background-color:#ffffff;">
        <div style="max-width:1200px; margin:0 auto;">
          {intro_html}
          <div style="margin-top:16px; margin-bottom:12px; font-weight:700; font-size:16px;">
            Monthly report
          </div>
          <div style="border:1px solid #d9d9d9; padding:8px; background:#fafafa; text-align:center;">
            <img
              src="cid:{content_id}"
              alt="Monthly VILC report"
              style="display:block; width:100%; max-width:1100px; height:auto; margin:0 auto;"
            />
          </div>
          <p style="margin:16px 0 0 0; line-height:1.5;">Regards,</p>
          <p style="margin:0 0 12px 0; line-height:1.5;">Yajas Menon</p>
        </div>
      </body>
    </html>
    """


def send_email_via_graph(
    access_token: str,
    sender_email: str,
    to_recipients: list[str],
    cc_recipients: list[str],
    subject: str,
    body_text: str,
    image_path: str | Path,
    month: str,
    year: str,
) -> None:
    content_id = "monthly_report_image"
    html_body = _build_html_body(month=month, year=year, body_text=body_text, content_id=content_id)
    url = f"https://graph.microsoft.com/v1.0/users/{sender_email}/sendMail"

    payload = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": html_body,
            },
            "toRecipients": [
                {"emailAddress": {"address": email}}
                for email in to_recipients
            ],
            "ccRecipients": [
                {"emailAddress": {"address": email}}
                for email in cc_recipients
            ],
            "attachments": [
                _build_inline_image_attachment(image_path, content_id)
            ],
        },
        "saveToSentItems": True,
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if response.status_code != 202:
        raise RuntimeError(f"Graph sendMail failed: {response.status_code} {response.text}")


def main() -> int:
    args = _parse_args()
    settings = _load_settings()

    result = generate_monthly_report(month=args.month, year=args.year)

    html_path = Path(result["html"]).resolve()
    png_path = Path(result["png"]).resolve()

    print(f"HTML: {html_path}")
    print(f"PNG: {png_path}")

    subject = settings["mail_subject"] or f"Monthly VILC Report - {args.month} {args.year}"
    body_text = settings["mail_body"] or (
        f"Hi Team,\n\n"
        f"Please find below the monthly VILC report for {args.month} {args.year}."
    )

    token = _get_graph_token(
        tenant_id=settings["tenant_id"],
        client_id=settings["client_id"],
        client_secret=settings["client_secret"],
    )

    send_email_via_graph(
        access_token=token,
        sender_email=settings["sender_email"],
        to_recipients=_split_recipients(settings["mail_to"]),
        cc_recipients=_split_recipients(settings["mail_cc"]),
        subject=subject,
        body_text=body_text,
        image_path=png_path,
        month=args.month,
        year=args.year,
    )

    print("Email sent successfully using Microsoft Graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())