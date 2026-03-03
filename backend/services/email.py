import html
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from database import SubscriptionPlan, get_admin_setting
from services.secret_store import decrypt_value

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_pass = os.environ.get("SMTP_PASS", "")
        self.from_email = os.environ.get("FROM_EMAIL", "noreply@tokemizer.com")
        self.enabled = bool(self.smtp_user and self.smtp_pass)

    def send_email(self, to_email: str, subject: str, body: str, html: bool = False):
        # Reload SMTP settings from admin config if available
        self.smtp_host = get_admin_setting("smtp_host", self.smtp_host)
        self.smtp_port = int(get_admin_setting("smtp_port", self.smtp_port))
        self.smtp_user = get_admin_setting("smtp_user", self.smtp_user)
        self.from_email = get_admin_setting("smtp_from_email", self.from_email)
        stored_password = get_admin_setting("smtp_password", None)
        decrypted_password = decrypt_value(stored_password) if stored_password else None
        if decrypted_password is not None:
            self.smtp_pass = decrypted_password
        self.enabled = bool(self.smtp_user and self.smtp_pass)
        if not self.enabled:
            logger.warning(
                f"Email service disabled. Would have sent email to {to_email} with subject: {subject}"
            )
            return

        msg = MIMEMultipart()
        msg["From"] = self.from_email
        msg["To"] = to_email
        msg["Subject"] = subject

        if html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.sendmail(self.from_email, to_email, msg.as_string())
            server.quit()
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")


# Global instance
email_service = EmailService()


def _format_money(cents: int) -> str:
    return f"${cents / 100:.2f}"


def _format_plan_listing(plan: SubscriptionPlan) -> str:
    price_parts: list[str] = []

    if plan.monthly_price_cents > 0:
        price_parts.append(f"{_format_money(plan.monthly_price_cents)}/month")
    elif plan.monthly_price_cents == 0:
        price_parts.append("Free of charge")

    if plan.annual_price_cents:
        price_parts.append(f"{_format_money(plan.annual_price_cents)}/year")

    price_description = " / ".join(price_parts)
    cycle_description = (
        "Monthly renewal cycle, billed at the start of each month."
        if plan.plan_term != "yearly"
        else "Annual renewal cycle, billed once per year."
    )
    quota_description = (
        "Unlimited API calls"
        if plan.monthly_quota < 0
        else f"{plan.monthly_quota:,} calls per month"
    )
    api_key_description = (
        "Unlimited" if plan.max_api_keys < 0 else f"{plan.max_api_keys} API keys"
    )
    rate_description = f"{plan.rate_limit_rpm} requests per minute"

    features = plan.features or ["Access to the full Tokemizer prompt optimizer"]
    escaped_features = "".join(
        f"<li>{html.escape(feature)}</li>" for feature in features
    )

    return f"""
    <p><strong>Plan overview</strong></p>
    <ul>
        <li><strong>Name:</strong> {html.escape(plan.name)}</li>
        <li><strong>Billing:</strong> {price_description}</li>
        <li><strong>Renewal cycle:</strong> {cycle_description}</li>
        <li><strong>Quota:</strong> {quota_description}</li>
        <li><strong>Rate limit:</strong> {rate_description}</li>
        <li><strong>API keys:</strong> {api_key_description}</li>
    </ul>
    <p><strong>Inclusions:</strong></p>
    <ul>
        {escaped_features}
    </ul>
    <p>Manage your subscription anytime from the Tokemizer dashboard.</p>
    """


def send_welcome_email(
    to_email: str, customer_name: str, plan: SubscriptionPlan
) -> None:
    greeting = html.escape(customer_name or "valued customer")
    subject = f"Welcome to Tokemizer — {plan.name} plan activated"
    body = f"""
    <p>Dear {greeting},</p>
    <p>
        Thank you for choosing Tokemizer. Your {html.escape(plan.name)} plan is now live and ready to power your
        prompt optimization workflow. Below are the highlights of the plan you selected, including pricing,
        renewal cycle, and what is included.
    </p>
    {_format_plan_listing(plan)}
    <p>
        If you have any questions, reply to this email or visit the Subscription page in your dashboard for a
        detailed overview.
    </p>
    <p>Sincerely,<br>The Tokemizer Team</p>
    """
    email_service.send_email(to_email, subject, body, html=True)


def send_plan_change_email(
    to_email: str, customer_name: str, old_plan_name: str, plan: SubscriptionPlan
) -> None:
    subject = f"Plan update confirmed — {plan.name}"
    greeting = html.escape(customer_name or "valued customer")
    previous_plan = html.escape(old_plan_name or "your previous plan")
    body = f"""
    <p>Dear {greeting},</p>
    <p>
        This is to confirm that your subscription has transitioned from <strong>{previous_plan}</strong> to the
        <strong>{html.escape(plan.name)}</strong> plan. Below is what you can expect from the new plan, including
        renewals and key inclusions.
    </p>
    {_format_plan_listing(plan)}
    <p>
        If you need to make further adjustments or have questions about billing, our support team is happy to help.
    </p>
    <p>Sincerely,<br>The Tokemizer Team</p>
    """
    email_service.send_email(to_email, subject, body, html=True)


def send_password_reset_email(to_email: str, reset_token: str):
    subject = "Password Reset Request"
    reset_link = (
        f"https://tokemizer.com/reset-password?token={reset_token}"  # Adjust domain
    )
    body = f"""
    <p>You requested a password reset. Click the link below to reset your password:</p>
    <p><a href="{reset_link}">Reset Password</a></p>
    <p>If you didn't request this, you can ignore this email.</p>
    """
    email_service.send_email(to_email, subject, body, html=True)
