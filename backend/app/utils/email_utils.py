import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from ..config.config import SENDGRID_API_KEY, SENDGRID_FROM_EMAIL
import sys

def send_otp_email(to_email: str, otp: str):
    """
    Sends an OTP email using SendGrid.
    Falls back to printing to the console if SENDGRID_API_KEY is not configured.
    """
    subject = "Your Verification Code"
    content_text = f"Your verification code is: {otp}\n\nThis code will expire in 10 minutes."
    
    if not SENDGRID_API_KEY or SENDGRID_API_KEY == "your_key_here":
        print(f"\n[DEV MODE] SENDGRID NOT CONFIGURED")
        print(f"Would have sent email to: {to_email}")
        print(f"Subject: {subject}")
        print(f"OTP: {otp}\n")
        return True

    try:
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        from_email = Email(SENDGRID_FROM_EMAIL)
        to_email_obj = To(to_email)
        content = Content("text/plain", content_text)
        
        mail = Mail(from_email, to_email_obj, subject, content)
        
        response = sg.client.mail.send.post(request_body=mail.get())
        if response.status_code >= 200 and response.status_code < 300:
            print(f"[SendGrid] Sent OTP to {to_email}")
            return True
        else:
            print(f"[SendGrid] Error sending email: {response.status_code} {response.body}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[SendGrid] Exception sending email: {str(e)}", file=sys.stderr)
        return False
