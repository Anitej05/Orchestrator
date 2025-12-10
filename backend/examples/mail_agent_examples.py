"""
Mail Agent Usage Examples
Demonstrates the enhanced email capabilities
"""

import asyncio
import httpx
from pathlib import Path

MAIL_AGENT_URL = "http://localhost:8040"


async def example_1_simple_html_email():
    """Example 1: Send a simple HTML email"""
    print("\n=== Example 1: Simple HTML Email ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAIL_AGENT_URL}/send_email",
            json={
                "to": ["recipient@example.com"],
                "subject": "Welcome to Our Service!",
                "body": """
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: #4CAF50;">Welcome!</h2>
                    <p>Thank you for signing up. We're excited to have you on board.</p>
                    <p>Get started by exploring our features:</p>
                    <ul>
                        <li>Feature 1: Easy to use</li>
                        <li>Feature 2: Powerful tools</li>
                        <li>Feature 3: Great support</li>
                    </ul>
                    <p>Best regards,<br>The Team</p>
                </body>
                </html>
                """,
                "is_html": True
            }
        )
        print(f"Response: {response.json()}")


async def example_2_professional_template():
    """Example 2: Use professional template for business email"""
    print("\n=== Example 2: Professional Template ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAIL_AGENT_URL}/compose_html_email",
            json={
                "to": ["client@company.com"],
                "subject": "Q4 2024 Business Review",
                "content": """
                Dear Valued Client,
                
                We are pleased to share our Q4 2024 performance highlights:
                
                • Revenue increased by 25% year-over-year
                • Successfully launched 3 new product features
                • Expanded our team by 15 talented professionals
                • Achieved 98% customer satisfaction rating
                
                Looking ahead to 2025, we're committed to delivering even greater value 
                through innovation and exceptional service.
                
                Thank you for your continued partnership.
                
                Best regards,
                John Smith
                CEO
                """,
                "template": "professional",
                "header_text": "Q4 2024 Business Review",
                "footer_text": "© 2024 Your Company | Confidential",
                "cc": ["manager@company.com"]
            }
        )
        print(f"Response: {response.json()}")


async def example_3_newsletter_template():
    """Example 3: Send newsletter with gradient header"""
    print("\n=== Example 3: Newsletter Template ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAIL_AGENT_URL}/compose_html_email",
            json={
                "to": ["subscribers@example.com"],
                "subject": "🎉 New Features Released!",
                "content": """
                Hello Subscribers!
                
                We're thrilled to announce the release of exciting new features:
                
                🚀 Lightning-fast performance improvements
                📱 Mobile app now available on iOS and Android
                🎨 Redesigned user interface with dark mode
                🔒 Enhanced security with two-factor authentication
                
                Update now to experience these improvements!
                
                What's coming next? Stay tuned for our AI-powered recommendations 
                launching next month.
                
                Happy exploring!
                The Product Team
                """,
                "template": "newsletter",
                "header_text": "🎉 New Features Released!",
                "footer_text": "Unsubscribe | Update Preferences | Contact Us"
            }
        )
        print(f"Response: {response.json()}")


async def example_4_email_with_attachments():
    """Example 4: Send email with file attachments"""
    print("\n=== Example 4: Email with Attachments ===")
    
    # Note: This example assumes you have files in these locations
    # Adjust paths according to your setup
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAIL_AGENT_URL}/send_email",
            json={
                "to": ["recipient@example.com"],
                "subject": "Monthly Report - December 2024",
                "body": """
                <html>
                <body>
                    <p>Hi Team,</p>
                    <p>Please find attached the monthly report for December 2024.</p>
                    <p>Key highlights:</p>
                    <ul>
                        <li>Sales exceeded targets by 15%</li>
                        <li>Customer retention at 92%</li>
                        <li>New product launch successful</li>
                    </ul>
                    <p>Let me know if you have any questions.</p>
                    <p>Best,<br>Sarah</p>
                </body>
                </html>
                """,
                "is_html": True,
                "attachment_paths": [
                    "storage/reports/december_report.pdf",
                    "storage/reports/sales_chart.png"
                ]
            }
        )
        print(f"Response: {response.json()}")


async def example_5_markdown_formatting():
    """Example 5: Convert markdown to HTML and send"""
    print("\n=== Example 5: Markdown Formatting ===")
    
    markdown_content = """
# Project Update

## Completed Tasks

We've made **significant progress** this week:

- Implemented user authentication
- Fixed critical bugs in payment system
- Updated documentation

## Next Steps

1. Deploy to staging environment
2. Conduct user acceptance testing
3. Prepare for production release

For more details, visit [our project board](https://project.example.com).

*Note: All tasks are on schedule for the Q1 deadline.*
    """
    
    async with httpx.AsyncClient() as client:
        # First, format the markdown
        format_response = await client.post(
            f"{MAIL_AGENT_URL}/format_email_content",
            json={
                "content": markdown_content,
                "format_type": "markdown"
            }
        )
        
        formatted_html = format_response.json()["result"]["formatted_content"]
        print(f"Formatted HTML preview: {formatted_html[:200]}...")
        
        # Then send the email
        send_response = await client.post(
            f"{MAIL_AGENT_URL}/send_email",
            json={
                "to": ["team@example.com"],
                "subject": "Weekly Project Update",
                "body": formatted_html,
                "is_html": True
            }
        )
        print(f"Send response: {send_response.json()}")


async def example_6_bullet_points():
    """Example 6: Format content as bullet points"""
    print("\n=== Example 6: Bullet Points ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MAIL_AGENT_URL}/format_email_content",
            json={
                "content": """
Increased server capacity by 50%
Reduced page load time by 2 seconds
Implemented automated backups
Enhanced security protocols
Updated SSL certificates
                """.strip(),
                "format_type": "bullet_points"
            }
        )
        
        formatted = response.json()["result"]["formatted_content"]
        print(f"Formatted as bullets: {formatted}")


async def example_7_forward_with_attachments():
    """Example 7: Download attachments and forward them"""
    print("\n=== Example 7: Forward Email with Attachments ===")
    
    async with httpx.AsyncClient() as client:
        # Step 1: Download attachments from an incoming email
        # (Assuming you have a message_id from a previous email)
        message_id = "example_message_id_123"
        
        download_response = await client.post(
            f"{MAIL_AGENT_URL}/download_attachments",
            json={"message_id": message_id}
        )
        
        if download_response.json().get("success"):
            files = download_response.json()["result"]["files"]
            file_ids = [f["file_id"] for f in files]
            
            print(f"Downloaded {len(file_ids)} attachments")
            
            # Step 2: Forward the email with attachments
            forward_response = await client.post(
                f"{MAIL_AGENT_URL}/send_email",
                json={
                    "to": ["colleague@example.com"],
                    "subject": "Fwd: Important Documents",
                    "body": "Hi, forwarding these documents as requested.",
                    "attachment_file_ids": file_ids
                }
            )
            print(f"Forward response: {forward_response.json()}")


async def example_8_list_templates():
    """Example 8: List available email templates"""
    print("\n=== Example 8: List Email Templates ===")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MAIL_AGENT_URL}/email_templates")
        templates = response.json()["result"]["templates"]
        
        print("\nAvailable Templates:")
        for name, info in templates.items():
            print(f"\n{name.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Use Case: {info['use_case']}")


async def main():
    """Run all examples"""
    print("=" * 70)
    print("Mail Agent Enhanced Features - Examples")
    print("=" * 70)
    print("\nNote: These examples demonstrate the API usage.")
    print("To actually send emails, ensure:")
    print("1. Mail agent is running (python backend/agents/mail_agent.py)")
    print("2. Gmail credentials are configured in .env")
    print("3. Update recipient email addresses before running")
    print("=" * 70)
    
    try:
        # Run examples (comment out the ones you don't want to run)
        # await example_1_simple_html_email()
        # await example_2_professional_template()
        # await example_3_newsletter_template()
        # await example_4_email_with_attachments()
        # await example_5_markdown_formatting()
        await example_6_bullet_points()
        await example_8_list_templates()
        
        print("\n" + "=" * 70)
        print("Examples completed!")
        print("=" * 70)
        
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to mail agent.")
        print("Make sure the mail agent is running: python backend/agents/mail_agent.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
