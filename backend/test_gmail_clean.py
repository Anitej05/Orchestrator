#!/usr/bin/env python3
"""Test Gmail agent with clean content extraction"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_clean_email_fetch():
    sys.path.insert(0, 'agents')
    from gmail_mcp_agent import GmailMCPClient
    
    print("=" * 70)
    print("Testing Gmail Agent with Clean Content Extraction")
    print("=" * 70)
    print()
    
    client = GmailMCPClient()
    
    # Test 1: Fetch emails with clean content
    print("Test 1: Fetching unread emails (clean content)")
    print("-" * 70)
    
    result = await client.call_tool(
        "GMAIL_FETCH_EMAILS",
        {
            "query": "is:unread",
            "max_results": 2
        }
    )
    
    if result.get("success"):
        data = result["data"]
        print(f"Found {data.get('count', 0)} email(s)")
        print()
        
        for i, msg in enumerate(data.get("messages", []), 1):
            print(f"Email {i}:")
            print(f"  Subject: {msg.get('subject', 'N/A')}")
            print(f"  From: {msg.get('from', 'N/A')}")
            print(f"  Date: {msg.get('date', 'N/A')}")
            print(f"  Has Attachments: {msg.get('has_attachments', False)}")
            print(f"  Preview: {msg.get('preview', msg.get('snippet', 'N/A'))[:100]}...")
            print()
            
            # Test 2: Get full message with clean body
            if i == 1:  # Test first email
                msg_id = msg.get("id")
                print(f"Test 2: Fetching full message (ID: {msg_id})")
                print("-" * 70)
                
                full_result = await client.call_tool(
                    "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
                    {"message_id": msg_id, "format": "full"}
                )
                
                if full_result.get("success"):
                    full_data = full_result["data"]
                    print(f"Subject: {full_data.get('subject', 'N/A')}")
                    print(f"From: {full_data.get('from', 'N/A')}")
                    print(f"Body Length: {len(full_data.get('body', ''))} characters")
                    print(f"Body Preview: {full_data.get('body', 'N/A')[:300]}...")
                    print()
                    
                    attachments = full_data.get("attachments", [])
                    if attachments:
                        print(f"Attachments ({len(attachments)}):")
                        for att in attachments:
                            print(f"  - {att['filename']} ({att['size']} bytes, {att['type']})")
                        print()
                        
                        # Test 3: Download attachments
                        print("Test 3: Downloading attachments...")
                        print("-" * 70)
                        
                        download_result = await client.download_email_attachments(msg_id)
                        
                        if download_result.get("success"):
                            print(download_result.get("message"))
                            for file_info in download_result.get("files", []):
                                print(f"  Saved: {file_info['path']}")
                        else:
                            print(f"Download failed: {download_result.get('error')}")
                    else:
                        print("No attachments in this email")
                else:
                    print(f"Failed to fetch full message: {full_result.get('error')}")
    else:
        print(f"Failed to fetch emails: {result.get('error')}")
    
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)
    print()
    print("Benefits of clean content extraction:")
    print("  - No base64-encoded HTML bloat")
    print("  - Plain text body only (max 2000 chars)")
    print("  - Attachments saved to disk (not in context)")
    print("  - Concise email summaries for orchestrator")

if __name__ == "__main__":
    asyncio.run(test_clean_email_fetch())
