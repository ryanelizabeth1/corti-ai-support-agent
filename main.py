"""
Corti AI Support Agent System - Simplified Version with Debug Logging
Basic support ticket handling with Claude AI
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
import json
import hashlib

# Core dependencies
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel, Field
import uvicorn

# AI
import anthropic

# Web scraping for docs
from bs4 import BeautifulSoup

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Draft response storage
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    INTERCOM_ACCESS_TOKEN = os.getenv("INTERCOM_ACCESS_TOKEN")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    WEBHOOK_SECRET = os.getenv("INTERCOM_WEBHOOK_SECRET")
    CORTI_DOCS_BASE_URL = "https://docs.corti.ai"
    
    # AI Model settings
    LLM_MODEL = "claude-3-sonnet-20240229"

config = Config()

# Pydantic models
class IntercomWebhookPayload(BaseModel):
    type: str
    data: Dict
    created_at: int

class SupportResponse(BaseModel):
    message: str
    confidence: float
    sources: List[str]
    should_escalate: bool

# Initialize FastAPI app
app = FastAPI(title="Corti AI Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Claude client
claude_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None

# Documentation scraper (simplified)
class DocumentationManager:
    def __init__(self):
        self.docs_content = ""
        
    async def fetch_docs(self) -> str:
        """Fetch basic documentation content"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{config.CORTI_DOCS_BASE_URL}/", timeout=10.0)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    self.docs_content = soup.get_text()[:5000]  # First 5000 chars
                    logger.info("Documentation fetched successfully")
                    return self.docs_content
        except Exception as e:
            logger.error(f"Error fetching docs: {e}")
            return "Documentation temporarily unavailable."
        
        return self.docs_content

docs_manager = DocumentationManager()

# AI Support Agent (simplified)
class SupportAgent:
    def __init__(self):
        self.system_prompt = """
You are the Corti Support Agent, an AI assistant for Corti's healthcare AI platform.

You help users with questions about Corti's API, documentation, and platform features.

Guidelines:
- Always identify yourself as an AI assistant in your response
- Provide helpful, accurate responses based on the documentation provided
- If you cannot confidently answer from the documentation, say so and suggest escalation
- Be professional, empathetic, and helpful
- For technical issues requiring debugging, acknowledge limitations and escalate
- Keep responses concise but comprehensive

Response format:
- Start with a clear, helpful answer
- Include relevant documentation references if available
- End with next steps or escalation if needed
"""

    async def generate_response(self, user_message: str, conversation_context: Dict) -> SupportResponse:
        """Generate AI response to support ticket"""
        
        logger.info(f"🤖 Generating response for message: {user_message[:100]}...")
        
        if not claude_client:
            logger.error("❌ Claude client not available")
            return SupportResponse(
                message=self._generate_error_message(),
                confidence=0.0,
                sources=[],
                should_escalate=True
            )
        
        # Get basic documentation context
        doc_context = await docs_manager.fetch_docs()
        logger.info(f"📚 Documentation context length: {len(doc_context)} chars")
        
        # Generate response using Claude
        try:
            logger.info("🧠 Calling Claude API...")
            response = claude_client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=1000,
                temperature=0.3,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
User Question: {user_message}

Available Documentation Context:
{doc_context}

Please provide a helpful response based on this information.
"""
                    }
                ]
            )
            
            ai_response = response.content[0].text
            logger.info(f"✅ Claude response generated: {len(ai_response)} chars")
            
            # Simple confidence scoring based on response length and keywords
            confidence = self._calculate_confidence(user_message, ai_response)
            
            # Check if we should escalate
            should_escalate = self._should_escalate(user_message, ai_response, confidence)
            
            sources = [config.CORTI_DOCS_BASE_URL] if doc_context else []
            
            logger.info(f"📊 Response confidence: {confidence:.2f}, Should escalate: {should_escalate}")
            
            return SupportResponse(
                message=ai_response,
                confidence=confidence,
                sources=sources,
                should_escalate=should_escalate
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating AI response: {e}")
            return SupportResponse(
                message=self._generate_error_message(),
                confidence=0.0,
                sources=[],
                should_escalate=True
            )
    
    def _calculate_confidence(self, user_message: str, ai_response: str) -> float:
        """Simple confidence calculation"""
        # Higher confidence for longer, more detailed responses
        if len(ai_response) > 200 and "documentation" in ai_response.lower():
            return 0.8
        elif len(ai_response) > 100:
            return 0.6
        else:
            return 0.4
    
    def _should_escalate(self, user_message: str, ai_response: str, confidence: float) -> bool:
        """Determine if ticket should be escalated to human"""
        escalation_keywords = [
            "bug", "error", "broken", "not working", "issue", "problem",
            "billing", "account", "urgent", "emergency", "security"
        ]
        
        # Low confidence responses should escalate
        if confidence < 0.5:
            return True
            
        # Check for escalation keywords
        message_lower = user_message.lower()
        if any(keyword in message_lower for keyword in escalation_keywords):
            return True
            
        return False
    
    def _generate_error_message(self) -> str:
        return """🤖 **Corti Support Agent (AI Assistant)**

I'm experiencing a technical issue and can't process your request right now. A human agent will review your message and respond shortly.

Thank you for your patience!"""

support_agent = SupportAgent()

# Draft response storage
class DraftManager:
    def __init__(self):
        self.db_path = "drafts.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for draft storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS draft_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE,
                user_message TEXT,
                ai_response TEXT,
                confidence REAL,
                sources TEXT,
                should_escalate BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                reviewed_at TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("📄 Draft database initialized")
    
    async def save_draft(self, conversation_id: str, user_message: str, response: SupportResponse, conversation_data: Dict):
        """Save AI draft response for human approval"""
        logger.info(f"💾 Saving draft for conversation {conversation_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sources_json = json.dumps(response.sources)
        
        cursor.execute("""
            INSERT OR REPLACE INTO draft_responses 
            (conversation_id, user_message, ai_response, confidence, sources, should_escalate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            user_message,
            response.message,
            response.confidence,
            sources_json,
            response.should_escalate
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Draft saved for conversation {conversation_id}")
    
    async def get_pending_drafts(self) -> List[Dict]:
        """Get all pending draft responses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM draft_responses 
            WHERE status = 'pending' 
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        drafts = []
        for row in rows:
            drafts.append({
                "id": row[0],
                "conversation_id": row[1],
                "user_message": row[2],
                "ai_response": row[3],
                "confidence": row[4],
                "sources": json.loads(row[5]),
                "should_escalate": bool(row[6]),
                "created_at": row[7],
                "status": row[8]
            })
        
        return drafts
    
    async def approve_draft(self, conversation_id: str, reviewer: str) -> bool:
        """Approve and send draft response"""
        logger.info(f"👍 Approving draft for conversation {conversation_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get draft
        cursor.execute("""
            SELECT ai_response, sources FROM draft_responses 
            WHERE conversation_id = ? AND status = 'pending'
        """, (conversation_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
        
        ai_response, sources_json = result
        sources = json.loads(sources_json)
        
        # Send to Intercom
        await intercom_client.reply_to_conversation(
            conversation_id=conversation_id,
            message=ai_response,
            sources=sources
        )
        
        # Update status
        cursor.execute("""
            UPDATE draft_responses 
            SET status = 'approved', reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        """, (reviewer, conversation_id))
        
        conn.commit()
        conn.close()
        
        # Update tags
        await intercom_client.add_tag_to_conversation(conversation_id, "ai-approved")
        
        return True
    
    async def reject_draft(self, conversation_id: str, reviewer: str) -> bool:
        """Reject draft response"""
        logger.info(f"👎 Rejecting draft for conversation {conversation_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE draft_responses 
            SET status = 'rejected', reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
        """, (reviewer, conversation_id))
        
        conn.commit()
        conn.close()
        
        # Update tags
        await intercom_client.add_tag_to_conversation(conversation_id, "ai-rejected")
        
        return True

draft_manager = DraftManager()

async def save_draft_response(conversation_id: str, user_message: str, ai_response: SupportResponse, conversation_data: Dict):
    """Save draft response for human approval"""
    await draft_manager.save_draft(conversation_id, user_message, ai_response, conversation_data)

# Intercom API client
class IntercomClient:
    def __init__(self):
        self.base_url = "https://api.intercom.io"
        self.headers = {
            "Authorization": f"Bearer {config.INTERCOM_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        } if config.INTERCOM_ACCESS_TOKEN else {}
    
    async def reply_to_conversation(self, conversation_id: str, message: str, sources: List[str] = None):
        """Send AI response to Intercom conversation"""
        
        if not config.INTERCOM_ACCESS_TOKEN:
            logger.error("❌ No Intercom access token configured")
            return
        
        # Format message with sources
        formatted_message = message
        if sources:
            formatted_message += "\n\n**📚 Relevant Documentation:**\n"
            for source in sources[:3]:  # Limit to 3 sources
                formatted_message += f"• {source}\n"
        
        formatted_message += "\n\n---\n*This is an automated response based on our documentation. A human agent will review and follow up if additional assistance is needed.*"
        
        payload = {
            "message_type": "comment",
            "type": "admin",
            "body": formatted_message
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/conversations/{conversation_id}/reply",
                    headers=self.headers,
                    json=payload,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Successfully replied to conversation {conversation_id}")
                else:
                    logger.error(f"❌ Failed to reply to conversation: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"❌ Error sending reply to Intercom: {e}")
    
    async def add_tag_to_conversation(self, conversation_id: str, tag: str):
        """Add tag to conversation for tracking"""
        if not config.INTERCOM_ACCESS_TOKEN:
            return
            
        payload = {
            "name": tag
        }
        
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{self.base_url}/conversations/{conversation_id}/tags",
                    headers=self.headers,
                    json=payload
                )
                logger.info(f"🏷️ Added tag '{tag}' to conversation {conversation_id}")
            except Exception as e:
                logger.error(f"❌ Error adding tag: {e}")

intercom_client = IntercomClient()

# Webhook handlers
@app.post("/webhook/intercom")
async def handle_intercom_webhook(
    request: Request, 
    background_tasks: BackgroundTasks
):
    """Handle incoming Intercom webhooks with debug logging"""
    
    try:
        payload = await request.json()
        
        # 🔍 DEBUG: Log the entire payload
        logger.info(f"🔍 WEBHOOK PAYLOAD: {json.dumps(payload, indent=2)}")
        
        # 🔍 DEBUG: Check webhook type
        webhook_type = payload.get("type", "unknown")
        logger.info(f"🔍 WEBHOOK TYPE: {webhook_type}")
        
        # 🔍 DEBUG: Look for message content
        data_item = payload.get("data", {}).get("item", {})
        source = data_item.get("source", {})
        body = source.get("body", "NO BODY FOUND")
        
        logger.info(f"🔍 MESSAGE BODY: {body}")
        logger.info(f"🔍 CONVERSATION ID: {data_item.get('id', 'NO ID FOUND')}")
        
        # Continue with normal processing
        webhook_data = IntercomWebhookPayload(**payload)
        
        if webhook_data.type == "conversation.user.created":
            logger.info("🔍 PROCESSING conversation.user.created")
            background_tasks.add_task(
                process_new_conversation, 
                webhook_data.data
            )
        else:
            logger.info(f"🔍 IGNORING webhook type: {webhook_data.type}")
            
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"❌ WEBHOOK ERROR: {e}")
        logger.error(f"❌ FULL ERROR DETAILS: {str(e)}")
        return {"status": "error", "message": str(e)}

async def process_new_conversation(conversation_data: Dict):
    """Process new conversation and generate AI draft response"""
    
    try:
        logger.info(f"🔄 Processing new conversation data: {json.dumps(conversation_data, indent=2)}")
        
        conversation_id = conversation_data["item"]["id"]
        logger.info(f"📞 Processing conversation {conversation_id}")
        
        # Extract user message
        source = conversation_data["item"].get("source", {})
        user_message = source.get("body", "")
        
        if not user_message:
            logger.warning(f"⚠️ No user message found in conversation {conversation_id}")
            logger.info(f"🔍 Available source keys: {list(source.keys())}")
            return
        
        # Clean HTML tags from message
        soup = BeautifulSoup(user_message, 'html.parser')
        clean_message = soup.get_text().strip()
        
        if len(clean_message) < 10:  # Skip very short messages
            logger.warning(f"⚠️ Message too short ({len(clean_message)} chars): {clean_message}")
            return
        
        logger.info(f"✅ Processing conversation {conversation_id}: {clean_message[:100]}...")
        
        # Generate AI response
        response = await support_agent.generate_response(clean_message, conversation_data)
        
        # DRAFT MODE: Save response for human approval instead of auto-sending
        await save_draft_response(
            conversation_id=conversation_id,
            user_message=clean_message,
            ai_response=response,
            conversation_data=conversation_data
        )
        
        # Add tag to indicate AI draft is ready
        await intercom_client.add_tag_to_conversation(conversation_id, "ai-draft-ready")
        
        logger.info(f"🎉 Draft prepared for conversation {conversation_id} - Escalate: {response.should_escalate}")
        
    except Exception as e:
        logger.error(f"❌ Error processing conversation: {e}")
        logger.error(f"❌ Full error details: {str(e)}")

# Draft approval endpoints
@app.get("/drafts")
async def get_pending_drafts():
    """Get all pending draft responses for human review"""
    try:
        drafts = await draft_manager.get_pending_drafts()
        logger.info(f"📋 Retrieved {len(drafts)} pending drafts")
        return {"drafts": drafts, "count": len(drafts)}
    except Exception as e:
        logger.error(f"❌ Error fetching drafts: {e}")
        raise HTTPException(status_code=500, detail="Error fetching drafts")

@app.post("/drafts/{conversation_id}/approve")
async def approve_draft(conversation_id: str, reviewer: str = "human_agent"):
    """Approve and send draft response"""
    try:
        success = await draft_manager.approve_draft(conversation_id, reviewer)
        if success:
            logger.info(f"✅ Draft approved for conversation {conversation_id}")
            return {"status": "approved", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Draft not found")
    except Exception as e:
        logger.error(f"❌ Error approving draft: {e}")
        raise HTTPException(status_code=500, detail="Error approving draft")

@app.post("/drafts/{conversation_id}/reject")
async def reject_draft(conversation_id: str, reviewer: str = "human_agent"):
    """Reject draft response"""
    try:
        success = await draft_manager.reject_draft(conversation_id, reviewer)
        if success:
            logger.info(f"❌ Draft rejected for conversation {conversation_id}")
            return {"status": "rejected", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Draft not found")
    except Exception as e:
        logger.error(f"❌ Error rejecting draft: {e}")
        raise HTTPException(status_code=500, detail="Error rejecting draft")

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Corti AI Support Agent System - Simplified with Debug",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "claude_api": "configured" if config.ANTHROPIC_API_KEY else "not_configured",
            "intercom": "configured" if config.INTERCOM_ACCESS_TOKEN else "not_configured"
        }
    }

# Startup tasks
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Starting Corti AI Support Agent System (Simplified with Debug)...")
    
    # Fetch basic documentation
    try:
        await docs_manager.fetch_docs()
        logger.info("📚 Basic documentation loaded")
    except Exception as e:
        logger.error(f"❌ Error loading docs: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
