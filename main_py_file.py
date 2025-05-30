"""
Corti AI Support Agent System
Automated first-pass support ticket handling with RAG documentation search
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

# AI and vector search
import anthropic
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

# Web scraping for docs
from bs4 import BeautifulSoup
import aiofiles

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Draft response storage
import sqlite3
from contextlib import asynccontextmanager

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
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "claude-3-sonnet-20240229"
    MAX_CONTEXT_LENGTH = 8000
    CONFIDENCE_THRESHOLD = 0.7

config = Config()

# Pydantic models
class IntercomWebhookPayload(BaseModel):
    type: str
    data: Dict
    created_at: int

class DocumentChunk(BaseModel):
    id: str
    content: str
    url: str
    title: str
    section: str
    embedding: Optional[List[float]] = None

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

# Initialize AI components
class AIComponents:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            name="corti_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.claude_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

ai = AIComponents()

# Documentation scraper and indexer
class DocumentationManager:
    def __init__(self):
        self.scraped_urls = set()
        
    async def scrape_docs(self) -> List[DocumentChunk]:
        """Scrape and process Corti documentation"""
        chunks = []
        
        # Start with main docs page
        urls_to_scrape = [
            f"{config.CORTI_DOCS_BASE_URL}/",
            f"{config.CORTI_DOCS_BASE_URL}/quickstart",
            f"{config.CORTI_DOCS_BASE_URL}/get-access",
            f"{config.CORTI_DOCS_BASE_URL}/languages", 
            f"{config.CORTI_DOCS_BASE_URL}/standard-templates",
            f"{config.CORTI_DOCS_BASE_URL}/real-time-ambient-documentation",
            f"{config.CORTI_DOCS_BASE_URL}/dictation-sdk"
        ]
        
        async with httpx.AsyncClient() as client:
            for url in urls_to_scrape:
                if url in self.scraped_urls:
                    continue
                    
                try:
                    logger.info(f"Scraping: {url}")
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        chunks.extend(await self._process_page(url, response.text))
                        self.scraped_urls.add(url)
                        
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    
                # Be nice to the server
                await asyncio.sleep(1)
        
        return chunks
    
    async def _process_page(self, url: str, html: str) -> List[DocumentChunk]:
        """Extract and chunk content from HTML page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        title = soup.title.string if soup.title else "Corti Documentation"
        
        # Extract main content sections
        chunks = []
        sections = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for i, section in enumerate(sections):
            section_title = section.get_text().strip()
            
            # Get content until next section
            content_parts = []
            for sibling in section.next_siblings:
                if hasattr(sibling, 'name') and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                if hasattr(sibling, 'get_text'):
                    text = sibling.get_text().strip()
                    if text:
                        content_parts.append(text)
            
            content = '\n'.join(content_parts)
            
            if len(content) > 50:  # Only include substantial content
                chunk_id = hashlib.md5(f"{url}#{section_title}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=f"{section_title}\n\n{content}",
                    url=url,
                    title=title,
                    section=section_title
                )
                chunks.append(chunk)
        
        # If no sections found, use full page content
        if not chunks:
            content = soup.get_text()
            if len(content) > 100:
                chunk_id = hashlib.md5(url.encode()).hexdigest()
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=content[:2000],  # Limit chunk size
                    url=url,
                    title=title,
                    section="Main Content"
                )
                chunks.append(chunk)
        
        return chunks
    
    async def index_documentation(self):
        """Scrape docs and build vector index"""
        logger.info("Starting documentation indexing...")
        
        chunks = await self.scrape_docs()
        
        if not chunks:
            logger.warning("No documentation chunks found")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        contents = [chunk.content for chunk in chunks]
        embeddings = ai.embedding_model.encode(contents).tolist()
        
        # Store in ChromaDB
        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {
                "url": chunk.url,
                "title": chunk.title,
                "section": chunk.section
            }
            for chunk in chunks
        ]
        
        ai.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        logger.info(f"Indexed {len(chunks)} documentation chunks")

docs_manager = DocumentationManager()

# RAG search functionality
class RAGSearcher:
    def search_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documentation using vector similarity"""
        query_embedding = ai.embedding_model.encode([query]).tolist()[0]
        
        results = ai.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return search_results

rag_searcher = RAGSearcher()

# AI Support Agent
class SupportAgent:
    def __init__(self):
        self.system_prompt = """
You are the Corti Support Agent, an AI assistant for Corti's healthcare AI platform.

You help users with questions about Corti's API, documentation, and platform features.

Guidelines:
- Always identify yourself as an AI assistant in your response
- Provide helpful, accurate responses based on the documentation provided
- Include relevant links when available
- If you cannot confidently answer from the documentation, say so and suggest escalation
- Be professional, empathetic, and helpful
- For technical issues requiring debugging, acknowledge limitations and escalate
- Keep responses concise but comprehensive

Response format:
- Start with a clear, helpful answer
- Include relevant documentation references
- End with next steps or escalation if needed
"""

    async def generate_response(self, user_message: str, conversation_context: Dict) -> SupportResponse:
        """Generate AI response to support ticket"""
        
        # Search relevant documentation
        search_results = rag_searcher.search_docs(user_message, top_k=3)
        
        # Filter results by confidence
        relevant_docs = [
            result for result in search_results 
            if result['similarity'] > config.CONFIDENCE_THRESHOLD
        ]
        
        if not relevant_docs:
            return SupportResponse(
                message=self._generate_escalation_message(),
                confidence=0.0,
                sources=[],
                should_escalate=True
            )
        
        # Prepare context for LLM
        doc_context = "\n\n".join([
            f"**{doc['metadata']['section']}** (from {doc['metadata']['url']}):\n{doc['content']}"
            for doc in relevant_docs
        ])
        
        # Generate response using Claude
        try:
            response = ai.claude_client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=1000,
                temperature=0.3,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
User Question: {user_message}

Relevant Documentation:
{doc_context}

Please provide a helpful response based on this documentation.
"""
                    }
                ]
            )
            
            ai_response = response.content[0].text
            
            # Calculate confidence based on doc relevance
            avg_similarity = np.mean([doc['similarity'] for doc in relevant_docs])
            
            # Check if we should escalate
            should_escalate = self._should_escalate(user_message, ai_response, avg_similarity)
            
            sources = [doc['metadata']['url'] for doc in relevant_docs]
            
            return SupportResponse(
                message=ai_response,
                confidence=avg_similarity,
                sources=sources,
                should_escalate=should_escalate
            )
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return SupportResponse(
                message=self._generate_error_message(),
                confidence=0.0,
                sources=[],
                should_escalate=True
            )
    
    def _should_escalate(self, user_message: str, ai_response: str, confidence: float) -> bool:
        """Determine if ticket should be escalated to human"""
        escalation_keywords = [
            "bug", "error", "broken", "not working", "issue", "problem",
            "billing", "account", "urgent", "emergency", "security"
        ]
        
        # Low confidence responses should escalate
        if confidence < 0.6:
            return True
            
        # Check for escalation keywords
        message_lower = user_message.lower()
        if any(keyword in message_lower for keyword in escalation_keywords):
            return True
            
        return False
    
    def _generate_escalation_message(self) -> str:
        return """🤖 **Corti Support Agent (AI Assistant)**

I don't have enough information in our documentation to confidently answer your question. Let me connect you with a human agent who can provide better assistance.

A member of our support team will respond shortly!"""
    
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
    
    async def save_draft(self, conversation_id: str, user_message: str, response: SupportResponse, conversation_data: Dict):
        """Save AI draft response for human approval"""
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
        
        logger.info(f"Draft saved for conversation {conversation_id}")
    
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
        }
    
    async def reply_to_conversation(self, conversation_id: str, message: str, sources: List[str] = None):
        """Send AI response to Intercom conversation"""
        
        # Format message with sources
        formatted_message = message
        if sources:
            formatted_message += "\n\n**📚 Relevant Documentation:**\n"
            for source in sources[:3]:  # Limit to 3 sources
                formatted_message += f"• {source}\n"
        
        formatted_message += "\n\n---\n*This is an automated first response based on our documentation. A human agent will review and follow up if additional assistance is needed.*"
        
        payload = {
            "message_type": "comment",
            "type": "admin",
            "body": formatted_message,
            "admin_id": "ai_agent"  # You'll need to create or use an admin ID
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
                    logger.info(f"Successfully replied to conversation {conversation_id}")
                else:
                    logger.error(f"Failed to reply to conversation: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error sending reply to Intercom: {e}")
    
    async def add_tag_to_conversation(self, conversation_id: str, tag: str):
        """Add tag to conversation for tracking"""
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
            except Exception as e:
                logger.error(f"Error adding tag: {e}")

    async def add_internal_note(self, conversation_id: str, note: str):
        """Add internal note for human agents"""
        payload = {
            "message_type": "note",
            "type": "admin", 
            "body": note,
            "admin_id": "ai_agent"
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
                    logger.info(f"Successfully added note to conversation {conversation_id}")
                else:
                    logger.error(f"Failed to add note: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"Error adding note to Intercom: {e}")

intercom_client = IntercomClient()

# Webhook handlers
@app.post("/webhook/intercom")
async def handle_intercom_webhook(
    request: Request, 
    background_tasks: BackgroundTasks
):
    """Handle incoming Intercom webhooks"""
    
    # Verify webhook (optional but recommended)
    # You can implement webhook signature verification here
    
    try:
        payload = await request.json()
        webhook_data = IntercomWebhookPayload(**payload)
        
        # Only process new conversations
        if webhook_data.type == "conversation.created":
            background_tasks.add_task(
                process_new_conversation, 
                webhook_data.data
            )
            
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

async def process_new_conversation(conversation_data: Dict):
    """Process new conversation and generate AI draft response"""
    
    try:
        conversation_id = conversation_data["item"]["id"]
        
        # Extract user message
        source = conversation_data["item"].get("source", {})
        user_message = source.get("body", "")
        
        if not user_message:
            logger.info(f"No user message found in conversation {conversation_id}")
            return
        
        # Clean HTML tags from message
        soup = BeautifulSoup(user_message, 'html.parser')
        clean_message = soup.get_text().strip()
        
        if len(clean_message) < 10:  # Skip very short messages
            return
        
        logger.info(f"Processing conversation {conversation_id}: {clean_message[:100]}...")
        
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
        
        # Optionally add internal note for human agents
        await intercom_client.add_internal_note(
            conversation_id=conversation_id,
            note=f"🤖 AI Draft Ready (Confidence: {response.confidence:.2f})\n\nRecommendation: {'Escalate to human' if response.should_escalate else 'Ready to send'}"
        )
        
        logger.info(f"Draft prepared for conversation {conversation_id} - Escalate: {response.should_escalate}")
        
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")

# Draft approval endpoints
@app.get("/drafts")
async def get_pending_drafts():
    """Get all pending draft responses for human review"""
    try:
        drafts = await draft_manager.get_pending_drafts()
        return {"drafts": drafts, "count": len(drafts)}
    except Exception as e:
        logger.error(f"Error fetching drafts: {e}")
        raise HTTPException(status_code=500, detail="Error fetching drafts")

@app.post("/drafts/{conversation_id}/approve")
async def approve_draft(conversation_id: str, reviewer: str = "human_agent"):
    """Approve and send draft response"""
    try:
        success = await draft_manager.approve_draft(conversation_id, reviewer)
        if success:
            return {"status": "approved", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Draft not found")
    except Exception as e:
        logger.error(f"Error approving draft: {e}")
        raise HTTPException(status_code=500, detail="Error approving draft")

@app.post("/drafts/{conversation_id}/reject")
async def reject_draft(conversation_id: str, reviewer: str = "human_agent"):
    """Reject draft response"""
    try:
        success = await draft_manager.reject_draft(conversation_id, reviewer)
        if success:
            return {"status": "rejected", "conversation_id": conversation_id}
        else:
            raise HTTPException(status_code=404, detail="Draft not found")
    except Exception as e:
        logger.error(f"Error rejecting draft: {e}")
        raise HTTPException(status_code=500, detail="Error rejecting draft")

@app.get("/drafts/{conversation_id}")
async def get_draft_details(conversation_id: str):
    """Get specific draft details"""
    try:
        drafts = await draft_manager.get_pending_drafts()
        draft = next((d for d in drafts if d["conversation_id"] == conversation_id), None)
        
        if not draft:
            raise HTTPException(status_code=404, detail="Draft not found")
            
        return draft
    except Exception as e:
        logger.error(f"Error fetching draft details: {e}")
        raise HTTPException(status_code=500, detail="Error fetching draft details")

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Corti AI Support Agent System",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/test-response")
async def test_ai_response(message: str):
    """Test endpoint for AI response generation"""
    response = await support_agent.generate_response(message, {})
    return response.dict()

@app.post("/reindex-docs")
async def reindex_documentation():
    """Manually trigger documentation reindexing"""
    try:
        await docs_manager.index_documentation()
        return {"status": "success", "message": "Documentation reindexed"}
    except Exception as e:
        logger.error(f"Error reindexing docs: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "ai_model": "loaded",
            "vector_db": "connected",
            "intercom": "configured" if config.INTERCOM_ACCESS_TOKEN else "not_configured"
        }
    }

# Startup tasks
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Corti AI Support Agent System...")
    
    # Index documentation on startup
    try:
        await docs_manager.index_documentation()
        logger.info("Documentation indexing completed")
    except Exception as e:
        logger.error(f"Error during startup indexing: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
