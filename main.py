"""
HR AI Agent Backend - Complete Single File Version
Everything in one file to avoid import issues!

Just run: python main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List, Dict, Any, Literal, Tuple
from datetime import datetime, timedelta
import uuid
import json
import random

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application settings"""
    qdrant_url: str
    qdrant_api_key: str
    employee_collection: str = "employee_requests"
    policies_collection: str = "company_policies"
    cases_collection: str = "logged_cases"
    gemini_api_key: str
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug_mode: bool = True
    allowed_origins: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    employee_id: str
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    case_created: bool = False
    case_id: Optional[str] = None
    status: Optional[str] = None
    escalated: bool = False
    confidence_score: Optional[float] = None

class LeaveRequest(BaseModel):
    employee_id: str
    leave_type: Literal["annual_leave", "sick_leave", "emergency_leave"]
    start_date: str
    end_date: str
    reason: Optional[str] = None
    notes: Optional[str] = None

class DocumentRequest(BaseModel):
    employee_id: str
    document_type: Literal["employment_certificate", "salary_certificate", "experience_letter", "tax_document"]
    purpose: Optional[str] = None

class ComplaintRequest(BaseModel):
    employee_id: str
    complaint_type: str
    description: str
    is_urgent: bool = False

class CaseUpdateRequest(BaseModel):
    case_id: str
    status: Literal["approved", "denied", "pending_review", "under_investigation", "completed"]
    hr_notes: Optional[str] = None
    reviewer_id: str

class EmployeeInfo(BaseModel):
    employee_id: str
    full_name: str
    department: str
    job_level: str
    annual_leave_remaining: int
    sick_leave_remaining: int
    remote_days_remaining: int
    probation_period: bool

class CaseDetails(BaseModel):
    case_id: str
    case_number: str
    employee_id: str
    employee_name: str
    request_type: str
    status: str
    created_at: str
    updated_at: str
    escalated: bool
    priority: str
    agent_response: Optional[str] = None
    hr_notes: Optional[str] = None

class CaseListResponse(BaseModel):
    total: int
    cases: List[CaseDetails]
    page: int
    page_size: int

# ============================================================================
# QDRANT SERVICE
# ============================================================================

class QdrantService:
    """Service for all Qdrant database operations"""
    
    def __init__(self, url: str, api_key: str, 
                 employee_collection: str, 
                 policies_collection: str,
                 cases_collection: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.employee_collection = employee_collection
        self.policies_collection = policies_collection
        self.cases_collection = cases_collection
    
    def get_employee_by_id(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Fetch employee data by ID"""
        try:
            results = self.client.scroll(
                collection_name=self.employee_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="employee_id", match=MatchValue(value=employee_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if results[0]:
                return results[0][0].payload
            return None
        except Exception as e:
            print(f"Error fetching employee: {e}")
            return None
    
    def update_leave_balance(self, employee_id: str, leave_type: str, days_to_deduct: int) -> bool:
        """Update employee leave balance"""
        try:
            employee = self.get_employee_by_id(employee_id)
            if not employee:
                return False
            
            balance_field = f"{leave_type}_remaining"
            used_field = f"{leave_type}_used"
            
            employee[balance_field] = employee.get(balance_field, 0) - days_to_deduct
            employee[used_field] = employee.get(used_field, 0) + days_to_deduct
            employee["last_updated"] = datetime.now().isoformat()
            
            results = self.client.scroll(
                collection_name=self.employee_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="employee_id", match=MatchValue(value=employee_id))]
                ),
                limit=1,
                with_vectors=True
            )
            
            if results[0]:
                point_id = results[0][0].id
                vector = results[0][0].vector
                self.client.upsert(
                    collection_name=self.employee_collection,
                    points=[PointStruct(id=point_id, vector=vector, payload=employee)]
                )
                return True
            return False
        except Exception as e:
            print(f"Error updating balance: {e}")
            return False
    
    def search_policies(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Semantic search for relevant policy sections"""
        try:
            results = self.client.search(
                collection_name=self.policies_collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            return [{
                "policy_id": r.payload.get("policy_id"),
                "section_id": r.payload.get("section_id"),
                "section_title": r.payload.get("section_title"),
                "section_content": r.payload.get("section_content"),
                "keywords": r.payload.get("keywords", []),
                "relevance_score": r.score
            } for r in results]
        except Exception as e:
            print(f"Error searching policies: {e}")
            return []
    
    def create_case(self, case_data: Dict[str, Any]) -> Optional[str]:
        """Create a new case"""
        try:
            case_id = str(uuid.uuid4())
            case_number = f"CASE_{datetime.now().year}_{str(random.randint(1, 999)).zfill(3)}"
            
            case_data.update({
                "case_id": case_id,
                "case_number": case_number,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            dummy_vector = [0.1] * 384
            next_id = random.randint(10000, 99999)
            
            point = PointStruct(id=next_id, vector=dummy_vector, payload=case_data)
            self.client.upsert(collection_name=self.cases_collection, points=[point])
            
            return case_id
        except Exception as e:
            print(f"Error creating case: {e}")
            return None
    
    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Fetch case by ID"""
        try:
            results = self.client.scroll(
                collection_name=self.cases_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if results[0]:
                return results[0][0].payload
            return None
        except Exception as e:
            print(f"Error fetching case: {e}")
            return None
    
    def update_case_status(self, case_id: str, status: str, hr_notes: Optional[str] = None, 
                          reviewer_id: Optional[str] = None) -> bool:
        """Update case status"""
        try:
            case = self.get_case_by_id(case_id)
            if not case:
                return False
            
            case["status"] = status
            case["updated_at"] = datetime.now().isoformat()
            if hr_notes:
                case["hr_notes"] = hr_notes
            if reviewer_id:
                case["reviewer_id"] = reviewer_id
            if status in ["approved", "denied", "completed"]:
                case["resolved_at"] = datetime.now().isoformat()
            
            results = self.client.scroll(
                collection_name=self.cases_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
                ),
                limit=1,
                with_vectors=True
            )
            
            if results[0]:
                point_id = results[0][0].id
                vector = results[0][0].vector
                self.client.upsert(
                    collection_name=self.cases_collection,
                    points=[PointStruct(id=point_id, vector=vector, payload=case)]
                )
                return True
            return False
        except Exception as e:
            print(f"Error updating case: {e}")
            return False
    
    def get_cases_by_status(self, status: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get cases by status"""
        try:
            results = self.client.scroll(
                collection_name=self.cases_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value=status))]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [point.payload for point in results[0]]
        except Exception as e:
            print(f"Error fetching cases: {e}")
            return []
    
    def get_employee_cases(self, employee_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all cases for an employee"""
        try:
            results = self.client.scroll(
                collection_name=self.cases_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="employee_id", match=MatchValue(value=employee_id))]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [point.payload for point in results[0]]
        except Exception as e:
            print(f"Error fetching employee cases: {e}")
            return []

# ============================================================================
# RAG SERVICE
# ============================================================================

class RAGService:
    """RAG pipeline for policy documents"""
    
    def __init__(self, qdrant_service):
        self.qdrant_service = qdrant_service
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_policy_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve and format relevant policies"""
        query_embedding = self.embedding_model.encode(query).tolist()
        policies = self.qdrant_service.search_policies(query_embedding, limit=top_k)
        
        if not policies:
            return "No relevant policies found."
        
        context = "RELEVANT COMPANY POLICIES:\n\n"
        for i, section in enumerate(policies, 1):
            context += f"--- Policy Section {i} ---\n"
            context += f"Policy: {section.get('section_title', 'Unknown')}\n"
            context += f"Content: {section.get('section_content', '')}\n"
            context += f"Keywords: {', '.join(section.get('keywords', []))}\n\n"
        
        return context

# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """Business logic for auto-approval"""
    
    def check_leave_eligibility(self, employee: Dict, leave_type: str, 
                                start_date: str, end_date: str, days: int) -> Dict:
        """Check leave eligibility"""
        result = {"eligible": True, "auto_approve": False, "escalate": False, "reason": "", "details": {}}
        
        if employee.get("probation_period"):
            result["escalate"] = True
            result["reason"] = "Employee on probation - requires manager approval"
            return result
        
        balance_field = f"{leave_type}_remaining"
        remaining = employee.get(balance_field, 0)
        
        if remaining < days:
            result["eligible"] = False
            result["escalate"] = True
            result["reason"] = f"Insufficient balance: {remaining} days remaining, {days} requested"
            return result
        
        if days > 10:
            result["escalate"] = True
            result["reason"] = "Request exceeds 10 days - requires director approval"
            return result
        
        if days <= 5:
            result["auto_approve"] = True
            result["reason"] = "Auto-approved - all requirements met"
            result["details"]["balance_before"] = remaining
            result["details"]["balance_after"] = remaining - days
            return result
        
        result["escalate"] = True
        result["reason"] = f"{days} days requires manager approval"
        return result
    
    def check_sick_leave_eligibility(self, employee: Dict, days: int) -> Dict:
        """Check sick leave eligibility"""
        result = {"eligible": True, "auto_approve": False, "escalate": False, "reason": ""}
        
        remaining = employee.get("sick_leave_remaining", 0)
        if remaining < days:
            result["eligible"] = False
            result["escalate"] = True
            result["reason"] = f"Insufficient sick leave: {remaining} days remaining"
            return result
        
        if days <= 2:
            result["auto_approve"] = True
            result["reason"] = "Auto-approved (certificate required for 3+ days)"
            return result
        
        result["escalate"] = True
        result["reason"] = f"{days} days - medical certificate required"
        return result
    
    def classify_complaint_urgency(self, complaint_type: str, description: str) -> Dict:
        """Classify complaint urgency"""
        urgent_keywords = ["harassment", "sexual", "discrimination", "safety", "threat", "violence"]
        is_urgent = any(kw in description.lower() or kw in complaint_type.lower() for kw in urgent_keywords)
        
        return {
            "urgent": is_urgent,
            "priority": "urgent" if is_urgent else "normal",
            "escalate": True,
            "reason": "Protected category - immediate HR attention" if is_urgent else "Requires HR review"
        }
    
    def calculate_duration_days(self, start_date: str, end_date: str) -> int:
        """Calculate duration"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return (end - start).days + 1
        except:
            return 1

# ============================================================================
# AGENT TOOLS
# ============================================================================

class HRAgentTools:
    """LangChain tools"""
    
    def __init__(self, qdrant_service, rag_service, decision_engine):
        self.qdrant_service = qdrant_service
        self.rag_service = rag_service
        self.decision_engine = decision_engine
    
    def get_tools(self):
        return [
            Tool(
                name="get_employee_info",
                func=self._get_employee_info,
                description="Get employee info. Input: employee_id"
            ),
            Tool(
                name="search_policies",
                func=self._search_policies,
                description="Search policies. Input: query string"
            ),
            Tool(
                name="create_leave_request",
                func=self._create_leave_request,
                description="Create leave request. Input: JSON with employee_id, leave_type, start_date, end_date, reason"
            ),
        ]
    
    def _get_employee_info(self, employee_id: str) -> str:
        emp = self.qdrant_service.get_employee_by_id(employee_id.strip())
        if not emp:
            return f"Employee {employee_id} not found"
        
        info = {
            "employee_id": emp.get("employee_id"),
            "name": f"{emp.get('first_name')} {emp.get('last_name')}",
            "department": emp.get("department"),
            "job_level": emp.get("job_level"),
            "annual_leave_remaining": emp.get("annual_leave_remaining"),
            "sick_leave_remaining": emp.get("sick_leave_remaining"),
        }
        return json.dumps(info, indent=2)
    
    def _search_policies(self, query: str) -> str:
        return self.rag_service.get_policy_context(query, top_k=2)
    
    def _create_leave_request(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            employee = self.qdrant_service.get_employee_by_id(data["employee_id"])
            if not employee:
                return json.dumps({"error": "Employee not found"})
            
            days = self.decision_engine.calculate_duration_days(data["start_date"], data["end_date"])
            
            if data["leave_type"] == "sick_leave":
                eligibility = self.decision_engine.check_sick_leave_eligibility(employee, days)
            else:
                eligibility = self.decision_engine.check_leave_eligibility(
                    employee, data["leave_type"], data["start_date"], data["end_date"], days
                )
            
            case_data = {
                "employee_id": data["employee_id"],
                "employee_name": f"{employee.get('first_name')} {employee.get('last_name')}",
                "request_type": data["leave_type"],
                "request_category": "time_off",
                "subject": f"{data['leave_type']} Request",
                "description": data.get("reason", ""),
                "start_date": data["start_date"],
                "end_date": data["end_date"],
                "duration_days": days,
                "status": "approved" if eligibility["auto_approve"] else "pending_review",
                "auto_approved": eligibility["auto_approve"],
                "escalated": eligibility["escalate"],
                "priority": "normal",
                "decision_reason": eligibility["reason"],
                "agent_response": f"✅ Approved! {eligibility['reason']}" if eligibility["auto_approve"] 
                                  else f"⏳ Pending review: {eligibility['reason']}",
            }
            
            case_id = self.qdrant_service.create_case(case_data)
            
            if eligibility["auto_approve"]:
                self.qdrant_service.update_leave_balance(data["employee_id"], data["leave_type"], days)
            
            return json.dumps({
                "case_id": case_id,
                "status": case_data["status"],
                "message": case_data["agent_response"]
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

# ============================================================================
# LANGCHAIN AGENT
# ============================================================================

AGENT_PROMPT = """You are an HR assistant. Help employees with leave requests and questions.

TOOLS:
{tools}

TOOL NAMES: {tool_names}

POLICIES:
{policy_context}

Employee: {input}
{agent_scratchpad}"""

class HRLangChainAgent:
    """LangChain agent"""
    
    def __init__(self, gemini_api_key: str, tools_instance):
        self.tools_instance = tools_instance
        self.tools = tools_instance.get_tools()
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        self.prompt = PromptTemplate(
            template=AGENT_PROMPT,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names", "policy_context"]
        )
        
        self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def process_message(self, employee_id: str, message: str, session_id: str = None) -> Dict:
        try:
            policy_context = self.tools_instance.rag_service.get_policy_context(message, top_k=2)
            enhanced_input = f"Employee ID: {employee_id}\nRequest: {message}"
            
            result = self.agent_executor.invoke({
                "input": enhanced_input,
                "policy_context": policy_context,
                "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
                "tool_names": ", ".join([t.name for t in self.tools])
            })
            
            return {
                "response": result["output"],
                "case_created": "case_id" in result["output"].lower()
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "case_created": False
            }

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="HR Agent API", version="1.0.0")

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
qdrant_service = None
rag_service = None
decision_engine = None
hr_tools = None
hr_agent = None

@app.on_event("startup")
async def startup_event():
    global qdrant_service, rag_service, decision_engine, hr_tools, hr_agent
    
    print("🚀 Initializing HR Agent...")
    
    qdrant_service = QdrantService(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        employee_collection=settings.employee_collection,
        policies_collection=settings.policies_collection,
        cases_collection=settings.cases_collection
    )
    print("✅ Qdrant connected")
    
    rag_service = RAGService(qdrant_service)
    print("✅ RAG service ready")
    
    decision_engine = DecisionEngine()
    print("✅ Decision engine ready")
    
    hr_tools = HRAgentTools(qdrant_service, rag_service, decision_engine)
    print("✅ Tools ready")
    
    hr_agent = HRLangChainAgent(
        gemini_api_key=settings.gemini_api_key,
        tools_instance=hr_tools
    )
    print("✅ Agent ready")
    print("🎉 HR Agent Backend is ready!")

@app.get("/")
async def root():
    return {"status": "healthy", "service": "HR Agent API"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "qdrant": qdrant_service is not None,
            "agent": hr_agent is not None
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = hr_agent.process_message(
            employee_id=request.employee_id,
            message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            response=result["response"],
            case_created=result.get("case_created", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/employee/{employee_id}", response_model=EmployeeInfo)
async def get_employee(employee_id: str):
    try:
        emp = qdrant_service.get_employee_by_id(employee_id)
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return EmployeeInfo(
            employee_id=emp["employee_id"],
            full_name=f"{emp['first_name']} {emp['last_name']}",
            department=emp["department"],
            job_level=emp["job_level"],
            annual_leave_remaining=emp["annual_leave_remaining"],
            sick_leave_remaining=emp["sick_leave_remaining"],
            remote_days_remaining=emp.get("remote_days_per_month", 0),
            probation_period=emp["probation_period"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cases/pending")
async def get_pending_cases():
    try:
        cases = qdrant_service.get_cases_by_status("pending_review", limit=50)
        case_list = [
            CaseDetails(
                case_id=c["case_id"],
                case_number=c["case_number"],
                employee_id=c["employee_id"],
                employee_name=c["employee_name"],
                request_type=c["request_type"],
                status=c["status"],
                created_at=c["created_at"],
                updated_at=c["updated_at"],
                escalated=c.get("escalated", False),
                priority=c.get("priority", "normal"),
                agent_response=c.get("agent_response"),
                hr_notes=c.get("hr_notes")
            ) for c in cases
        ]
        return CaseListResponse(total=len(case_list), cases=case_list, page=1, page_size=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)