from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid

from app.config import get_settings
from app.models.schemas import (
    ChatRequest, ChatResponse, LeaveRequest, DocumentRequest,
    ComplaintRequest, CaseUpdateRequest, EmployeeInfo, CaseDetails,
    CaseListResponse
)
from app.services.qdrant_service import QdrantService
from app.services.rag_service import RAGService
from app.services.decision_engine import DecisionEngine
from app.agent.tools import HRAgentTools
from app.agent.langchain_agent import HRLangChainAgent


qdrant_service = None
rag_service = None
decision_engine = None
hr_tools = None
hr_agent = None

# ============================================================================

app = FastAPI(
    title="HR Agent API",
    description="AI-powered HR assistant backend",
    version="1.0.0"
)

# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global qdrant_service, rag_service, decision_engine, hr_tools, hr_agent
    
    settings = get_settings()
    
    print("🚀 Initializing HR Agent Backend...")
    
    # Initialize Qdrant service
    qdrant_service = QdrantService(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        employee_collection=settings.employee_collection,
        policies_collection=settings.policies_collection,
        cases_collection=settings.cases_collection
    )
    print("✅ Qdrant service initialized")
    
    # Initialize RAG service
    rag_service = RAGService(qdrant_service)
    print("✅ RAG service initialized")
    
    # Initialize decision engine
    decision_engine = DecisionEngine()
    print("✅ Decision engine initialized")
    
    # Initialize tools
    hr_tools = HRAgentTools(qdrant_service, rag_service, decision_engine)
    print("✅ HR tools initialized")
    
    # Initialize LangChain agent
    hr_agent = HRLangChainAgent(
        gemini_api_key=settings.gemini_api_key,
        tools_instance=hr_tools
    )
    print("✅ LangChain agent initialized")
    
    print("🎉 HR Agent Backend ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down HR Agent Backend...")

# ============================================================================

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "HR Agent API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "qdrant": qdrant_service is not None,
            "rag": rag_service is not None,
            "agent": hr_agent is not None
        }
    }

# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        result = hr_agent.process_message(
            employee_id=request.employee_id,
            message=request.message,
            session_id=session_id
        )
        return ChatResponse(
            response=result["response"],
            case_created=result.get("case_created", False),
            case_id=None,
            status=None,
            escalated=False,
            confidence_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================

@app.get("/employee/{employee_id}", response_model=EmployeeInfo)
async def get_employee(employee_id: str):
    try:
        employee = qdrant_service.get_employee_by_id(employee_id)
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        return EmployeeInfo(
            employee_id=employee["employee_id"],
            full_name=f"{employee['first_name']} {employee['last_name']}",
            department=employee["department"],
            job_level=employee["job_level"],
            annual_leave_remaining=employee["annual_leave_remaining"],
            sick_leave_remaining=employee["sick_leave_remaining"],
            remote_days_remaining=employee["remote_days_per_month"] - employee["remote_days_used_this_month"],
            probation_period=employee["probation_period"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================

@app.get("/employee/{employee_id}/cases")
async def get_employee_cases(employee_id: str, limit: int = 20):
    try:
        cases = qdrant_service.get_employee_cases(employee_id, limit=limit)
        case_list = [
            CaseDetails(
                case_id=case["case_id"],
                case_number=case["case_number"],
                employee_id=case["employee_id"],
                employee_name=case["employee_name"],
                request_type=case["request_type"],
                status=case["status"],
                created_at=case["created_at"],
                updated_at=case["updated_at"],
                escalated=case.get("escalated", False),
                priority=case.get("priority", "normal"),
                agent_response=case.get("agent_response"),
                hr_notes=case.get("hr_notes")
            )
            for case in cases
        ]
        return CaseListResponse(
            total=len(case_list),
            cases=case_list,
            page=1,
            page_size=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode
    )
