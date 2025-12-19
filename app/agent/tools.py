from langchain.tools import Tool
from typing import Dict, Any
import json

class HRAgentTools:
    """LangChain tools for HR agent"""
    
    def __init__(self, qdrant_service, rag_service, decision_engine):
        self.qdrant_service = qdrant_service
        self.rag_service = rag_service
        self.decision_engine = decision_engine
    
    def get_tools(self):
        """Return list of LangChain tools"""
        return [
            Tool(
                name="get_employee_info",
                func=self._get_employee_info,
                description="Get employee information including leave balances, department, job level. Input: employee_id (e.g., 'EMP_19525')"
            ),
            Tool(
                name="search_policies",
                func=self._search_policies,
                description="Search company policies for relevant information. Input: query string (e.g., 'vacation policy', 'sick leave requirements')"
            ),
            Tool(
                name="check_leave_eligibility",
                func=self._check_leave_eligibility,
                description="Check if employee is eligible for leave request. Input: JSON string with {employee_id, leave_type, start_date, end_date, days}"
            ),
            Tool(
                name="create_leave_request",
                func=self._create_leave_request,
                description="Create a leave request case. Input: JSON string with {employee_id, leave_type, start_date, end_date, reason}"
            ),
            Tool(
                name="create_document_request",
                func=self._create_document_request,
                description="Create document request case. Input: JSON string with {employee_id, document_type, purpose}"
            ),
            Tool(
                name="create_complaint",
                func=self._create_complaint,
                description="File a workplace complaint. Input: JSON string with {employee_id, complaint_type, description}"
            ),
            Tool(
                name="get_employee_cases",
                func=self._get_employee_cases,
                description="Get all cases for an employee. Input: employee_id"
            )
        ]
    
    
    def _get_employee_info(self, employee_id: str) -> str:
        """Get employee information"""
        employee = self.qdrant_service.get_employee_by_id(employee_id.strip())
        
        if not employee:
            return f"Employee {employee_id} not found"
        
        info = {
            "employee_id": employee.get("employee_id"),
            "name": f"{employee.get('first_name')} {employee.get('last_name')}",
            "department": employee.get("department"),
            "job_level": employee.get("job_level"),
            "annual_leave_remaining": employee.get("annual_leave_remaining"),
            "sick_leave_remaining": employee.get("sick_leave_remaining"),
            "emergency_leave_remaining": employee.get("emergency_leave_remaining"),
            "remote_work_eligible": employee.get("remote_work_eligible"),
            "remote_days_remaining": employee.get("remote_days_per_month", 0) - employee.get("remote_days_used_this_month", 0),
            "probation_period": employee.get("probation_period"),
            "manager": employee.get("manager_name")
        }
        
        return json.dumps(info, indent=2)
    
    def _search_policies(self, query: str) -> str:
        """Search company policies"""
        policy_context = self.rag_service.get_policy_context(query, top_k=3)
        return policy_context
    
    def _check_leave_eligibility(self, input_json: str) -> str:
        """Check leave eligibility"""
        try:
            data = json.loads(input_json)
            employee_id = data["employee_id"]
            leave_type = data["leave_type"]
            start_date = data["start_date"]
            end_date = data["end_date"]
            days = data.get("days", 1)
            
            employee = self.qdrant_service.get_employee_by_id(employee_id)
            if not employee:
                return json.dumps({"error": "Employee not found"})
            
            if leave_type == "sick_leave":
                result = self.decision_engine.check_sick_leave_eligibility(employee, days)
            else:
                result = self.decision_engine.check_leave_eligibility(
                    employee, leave_type, start_date, end_date, days
                )
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _create_leave_request(self, input_json: str) -> str:
        """Create leave request case"""
        try:
            data = json.loads(input_json)
            employee_id = data["employee_id"]
            leave_type = data["leave_type"]
            start_date = data["start_date"]
            end_date = data["end_date"]
            reason = data.get("reason", "")
            
            employee = self.qdrant_service.get_employee_by_id(employee_id)
            if not employee:
                return json.dumps({"error": "Employee not found"})
            
            days = self.decision_engine.calculate_duration_days(start_date, end_date)
            
            if leave_type == "sick_leave":
                eligibility = self.decision_engine.check_sick_leave_eligibility(employee, days)
            else:
                eligibility = self.decision_engine.check_leave_eligibility(
                    employee, leave_type, start_date, end_date, days
                )
            
            case_data = {
                "employee_id": employee_id,
                "employee_name": f"{employee.get('first_name')} {employee.get('last_name')}",
                "request_type": leave_type,
                "request_category": "time_off",
                "subject": f"{leave_type.replace('_', ' ').title()} Request",
                "description": reason,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": days,
                "status": "approved" if eligibility.get("auto_approve") else "pending_review",
                "auto_approved": eligibility.get("auto_approve", False),
                "escalated": eligibility.get("escalate", False),
                "priority": "normal",
                "decision_made_by": "agent" if eligibility.get("auto_approve") else "pending",
                "approval_decision": "approved" if eligibility.get("auto_approve") else None,
                "decision_reason": eligibility.get("reason", ""),
                "agent_confidence_score": 0.95 if eligibility.get("auto_approve") else 0.6,
                "policy_references": ["POL_001", "POL_002"],
                "eligibility_check_passed": eligibility.get("eligible", False),
                "balance_before": employee.get(f"{leave_type}_remaining", 0),
                "balance_after": employee.get(f"{leave_type}_remaining", 0) - days if eligibility.get("auto_approve") else employee.get(f"{leave_type}_remaining", 0),
                "agent_response": self._generate_response_message(eligibility, leave_type, days),
                "employee_notes": reason,
                "hr_notes": None
            }
            
            case_id = self.qdrant_service.create_case(case_data)
            
            if eligibility.get("auto_approve"):
                self.qdrant_service.update_leave_balance(employee_id, leave_type, days)
            
            return json.dumps({
                "case_id": case_id,
                "status": case_data["status"],
                "auto_approved": case_data["auto_approved"],
                "escalated": case_data["escalated"],
                "message": case_data["agent_response"]
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _create_document_request(self, input_json: str) -> str:
        """Create document request case"""
        try:
            data = json.loads(input_json)
            employee_id = data["employee_id"]
            document_type = data["document_type"]
            purpose = data.get("purpose", "")
            
            employee = self.qdrant_service.get_employee_by_id(employee_id)
            if not employee:
                return json.dumps({"error": "Employee not found"})
            
            case_data = {
                "employee_id": employee_id,
                "employee_name": f"{employee.get('first_name')} {employee.get('last_name')}",
                "request_type": document_type,
                "request_category": "document_request",
                "subject": f"{document_type.replace('_', ' ').title()} Request",
                "description": f"Document needed for: {purpose}",
                "status": "approved",
                "auto_approved": True,
                "escalated": False,
                "priority": "normal",
                "decision_made_by": "agent",
                "approval_decision": "approved",
                "decision_reason": "Standard document request auto-approved per policy",
                "agent_confidence_score": 0.99,
                "policy_references": ["POL_005"],
                "agent_response": f"Your {document_type.replace('_', ' ')} has been approved and will be ready within 1-2 business days.",
                "employee_notes": purpose,
                "document_generated": False,
                "document_type": document_type
            }
            
            case_id = self.qdrant_service.create_case(case_data)
            
            return json.dumps({
                "case_id": case_id,
                "status": "approved",
                "message": case_data["agent_response"]
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _create_complaint(self, input_json: str) -> str:
        """Create complaint case"""
        try:
            data = json.loads(input_json)
            employee_id = data["employee_id"]
            complaint_type = data["complaint_type"]
            description = data["description"]
            
            employee = self.qdrant_service.get_employee_by_id(employee_id)
            if not employee:
                return json.dumps({"error": "Employee not found"})
            
            urgency = self.decision_engine.classify_complaint_urgency(complaint_type, description)
            
            case_data = {
                "employee_id": employee_id,
                "employee_name": f"{employee.get('first_name')} {employee.get('last_name')}",
                "request_type": "complaint",
                "request_category": "workplace_conduct",
                "subject": f"Complaint: {complaint_type}",
                "description": description,
                "status": "under_investigation" if urgency["urgent"] else "pending_review",
                "auto_approved": False,
                "escalated": True,
                "escalation_reason": urgency["reason"],
                "priority": urgency["priority"],
                "decision_made_by": "human_reviewer",
                "policy_references": ["POL_006"],
                "agent_response": "Your complaint has been received and escalated to HR. You are protected from retaliation. An HR representative will contact you within 24 hours." if urgency["urgent"] else "Your complaint has been received and will be reviewed by HR.",
                "employee_notes": description,
                "hr_notes": "Pending investigation"
            }
            
            case_id = self.qdrant_service.create_case(case_data)
            
            return json.dumps({
                "case_id": case_id,
                "status": case_data["status"],
                "urgent": urgency["urgent"],
                "message": case_data["agent_response"]
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _get_employee_cases(self, employee_id: str) -> str:
        """Get employee case history"""
        cases = self.qdrant_service.get_employee_cases(employee_id.strip())
        
        if not cases:
            return json.dumps({"message": "No cases found for this employee"})
        
        case_list = []
        for case in cases[:5]:  
            case_list.append({
                "case_number": case.get("case_number"),
                "request_type": case.get("request_type"),
                "status": case.get("status"),
                "created_at": case.get("created_at"),
                "escalated": case.get("escalated", False)
            })
        
        return json.dumps(case_list, indent=2)
    
    
    def _generate_response_message(self, eligibility: Dict, leave_type: str, days: int) -> str:
        """Generate user-friendly response message"""
        if eligibility.get("auto_approve"):
            return f"Your {leave_type.replace('_', ' ')} request for {days} days has been approved! {eligibility.get('reason', '')}"
        elif eligibility.get("escalate"):
            return f"Your request requires additional review. Reason: {eligibility.get('reason', '')}. You'll be notified once a decision is made."
        else:
            return f"Your request cannot be approved. Reason: {eligibility.get('reason', '')}"