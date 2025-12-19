from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

AGENT_PROMPT = """You are an intelligent HR assistant helping employees with their workplace requests and questions. You have access to company policies, employee data, and can process various HR requests.

IMPORTANT GUIDELINES:
1. Always be professional, friendly, and helpful
2. Use the tools provided to access employee data and company policies
3. For leave requests, check eligibility before creating the request
4. Always explain decisions clearly, referencing specific policies when relevant
5. If you need to escalate to HR, explain why clearly
6. Keep responses concise but complete

AVAILABLE TOOLS:
{tools}

TOOL NAMES:
{tool_names}

CONTEXT FROM COMPANY POLICIES:
{policy_context}

When processing requests, follow this workflow:
1. Identify what the employee needs (leave, document, complaint, information)
2. Use get_employee_info to fetch their details
3. Use search_policies to find relevant company policies
4. For leave requests: use check_leave_eligibility first, then create_leave_request
5. Provide clear, actionable responses

Current conversation:
{chat_history}

Employee: {input}

Thought: {agent_scratchpad}"""

class HRLangChainAgent:
    """LangChain agent for HR operations"""
    
    def __init__(self, gemini_api_key: str, tools_instance):
        self.tools_instance = tools_instance
        self.tools = tools_instance.get_tools()
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3,  
            convert_system_message_to_human=True
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=AGENT_PROMPT,
            input_variables=["input", "chat_history", "agent_scratchpad", "policy_context"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=60
        )
        
        self.conversations = {}
    
    def process_message(
        self,
        employee_id: str,
        message: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process employee message and return response
        
        Args:
            employee_id: Employee ID
            message: User's message
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            if session_id and session_id in self.conversations:
                chat_history = self.conversations[session_id]
            else:
                chat_history = ""
                if session_id:
                    self.conversations[session_id] = ""
            
            policy_context = self.tools_instance.rag_service.get_policy_context(message, top_k=2)
            
            enhanced_input = f"Employee ID: {employee_id}\nQuestion/Request: {message}"
            
            result = self.agent_executor.invoke({
                "input": enhanced_input,
                "chat_history": chat_history,
                "policy_context": policy_context
            })
            
            if session_id:
                self.conversations[session_id] += f"\nHuman: {message}\nAssistant: {result['output']}"
            
            output = result["output"]
            case_created = "case_id" in output.lower() or "case created" in output.lower()
            
            return {
                "response": result["output"],
                "case_created": case_created,
                "session_id": session_id,
                "intermediate_steps": len(result.get("intermediate_steps", []))
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error processing your request: {str(e)}. Please try rephrasing your question or contact HR directly.",
                "error": str(e),
                "case_created": False
            }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]