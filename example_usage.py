#!/usr/bin/env python3
"""Example usage of the LangGraph agent with feedback loop."""

from langgraph_agent import create_agent_graph, AgentState


def run_example(user_input: str, feedback: str = None, max_iterations: int = 3):
    """Run the agent with a sample input.
    
    Args:
        user_input: The request to process
        feedback: Optional feedback to trigger replanning
        max_iterations: Maximum replanning iterations
    """
    graph = create_agent_graph(max_iterations=max_iterations)
    
    initial_state: AgentState = {
        "input": user_input,
        "plan": [],
        "current_step": 0,
        "research_findings": None,
        "builder_output": None,
        "messages": [],
        "status": "started",
        "next_node": "",
        "feedback": feedback,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    
    print(f"\n{'='*60}")
    print(f"Input: {user_input}")
    if feedback:
        print(f"Feedback: {feedback}")
    print(f"{'='*60}\n")
    
    result = graph.invoke(initial_state)
    
    print("Plan:")
    for i, step in enumerate(result["plan"], 1):
        print(f"  {i}. {step}")
    
    print(f"\nMessages:")
    for msg in result["messages"]:
        print(f"  - {msg}")
    
    print(f"\nStatus: {result['status']}")
    print(f"Iterations: {result.get('iteration', 0)}")
    if result["research_findings"]:
        print(f"\nResearch:\n  {result['research_findings'][:200]}...")
    if result["builder_output"]:
        print(f"\nBuilder Output:\n  {result['builder_output'][:300]}...")
    
    return result


if __name__ == "__main__":
    # Example 1: Research task
    run_example("Research the best Python async frameworks")
    
    # Example 2: Build task
    run_example("Create a REST API with FastAPI")
    
    # Example 3: Build task with feedback (triggers replanning)
    print("\n\n" + "="*60)
    print("EXAMPLE WITH FEEDBACK LOOP")
    print("="*60)
    run_example(
        "Create a simple todo list app",
        feedback="The approach is wrong, please use a database instead of in-memory storage",
        max_iterations=3
    )
