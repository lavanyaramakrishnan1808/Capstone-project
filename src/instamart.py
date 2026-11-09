import os
import re
import json
import time
import uuid
import sqlite3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm
# TASK 1: CHUNK POLICY DOCUMENTS
def chunk_documents(chunk_size=40, overlap=10):
    policy_chunks = []
    policy_folder = "data/policies"
    for filename in sorted(os.listdir(policy_folder)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(policy_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            policy_chunks.append({
                "source": filename,
                "text": chunk_text
            })
            if end >= len(words):
                break
            start += chunk_size - overlap
    return policy_chunks
# Execute chunking
policy_chunks = chunk_documents()
print("\n========== POLICY CHUNKING ==========")
print("Total policy_chunks:", len(policy_chunks))
print(pd.Series([c["source"] for c in policy_chunks]).value_counts())
# TASK 2: TF-IDF VECTORISATION AND RETRIEVAL
# Extract text from all policy_chunks
policy_chunk_texts = [chunk["text"] for chunk in policy_chunks]
# Fit TF-IDF on policy policy_chunks
tfidf_vectorizer = TfidfVectorizer()
policy_chunk_vectors = tfidf_vectorizer.fit_transform(policy_chunk_texts)
def retrieve_policy_chunks(sql_query, k=3):
    # Transform sql_query using the same fitted tfidf_vectorizer
    search_query_vector = tfidf_vectorizer.transform([sql_query])
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(search_query_vector, policy_chunk_vectors)[0]
    # Get top k chunk indexes
    top_chunk_indices = np.argsort(similarity_scores)[::-1][:k]
    retrieval_results = []
    for index in top_chunk_indices:
        retrieval_results.append({
            "source": policy_chunks[index]["source"],
            "score": float(similarity_scores[index]),
            "text": policy_chunks[index]["text"]
        })
    return retrieval_results
# Test retrieval
test_results = retrieve_policy_chunks("refund policy late delivery")
print("\n========== RETRIEVAL TEST ==========")
for policy_result in test_results:
    print("Source:", policy_result["source"])
    print("Score:", round(policy_result["score"], 4))
    print("Text:", policy_result["text"][:100], "...")
    print()
# TASK 3: POLICY ASSISTANT
POLICY_SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.
Answer questions ONLY using the supplied policy policy_context.
Do not use outside knowledge or invent information.
If the answer is not available in the policy_context,
respond exactly:
I could not find that information in the policy documents.
"""
def answer(customer_question):
    # Retrieve top 3 relevant policy policy_chunks
    retrieved_policy_chunks = retrieve_policy_chunks(customer_question, k=3)
    # Build policy_context
    policy_context_parts = []
    for item in retrieved_policy_chunks:
        policy_context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )
    policy_context = "\n".join(policy_context_parts)
    # User message format
    agent_user_message = (
        f"CONTEXT:\n{policy_context}\n"
        f"QUESTION:\n{customer_question}"
    )
    conversation_messages = [
        {
            "role": "system",
            "content": POLICY_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": agent_user_message
        }
    ]
    # Call mock LLM
    response = mock_llm.chat(conversation_messages)
    final_answer = response.choices[0].message.content
    # Return answer and sources
    sources = [item["source"] for item in retrieved_policy_chunks]
    return {
        "answer": final_answer,
        "sources": sources
    }
# Test policy assistant
print("\n========== POLICY ASSISTANT TEST ==========")
policy_result = answer("What is the refund policy for late delivery?")
print("Answer:", policy_result["answer"])
print("Sources:", policy_result["sources"])
# TASK 4: RETRIEVAL AND GROUNDING EVALUATION
def evaluate_policy():
    policy_evaluation_data = pd.read_csv(
        "data/raw/policy_eval.csv",
        keep_default_na=False
    )
    retrieval_results = []
    for _, row in policy_evaluation_data.iterrows():
        customer_question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]
        retrieved_policy_chunks = retrieve_policy_chunks(customer_question, k=3)
        policy_result = answer(customer_question)
        retrieved_docs = [
            item["source"] for item in retrieved_policy_chunks
        ]
        # Empty expected_doc means unanswerable
        answerable = expected_doc.strip().upper() != "NONE"
        doc_found = (
            expected_doc in retrieved_docs
            if answerable else False
        )
        fact_found = (
            expected_fact in policy_result["answer"]
            if answerable else False
        )
        fallback_found = policy_result["answer"].startswith(
            "I could not find"
        )
        retrieval_results.append({
            "customer_question": customer_question,
            "expected_doc": expected_doc,
            "answerable": answerable,
            "doc_found": doc_found,
            "fact_found": fact_found,
            "fallback_found": fallback_found
        })
    evaluation_results_df = pd.DataFrame(retrieval_results)
    answerable_evaluation = evaluation_results_df[
        evaluation_results_df["answerable"]
    ]
    unanswerable_evaluation = evaluation_results_df[
        ~evaluation_results_df["answerable"]
    ]
    print("\n========== POLICY EVALUATION ==========")
    print(
        "Expected document retrieved_policy_chunks:",
        answerable_evaluation["doc_found"].sum(),
        "/",
        len(answerable_evaluation)
    )
    print(
        "Expected fact found:",
        answerable_evaluation["fact_found"].sum(),
        "/",
        len(answerable_evaluation)
    )
    print(
        "Fallback answers:",
        unanswerable_evaluation["fallback_found"].sum(),
        "/",
        len(unanswerable_evaluation)
    )
    return evaluation_results_df
policy_evaluation_results = evaluate_policy()
# TASK 4B: ZERO-OVERLAP EVALUATION
# Create policy_chunks without overlap
zero_overlap_chunks = chunk_documents(
    chunk_size=40,
    overlap=0
)
# Temporarily replace original policy_chunks
original_policy_chunks = policy_chunks
policy_chunks = zero_overlap_chunks
# Fit a new TF-IDF tfidf_vectorizer
policy_chunk_texts = [chunk["text"] for chunk in policy_chunks]
tfidf_vectorizer = TfidfVectorizer()
policy_chunk_vectors = tfidf_vectorizer.fit_transform(policy_chunk_texts)
# Evaluate again
zero_overlap_results = evaluate_policy()
failed_zero_overlap = zero_overlap_results[
    zero_overlap_results["answerable"] & ~zero_overlap_results["fact_found"]
]
print("\nAnswerable questions that failed with zero overlap:")
for customer_question in failed_zero_overlap["customer_question"]:
    print("-", customer_question)
print("\n========== ZERO OVERLAP CHUNKING ==========")
print("Total policy_chunks:", len(zero_overlap_chunks))
# Restore original policy_chunks and tfidf_vectorizer
policy_chunks = original_policy_chunks
policy_chunk_texts = [chunk["text"] for chunk in policy_chunks]
tfidf_vectorizer = TfidfVectorizer()
policy_chunk_vectors = tfidf_vectorizer.fit_transform(policy_chunk_texts)
print("Original overlapping policy_chunks restored.")
# TASK 5: REFUND TOOLS
from datetime import datetime
# Tool 1: Get order
def get_order(order_id):
    database_connection = sqlite3.connect("data/instamart.db")
    sql_query = """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
    """
    policy_result = pd.read_sql_query(
        sql_query,
        database_connection,
        params=(order_id,)
    )
    database_connection.close()
    if policy_result.empty:
        return {"error": "order not found"}
    return policy_result.iloc[0].to_dict()
# Tool 2: Search policy
def search_policy(query):
    retrieval_results = retrieve_policy_chunks(query, k=3)
    return {"retrieval_results": retrieval_results}
# Tool 3: Calculate refund
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0
    return {"refund_inr": refund}
# Helper: Append ledger ledger_entry
def write_ledger(tool_name, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }
    with open(
        "outputs/refund_ledger.jsonl",
        "a",
        encoding="utf-8"
    ) as file:
        file.write(json.dumps(ledger_entry) + "\n")
# Tool 4: Issue refund
def issue_refund(order_id, refund_inr):
    write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )
    return {"refund_inr": refund_inr}
# Tool 5: Escalate to human
def escalate_to_human(order_id, refund_inr):
    write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )
    return {"refund_inr": refund_inr}
# Tool specifications in JSON-Schema style
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant refund policy documents.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount using delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically issued refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]
# Tool map
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
print("\n========== REFUND TOOL VERIFICATION ==========")
print("Tools:", list(TOOL_MAP.keys()))
print("Refund for 60 minutes:",
      calculate_refund(128.25, 60))
print("Refund for 30 minutes:",
      calculate_refund(128.25, 30))
# TASK 8: RESUMABLE AGENT
def save_checkpoint(run_id, scenario_id, conversation_messages,
                   tokens_used, agent_steps, executed_tools,
                   step_number):
    os.makedirs("state", exist_ok=True)
    agent_checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "conversation_messages": conversation_messages,
        "tokens_used": tokens_used,
        "agent_steps": agent_steps,
        "executed_tools": executed_tools,
        "step_number": step_number
    }
    agent_checkpoint_path = f"state/{run_id}.json"
    with open(agent_checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(agent_checkpoint, file, indent=4)
def run_agent(
    agent_user_message,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    scenario_id=None,
    run_id=None,
    crash_after_tool=None
):
    if run_id is None:
        run_id = str(uuid.uuid4())
    conversation_messages = [
        {
            "role": "system",
            "content": "AGENT: You are an Instamart refund agent. Use tools to verify orders, check policies, calculate refunds, and process or escalate refunds."
        },
        {
            "role": "user",
            "content": agent_user_message
        }
    ]
    agent_steps = []
    executed_tools = []
    llm_calls = 0
    tokens_used = 0
    start_time = time.monotonic()
    status = None
    final_answer = None
    for iteration in range(max_iterations):
        # Timeout check
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = (
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Please review manually."
            )
            break
        response = mock_llm.chat(conversation_messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        assistant_message = response.choices[0].message
        # Token limit check
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = (
                "FALLBACK_TOKEN_LIMIT: Token limit exceeded. "
                "Please review manually."
            )
            break
        tool_calls = assistant_message.tool_calls
        # Final answer
        if not tool_calls:
            final_answer = assistant_message.content
            status = final_answer.split(":")[0].strip()
            break
        conversation_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })
        for call in tool_calls:
            tool_arguments = call.arguments
            if isinstance(tool_arguments, str):
                tool_arguments = json.loads(tool_arguments)
            tool_name = call.name
            tool_result = TOOL_MAP[tool_name](**tool_arguments)
            agent_steps.append({
                "tool": tool_name,
                "arguments": tool_arguments,
                "tool_result": tool_result
            })
            executed_tools.append(tool_name)
            conversation_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
            # Save agent_checkpoint AFTER tool tool_result
            save_checkpoint(
                run_id,
                scenario_id,
                conversation_messages,
                tokens_used,
                agent_steps,
                executed_tools,
                len(agent_steps)
            )
            # Simulate crash after requested tool execution
            if crash_after_tool is not None:
                if len(executed_tools) == crash_after_tool:
                    raise RuntimeError("SIMULATED CRASH")
    if final_answer is None and status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = (
            "FALLBACK_MAX_ITERATIONS: Agent reached the maximum "
            "number of iterations. Please review manually."
        )
    return {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "agent_steps": agent_steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "executed_tools": executed_tools
    }
# RESUME AGENT
def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    agent_checkpoint_path = f"state/{run_id}.json"
    with open(agent_checkpoint_path, "r", encoding="utf-8") as file:
        agent_checkpoint = json.load(file)
    conversation_messages = agent_checkpoint["conversation_messages"]
    scenario_id = agent_checkpoint["scenario_id"]
    tokens_used = agent_checkpoint["tokens_used"]
    agent_steps = agent_checkpoint["agent_steps"]
    executed_tools = agent_checkpoint["executed_tools"]
    start_time = time.monotonic()
    status = None
    final_answer = None
    llm_calls = 0
    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = (
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Please review manually."
            )
            break
        response = mock_llm.chat(conversation_messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        assistant_message = response.choices[0].message
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = (
                "FALLBACK_TOKEN_LIMIT: Token limit exceeded. "
                "Please review manually."
            )
            break
        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            final_answer = assistant_message.content
            status = final_answer.split(":")[0].strip()
            break
        conversation_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })
        for call in tool_calls:
            tool_arguments = call.arguments
            if isinstance(tool_arguments, str):
                tool_arguments = json.loads(tool_arguments)
            tool_name = call.name
            tool_result = TOOL_MAP[tool_name](**tool_arguments)
            agent_steps.append({
                "tool": tool_name,
                "arguments": tool_arguments,
                "tool_result": tool_result
            })
            executed_tools.append(tool_name)
            conversation_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
            save_checkpoint(
                run_id,
                scenario_id,
                conversation_messages,
                tokens_used,
                agent_steps,
                executed_tools,
                len(agent_steps)
            )
    if final_answer is None and status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = (
            "FALLBACK_MAX_ITERATIONS: Agent reached the maximum "
            "number of iterations. Please review manually."
        )
    return {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "agent_steps": agent_steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "executed_tools": executed_tools
    }
# RUN ALL 7 AGENT SCENARIOS
agent_scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
os.makedirs("outputs", exist_ok=True)
open("outputs/refund_ledger.jsonl", "w", encoding="utf-8").close()
os.makedirs("state", exist_ok=True)
for old_checkpoint in os.listdir("state"):
    if old_checkpoint.endswith(".json"):
        os.remove(os.path.join("state", old_checkpoint))
scenario_results = []
# Reset mock before the independent agent_scenario batch
mock_llm.reset()
with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as file:
    for _, agent_scenario in agent_scenarios.iterrows():
        scenario_id = agent_scenario["scenario_id"]
        # Identify the user-message column
        agent_user_message = agent_scenario.get(
            "agent_user_message",
            agent_scenario.get("request", agent_scenario.get("message", None))
        )
        if pd.isna(agent_user_message):
            raise ValueError(
                f"User message missing for agent_scenario {scenario_id}"
            )
        policy_result = run_agent(
            agent_user_message=str(agent_user_message),
            scenario_id=scenario_id
        )
        # Save agent_scenario policy_result
        scenario_record = {
            "scenario_id": scenario_id,
            "status": policy_result["status"],
            "final_answer": policy_result["final_answer"],
            "llm_calls": policy_result["llm_calls"],
            "tokens_used": policy_result["tokens_used"],
            "agent_steps": policy_result["agent_steps"]
        }
        # DOCX Task 6 verification: expected status and refund amount
        expected_status = str(agent_scenario["expected_outcome"])
        if scenario_id == "S7":
            assert str(policy_result["status"]).startswith("FALLBACK"), (
                f"S7 must fall back, got {policy_result['status']}"
            )
        else:
            assert policy_result["status"] == expected_status, (
                f"{scenario_id}: expected status {expected_status}, "
                f"got {policy_result['status']}"
            )
        if scenario_id in {"S1", "S2", "S3", "S4", "S5", "S6"}:
            refund_observations = [
                step["tool_result"]["refund_inr"]
                for step in policy_result["agent_steps"]
                if step["tool"] == "calculate_refund"
                and "refund_inr" in step.get("tool_result", {})
            ]
            expected_refund = float(agent_scenario["expected_refund_inr"])
            if refund_observations:
                actual_refund = float(refund_observations[-1])
            else:
                # NOT_FOUND agent_scenarios never reach calculate_refund; no refund is issued.
                actual_refund = 0.0
            assert abs(actual_refund - expected_refund) <= 0.01, (
                f"{scenario_id}: expected refund INR {expected_refund:.2f}, "
                f"got INR {actual_refund:.2f}"
            )
        file.write(json.dumps(scenario_record) + "\n")
        scenario_results.append(scenario_record)
        print(f"\nScenario: {scenario_id}")
        print("Status:", policy_result["status"])
        print("Final answer:", policy_result["final_answer"])
        print("LLM calls:", policy_result["llm_calls"])
        print("Tokens:", policy_result["tokens_used"])
        print("Tools:", [step["tool"] for step in policy_result["agent_steps"]])
# AGENT SUMMARY
scenario_results_df = pd.DataFrame(scenario_results)
print("\n========== AGENT SCENARIO RESULTS ==========")
print(
    scenario_results_df[
        ["scenario_id", "status", "llm_calls", "tokens_used"]
    ].to_string(index=False)
)
print("\nSaved: outputs/agent_runs.jsonl")
# TASK 7: SAFETY LIMIT DEMONSTRATIONS
mock_llm.reset()
print("\n========== TOKEN LIMIT DEMO ==========")
s1_row = agent_scenarios.loc[
    agent_scenarios["scenario_id"] == "S1"
].iloc[0]
# Automatically identify the user-message column
message_column = next(
    col for col in ["agent_user_message", "request", "message"]
    if col in agent_scenarios.columns
)
s1_message = str(s1_row[message_column])
mock_llm.reset()
token_demo = run_agent(
    agent_user_message=s1_message,
    scenario_id="S1_TOKEN_DEMO",
    token_limit=200
)
print("Status:", token_demo["status"])
print("Final answer:", token_demo["final_answer"])
print("LLM calls:", token_demo["llm_calls"])
print("Tokens:", token_demo["tokens_used"])
# TASK 7: TIMEOUT DEMO
print("\n========== TIMEOUT DEMO ==========")
mock_llm.reset()
timeout_demo = run_agent(
    agent_user_message=s1_message,
    scenario_id="S1_TIMEOUT_DEMO",
    time_limit_seconds=0
)
print("Status:", timeout_demo["status"])
print("Final answer:", timeout_demo["final_answer"])
print("LLM calls:", timeout_demo["llm_calls"])
print("Tokens:", timeout_demo["tokens_used"])
# TASK 8: CRASH AND RESUME DEMONSTRATION
print("\n========== CRASH AND RESUME DEMO ==========")
# Reuse s1_message identified in the token-limit demo
# Create a unique run ID
resume_run_id = "S1_RESUME_DEMO"
# Reset mock before starting this independent run
mock_llm.reset()
# First execution: simulate crash after second tool
try:
    first_run = run_agent(
        agent_user_message=s1_message,
        scenario_id="S1",
        run_id=resume_run_id,
        crash_after_tool=2
    )
except RuntimeError as error:
    print("Simulated crash:", error)
# Resume from saved agent_checkpoint
resumed_result = resume_agent(resume_run_id)
print("\n========== RESUMED RESULT ==========")
print("Status:", resumed_result["status"])
print("Final answer:", resumed_result["final_answer"])
print("Tools executed:", resumed_result["executed_tools"])
print("LLM calls after resume:", resumed_result["llm_calls"])
print("Tokens after resume:", resumed_result["tokens_used"])
print("\nTotal tools across crash and resume:")
print(resumed_result["executed_tools"])

