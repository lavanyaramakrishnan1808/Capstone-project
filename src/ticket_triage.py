# PART C: SUPPORT TICKET TRIAGE
import pandas as pd
import numpy as np
import re
import json
import random
import time
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import mock_llm
# Configuration
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 100000
# Track retry failures separately for each pipeline run
RETRY_FAILURE_COUNT = 0
PERMANENT_FAILURE_COUNT = 0
ALLOWED_CATEGORIES = {

    "billing",

    "delivery",

    "technical",

    "fraud"

}
# Load support_tickets
support_tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)
print("\n========== TICKET DATA ==========")
print("Shape:", support_tickets.shape)
print("Columns:", support_tickets.columns.tolist())
print(support_tickets.head())
# TASK 1: INPUT GUARDRAILS
# Mark tickets that contain no text
support_tickets["text"] = support_tickets["text"].astype(str)
support_tickets["is_empty"] = support_tickets["text"].str.strip().eq("")
# Estimate token count with the mock LLM
support_tickets["estimated_tokens"] = support_tickets["text"].apply(

    mock_llm.est_tokens
)
# Identify long support_tickets
support_tickets["is_long"] = (

    support_tickets["estimated_tokens"] > MAX_INPUT_TOKENS
)
print("\n========== INPUT GUARDRAILS ==========")
print("Empty support_tickets:", support_tickets["is_empty"].sum())
print("Long support_tickets:", support_tickets["is_long"].sum())
print("\nLong ticket token estimates:")
print(

    support_tickets.loc[

        support_tickets["is_long"],

        ["ticket_id", "estimated_tokens"]

    ]
)
# TASK 2: PII REDACTION
# Patterns used to find PII and replacement labels
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"
def redact_pii(text):

    email_count = len(re.findall(EMAIL_PATTERN, text))

    phone_count = len(re.findall(PHONE_PATTERN, text))

    order_count = len(re.findall(ORDER_PATTERN, text))



    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)

    text = re.sub(PHONE_PATTERN, "[PHONE]", text)

    text = re.sub(ORDER_PATTERN, "[ORDER_ID]", text)



    return text, email_count, phone_count, order_count
# Apply redaction to all support_tickets
pii_redaction_results = support_tickets["text"].apply(redact_pii)
support_tickets["redacted_text"] = pii_redaction_results.apply(lambda x: x[0])
support_tickets["email_count"] = pii_redaction_results.apply(lambda x: x[1])
support_tickets["phone_count"] = pii_redaction_results.apply(lambda x: x[2])
support_tickets["order_count"] = pii_redaction_results.apply(lambda x: x[3])
print("\n========== REDACTION RESULTS ==========")
print("Emails:", support_tickets["email_count"].sum())
print("Phones:", support_tickets["phone_count"].sum())
print("Order IDs:", support_tickets["order_count"].sum())
print(

    "Total substitutions:",

    support_tickets[["email_count", "phone_count", "order_count"]].sum().sum()

)
# Check that redaction removed every matched PII value
unredacted_email_count = support_tickets["redacted_text"].apply(

    lambda x: len(re.findall(EMAIL_PATTERN, x))

).sum()
unredacted_phone_count = support_tickets["redacted_text"].apply(

    lambda x: len(re.findall(PHONE_PATTERN, x))

).sum()
unredacted_order_id_count = support_tickets["redacted_text"].apply(

    lambda x: len(re.findall(ORDER_PATTERN, x))

).sum()
print("\nRemaining emails:", unredacted_email_count)
print("Remaining phones:", unredacted_phone_count)
print("Remaining order IDs:", unredacted_order_id_count)
assert unredacted_email_count == 0
assert unredacted_phone_count == 0
assert unredacted_order_id_count == 0
# TASK 3: SYSTEM PROMPTS
ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.
Allowed categories:

- billing

- delivery

- technical

- fraud
Return JSON only in this format:

{"category": "...", "reason": "..."}

"""
FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.
Allowed categories:

- billing

- delivery

- technical

- fraud
EXAMPLES:
Ticket: I was charged twice for my order.
{"category": "billing", "reason": "Duplicate charge"}
Ticket: My package has not arrived.
{"category": "delivery", "reason": "Package not received"}
Ticket: Someone used my card without permission.
{"category": "fraud", "reason": "Unauthorized transaction"}
Return JSON only in this format:
{"category": "...", "reason": "..."}
"""
print("\n========== PROMPT VERIFICATION ==========")
print("Zero-shot contains CLASSIFY:", "CLASSIFY" in ZERO_SHOT_PROMPT)
print("Zero-shot contains EXAMPLES:", "EXAMPLES:" in ZERO_SHOT_PROMPT)
print("Few-shot contains CLASSIFY:", "CLASSIFY" in FEW_SHOT_PROMPT)
print("Few-shot contains EXAMPLES:", "EXAMPLES:" in FEW_SHOT_PROMPT)
# TASK 4: RETRY LOGIC
def call_llm_with_retry(llm_messages, ticket_id):

    global RETRY_FAILURE_COUNT, PERMANENT_FAILURE_COUNT



    for retry_number in range(MAX_RETRIES + 1):

        try:

            response = mock_llm.chat(llm_messages)

            return response



        except mock_llm.TransientLLMError:

            RETRY_FAILURE_COUNT += 1



            if retry_number == MAX_RETRIES:

                PERMANENT_FAILURE_COUNT += 1

                raise



            wait_time = (

                BASE_DELAY

                * (2 ** retry_number)

                * random.uniform(0.9, 1.15)

            )



            print(

                f"Ticket {ticket_id} | "

                f"Retry {retry_number} | "

                f"Waiting {wait_time:.4f} seconds"

            )



            time.sleep(wait_time)
# TASK 5: OUTPUT VALIDATION
def validate_response(llm_response_text):



    # Try JSON first

    try:

        result = json.loads(llm_response_text)



        category = result.get("category")



        if category in ALLOWED_CATEGORIES:

            return category, "classified_json"



        return None, "flagged_invalid_output"



    except (json.JSONDecodeError, AttributeError):

        pass



    # Look for category keywords when the reply is not valid JSON

    words = re.findall(

        r"[a-z]+",

        llm_response_text.lower()

    )



    matched_categories = [

        category

        for category in ALLOWED_CATEGORIES

        if category in words

    ]



    if len(matched_categories) == 1:

        return matched_categories[0], "classified_keyword"



    return None, "flagged_invalid_output"
print("\n========== VALIDATION TEST ==========")
print(

    validate_response(

        '{"category": "billing", "reason": "wrong invoice"}'

    )

)
print(

    validate_response(

        "The customer appears to have a billing-related problem."

    )

)
print(

    validate_response(

        '{"category": "shipping", "reason": "parcel"}'

    )

)
print(

    validate_response(

        "billing and fraud issue"

    )

)
# TASK 4 + 5: FULL ZERO-SHOT PIPELINE
def run_pipeline(system_prompt, token_budget=None):

    global RETRY_FAILURE_COUNT, PERMANENT_FAILURE_COUNT



    # Reset the mock LLM and retry counters at the start of each run

    mock_llm.reset()

    RETRY_FAILURE_COUNT = 0

    PERMANENT_FAILURE_COUNT = 0



    results = []

    total_tokens_used = 0



    for _, ticket in support_tickets.iterrows():



        ticket_id = ticket["ticket_id"]

        customer_id = ticket["customer_id"]

        original_text = ticket["text"]

        redacted_text = ticket["redacted_text"]



        category = None

        status = None

        tokens_used = 0



        # --------------------------

        # INPUT GUARDRAILS

        # --------------------------



        if original_text.strip() == "":

            status = "skipped_empty"



        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            status = "flagged_long"



        # --------------------------

        # TOKEN BUDGET

        # --------------------------



        elif token_budget is not None and total_tokens_used >= token_budget:

            status = "budget_exhausted"



        # --------------------------

        # MODEL CALL

        # --------------------------



        else:



            # Confirm the text is free of PII before sending it to the model

            assert not re.search(EMAIL_PATTERN, redacted_text)

            assert not re.search(PHONE_PATTERN, redacted_text)

            assert not re.search(ORDER_PATTERN, redacted_text)



            llm_messages = [

                {

                    "role": "system",

                    "content": system_prompt

                },

                {

                    "role": "user",

                    "content": redacted_text

                }

            ]



            response = call_llm_with_retry(

                llm_messages,

                ticket_id

            )



            tokens_used = response.usage.total_tokens

            total_tokens_used += tokens_used



            llm_response_text = response.choices[0].message.content



            category, status = validate_response(

                llm_response_text

            )



        # --------------------------

        # SAVE THE TICKET RESULT

        # --------------------------



        results.append({

            "ticket_id": ticket_id,

            "customer_id": customer_id,

            "status": status,

            "category": category,

            "tokens_used": tokens_used,

            "true_category": ticket["true_category"]

        })



    pipeline_results = pd.DataFrame(results)



    print("\n========== PIPELINE RESULTS ==========")

    print(pipeline_results["status"].value_counts())



    print("\nTotal tickets:", len(pipeline_results))

    print("Total tokens:", total_tokens_used)



    classified = pipeline_results[

        pipeline_results["status"].isin([

            "classified_json",

            "classified_keyword"

        ])

    ]



    print("Classified support_tickets:", len(classified))



    if len(classified) > 0:

        accuracy = (

            classified["category"]

            == classified["true_category"]

        ).mean()



        tokens_per_classified = total_tokens_used / len(classified)



        print("Accuracy:", accuracy)

        print("Tokens per classified ticket:", round(tokens_per_classified, 1))

    else:

        print("Accuracy: N/A")

        print("Tokens per classified ticket: N/A")



    print(f"Total transient failures: {RETRY_FAILURE_COUNT}")

    print(f"Total permanent failures: {PERMANENT_FAILURE_COUNT}")



    return pipeline_results
# ZERO-SHOT FULL RUN
# ==========================================
zero_shot_results = run_pipeline(

    ZERO_SHOT_PROMPT

)
print("\n========== ZERO-SHOT STATUS COUNTS ==========")
print(zero_shot_results["status"].value_counts())
# TASK 6: ZERO-SHOT CONFUSION MATRIX
zero_shot_classified = zero_shot_results[

    zero_shot_results["status"].isin([

        "classified_json",

        "classified_keyword"

    ])

]
zero_shot_confusion_matrix = pd.crosstab(

    zero_shot_classified["true_category"],

    zero_shot_classified["category"]

)
zero_shot_confusion_matrix = zero_shot_confusion_matrix.reindex(

    index=["billing", "delivery", "technical", "fraud"],

    columns=["billing", "delivery", "technical", "fraud"],

    fill_value=0

)
print("\n========== ZERO-SHOT CONFUSION MATRIX ==========")
print(zero_shot_confusion_matrix)
few_shot_results = run_pipeline(FEW_SHOT_PROMPT)
# FEW-SHOT METRICS
few_shot_classified = few_shot_results[

    few_shot_results["status"].isin([

        "classified_json",

        "classified_keyword"

    ])

]
few_accuracy = (

    few_shot_classified["category"]

    == few_shot_classified["true_category"]

).mean()
few_tokens = few_shot_results["tokens_used"].sum()
print("\n========== FEW-SHOT METRICS ==========")
print("Classified support_tickets:", len(few_shot_classified))
print("Accuracy:", few_accuracy)
print("Total tokens:", few_tokens)
print("Tokens per classified ticket:", few_tokens / len(few_shot_classified))
# TASK 7A: TOKEN BUDGET DEMO
os.makedirs("outputs", exist_ok=True)
TOKEN_BUDGET = 5000
token_budget_results = run_pipeline(

    ZERO_SHOT_PROMPT,

    token_budget=TOKEN_BUDGET

)
token_budget_results.to_csv(

    "outputs/triage_budget_demo.csv",

    index=False

)
budget_exhausted_tickets = (

    token_budget_results["status"] == "budget_exhausted"

).sum()
print("\n========== BUDGET DEMO ==========")
print("Token budget:", TOKEN_BUDGET)
print("Budget exhausted support_tickets:", budget_exhausted_tickets)
print("Saved: outputs/triage_budget_demo.csv")
# TASK 7B: FINAL FEW-SHOT RUN
TOKEN_BUDGET = 100000
routed_ticket_results = run_pipeline(

    FEW_SHOT_PROMPT,

    token_budget=TOKEN_BUDGET
)
# CUSTOMER SEGMENT INFORMATION
customer_segment_data = pd.read_csv(
    "outputs/customer_segments.csv"
)
# Keep only the required customer information
customer_segment_data = customer_segment_data[

    ["customer_id", "segment_name", "is_premium"]
]
# TASK 7C: ROUTING
routed_ticket_results = routed_ticket_results.merge(

    customer_segment_data,

    on="customer_id",

    how="left"
)
# Treat customers without a segment match as non-premium
routed_ticket_results["is_premium"] = (

    routed_ticket_results["is_premium"].fillna(False)
)
# Set the default support queue and normal priority
routed_ticket_results["route"] = "support_queue"
routed_ticket_results["priority"] = "NORMAL"
# Send skipped or flagged tickets for human review
manual_review_statuses = [

    "skipped_empty",

    "flagged_long",

    "flagged_invalid_output",

    "budget_exhausted"

]
manual_review_mask = routed_ticket_results["status"].isin(

    manual_review_statuses

)
routed_ticket_results.loc[

    manual_review_mask, "route"

] = "human_review"
# Fraud support_tickets go to security team
fraud_ticket_mask = (

    (routed_ticket_results["category"] == "fraud")

    & (~manual_review_mask)

)
routed_ticket_results.loc[

    fraud_ticket_mask, "route"

] = "security_team"
routed_ticket_results.loc[

    fraud_ticket_mask, "priority"

] = "HIGH"
# Route eligible premium customers to the premium desk
premium_customer_mask = (

    routed_ticket_results["is_premium"]

    & (~manual_review_mask)

    & (~fraud_ticket_mask)

)
routed_ticket_results.loc[

    premium_customer_mask, "route"

] = "premium_desk"
routed_ticket_results.loc[

    premium_customer_mask, "priority"

] = "HIGH"
# TASK 7D: SAVE FINAL OUTPUT
routed_ticket_results["segment_name"] = (

    routed_ticket_results["segment_name"].fillna("")
)
routed_ticket_results["category"] = (

    routed_ticket_results["category"].fillna("")

)
triage_output = routed_ticket_results[

    [

        "ticket_id",

        "customer_id",

        "status",

        "category",

        "segment_name",

        "priority",

        "route",

        "tokens_used"

    ]

]
triage_output.to_csv(

    "outputs/triage_results.csv",

    index=False

)
print("\n========== FINAL TRIAGE OUTPUT ==========")
print("Total tickets:", len(triage_output))
print("\nRoute counts:")
print(triage_output["route"].value_counts())
print("\nPriority counts:")
print(triage_output["priority"].value_counts())
print("\nStatus counts:")
print(triage_output["status"].value_counts())
print("\nSaved: outputs/triage_results.csv")
# TASK 7E: FINAL VALIDATION
assert len(triage_output) == 164
assert triage_output["ticket_id"].is_unique
assert triage_output.loc[

    triage_output["status"].isin(manual_review_statuses),

    "category"

].eq("").all()
print("\nFINAL OUTPUT VALIDATION PASSED")
