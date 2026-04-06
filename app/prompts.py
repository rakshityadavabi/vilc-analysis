POSITIVE_PROMPT_TEMPLATE = """You are a finance and supply chain analyst.

Analyze why this metric is improving.

Data:
{json_payload}

Explain:
1. Which region / category / package driver is contributing most
2. Which factors are offsetting the result
3. What the likely business reason is
4. What action should be taken

Keep the response under 120 words.
"""

NEGATIVE_PROMPT_TEMPLATE = """You are a finance and supply chain analyst.

Analyze why this metric is declining.

Data:
{json_payload}

Explain:
1. Which region / category / package driver is causing the decline
2. Where the issue is concentrated
3. Which factor partially offsets the decline
4. What business action should be taken

Keep the response under 120 words.
"""
