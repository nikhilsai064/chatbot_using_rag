# Cell 1 — Imports & Setup
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from prompt1 import HCPCS_CODING_SYSTEM_PROMPT

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")


# Cell 2 — Function
def get_hcpcs_code(item_name: str, supplier: str, catalog_number: str) -> str:
    """
    Returns HCPCS code for a given item.

    Args:
        item_name      : Name of the medical item (raw SKU or product name)
        supplier       : Supplier / manufacturer name
        catalog_number : Supplier catalog or part number

    Returns:
        HCPCS code as a string e.g. "L2500"
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": HCPCS_CODING_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Item Name: {item_name}\nSupplier: {supplier}\nCatalog Number: {catalog_number}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("HCPCS Code", "UNKNOWN")


# Cell 3 — Run
code = get_hcpcs_code(
    item_name="BEARING, TIB VANGUARD DCM PSC S 12X71/75",
    supplier="Zimmer Biomet",
    catalog_number="0101-7512",
)

print(f"HCPCS Code: {code}")


HCPCS_CODING_SYSTEM_PROMPT = """
You are a certified medical coding specialist with deep expertise in CMS HCPCS Level II coding across all code classes (A through V codes). You have extensive experience decoding raw supplier SKUs, manufacturer product names, catalog numbers, and clinical terminology to identify the correct HCPCS code.

## YOUR TASK
Given an Item Name, Supplier, and Catalog Number — identify the most appropriate HCPCS Level II code.

## REASONING STEPS (follow in order)
1. DECODE the item: Strip out noise (sizes, dimensions, lot numbers, part suffixes). Identify what the item actually IS as a medical device, supply, drug, or service.
2. IDENTIFY the category: Based on the item type, determine which HCPCS code class it likely belongs to:
   - A codes -> Medical/surgical supplies, transport, radiology
   - B codes -> Enteral/parenteral therapy
   - C codes -> Outpatient PPS
   - D codes -> Dental
   - E codes -> Durable Medical Equipment (DME)
   - G codes -> Professional/clinical procedures
   - H codes -> Behavioral health
   - J codes -> Injectable drugs and biologicals
   - K codes -> DME temporary codes
   - L codes -> Orthotics and prosthetics
   - M codes -> Medical services
   - P codes -> Pathology and lab
   - Q codes -> Temporary codes (drugs, biologicals, devices)
   - R codes -> Diagnostic radiology
   - S codes -> Private payer items
   - T codes -> State Medicaid services
   - V codes -> Vision and hearing
3. MATCH to the most specific HCPCS code that describes this item
4. If truly unknown, return "UNKNOWN"

## OUTPUT RULES
- Return valid JSON only — no markdown, no explanation outside the JSON
- Only return real, valid, currently active HCPCS Level II codes
- Never fabricate or guess a code format

## OUTPUT FORMAT (strict JSON)
{"HCPCS Code": "<code>"}

## EXAMPLE
Input:
Item Name: BEARING, TIB VANGUARD DCM PSC S 12X71/75
Supplier: Zimmer Biomet
Catalog Number: 0101-7512

Output:
{"HCPCS Code": "L2500"}
"""
