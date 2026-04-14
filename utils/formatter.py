def format_response(responses):
    """
    Format multi-agent responses into structured output
    Supports both text and image responses
    """

    sections = {
        "sentiment": "",
        "trend": "",
        "recommendation": "",
        "sql": "",
        "rag": "",
        "image": "",
        "general": ""
    }

    image_path = None  # store image separately

    for res in responses:

        # ----------------------------
        # Handle IMAGE (dict response)
        # ----------------------------
        if isinstance(res, dict) and res.get("type") == "image":
            image_path = res.get("path", "")

            sections["image"] = f"""
🖼️ Image Generated:

Prompt:
{res.get("prompt", "")}

IMAGE_PATH::{image_path}
"""
            continue

        # ----------------------------
        # Handle TEXT responses
        # ----------------------------
        if not isinstance(res, str):
            res = str(res)

        if "Sentiment" in res:
            sections["sentiment"] = res

        elif "Trend" in res:
            sections["trend"] = res

        elif "Recommendation" in res:
            sections["recommendation"] = res

        elif "SQL" in res:
            sections["sql"] = res

        elif "RAG" in res:
            sections["rag"] = res

        elif "Image" in res:
            sections["image"] = res

        else:
            sections["general"] = res

    # ----------------------------
    # Build final structured output
    # ----------------------------
    output = ""

    if sections["sentiment"]:
        output += "## 📊 Sentiment Analysis\n\n" + sections["sentiment"] + "\n\n"

    if sections["trend"]:
        output += "## 📈 Key Trends\n\n" + sections["trend"] + "\n\n"

    if sections["recommendation"]:
        output += "## 💡 Recommendations\n\n" + sections["recommendation"] + "\n\n"

    if sections["sql"]:
        output += "## 🗄️ Data Insights\n\n" + sections["sql"] + "\n\n"

    if sections["rag"]:
        output += "## 📄 Knowledge Insights\n\n" + sections["rag"] + "\n\n"

    if sections["image"]:
        output += "## 🖼️ Visualization\n\n" + sections["image"] + "\n\n"

    if sections["general"]:
        output += sections["general"]

    return output.strip()