from utils.vector_store import add_documents


def seed_rag():
    docs = [
    """
    Product Feedback Report Q1:

    Many customers reported delays in shipping due to logistics issues.
    Delivery timelines exceeded expectations in 40% of cases.
    Customers expressed dissatisfaction with late arrivals.
    """,

    """
    Customer Complaint Analysis:

    Product defects were observed in multiple categories including electronics and packaging.
    Several customers reported receiving damaged items.
    Quality control processes need improvement.
    """,

    """
    Customer Support Report:

    Customer support response time is slow.
    Average issue resolution takes more than 48 hours.
    Customers expect faster and more responsive support services.
    """,

    """
    Positive Feedback Summary:

    Some customers appreciated fast delivery and good service.
    Positive reviews highlighted ease of use and product reliability.
    """,

    """
    Packaging Review Document:

    Packaging quality is inconsistent.
    Several reports mention damaged packaging during transit.
    Improvement in packaging materials is recommended.
    """
    ]

    add_documents(docs)
    print("✅ RAG documents added")


if __name__ == "__main__":
    seed_rag()