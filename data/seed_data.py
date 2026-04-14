from ..utils.db import create_table, insert_feedback

def seed_data():
    create_table()

    sample_feedback = [
        ("Delivery was very slow", "negative"),
        ("Product quality is excellent", "positive"),
        ("Packaging was damaged", "negative"),
        ("Customer service was helpful", "positive"),
        ("The product stopped working after 2 days", "negative"),
        ("Fast delivery and good service", "positive"),
        ("The item was not as described", "negative"),
        ("Very satisfied with the purchase", "positive"),
        ("Late delivery again", "negative"),
        ("Support team resolved my issue quickly", "positive"),
        ("Poor build quality", "negative"),
        ("Great value for money", "positive"),
        ("Received wrong item", "negative"),
        ("Easy to use and reliable", "positive"),
        ("Product is okay but delivery is slow", "neutral"),
        ("Terrible customer support", "negative"),
        ("Amazing experience overall", "positive"),
        ("The packaging could be better", "neutral"),
        ("Item arrived broken", "negative"),
        ("Very quick shipping", "positive"),
        ("Not worth the price", "negative"),
        ("Loved the design and usability", "positive"),
        ("Delivery delay is frustrating", "negative"),
        ("Works as expected", "positive"),
        ("Instructions were unclear", "neutral"),
        ("Bad experience with returns", "negative"),
        ("Highly recommend this product", "positive"),
        ("Product quality is inconsistent", "neutral"),
        ("Shipping took too long", "negative"),
        ("Fantastic customer experience", "positive"),
    ]

    for text, sentiment in sample_feedback:
        insert_feedback(text, sentiment)

    print("✅ Sample data inserted successfully!")


if __name__ == "__main__":
    seed_data()