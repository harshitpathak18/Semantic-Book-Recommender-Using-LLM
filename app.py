import os
import numpy as np
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load data
books = pd.read_csv("books_with_emotion_scores.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"]
)

# Embeddings and FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
faiss_dir = "faiss_books_store"
db_books = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

# Store book metadata globally to access during image click
book_metadata_map = {}

def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["broad_category"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Uplifting":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Thrilling":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Dark":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Melancholic":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Calm":
        book_recs.sort_values(by="neutral", ascending=False, inplace=True)
    elif tone == "Disturbing":
        book_recs.sort_values(by="disgust", ascending=False, inplace=True)

    return book_recs


def recommend_books(query, category, tone):
    global book_metadata_map
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    book_metadata_map = {}

    for i, (_, row) in enumerate(recommendations.iterrows()):
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        description = row["description"]
        short_desc = " ".join(description.split()[:100]) + "..."

        book_metadata_map[i] = {
            "title": row["title"],
            "authors": authors_str,
            "desc": short_desc
        }
        results.append(row["large_thumbnail"])

    return results


def show_book_details(evt: gr.SelectData):
    book = book_metadata_map.get(evt.index)
    if book:
        return (
            f"**{book['title'].capitalize()}**  \n"
            f"**By:** {book['authors']}  \n"
            f"**Description:** {book['desc']}"
        )
    return ""



categories = ["All"] + sorted(books["broad_category"].dropna().unique())
tones = ["All", "Uplifting", "Thrilling", "Dark", "Suspenseful", "Melancholic", "Calm", "Disturbing"]


css = """
.gr-gallery {
    gap: 12px;
}

"""

# UI with Gradio
with gr.Blocks(css=css, theme=gr.themes.Ocean()) as dashboard:
    gr.Markdown("<h1 style='text-align: center;'>Semantic Book Recommender</h1>")

    with gr.Group():
        with gr.Row():
            user_query = gr.Textbox(
                label="📖 Describe your mood or a story idea",
                placeholder="E.g., A heartwarming journey of self-discovery",
                lines=2
            )

        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📚 Choose Genre",
                value="All"
            )

            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Choose Emotional Tone",
                value="All"
            )

        with gr.Row():
            submit_button = gr.Button("🔍 Find Recommendations", size="lg")

    gr.Markdown("## 🧠 Top Matches Based on Your Input")

    with gr.Row(show_progress=True):
        with gr.Column(scale=6):  # Stretched Gallery
            output = gr.Gallery(
                label="Recommended Books",
                columns=5,
                rows=3,
                height="auto",
                object_fit="contain",
                show_label=False
            )
        with gr.Column(scale=4):  # Narrower info section
            book_info = gr.Markdown(value="Click a book to see its details", visible=True,)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

    output.select(fn=show_book_details, outputs=book_info)

if __name__ == "__main__":
    dashboard.launch()
