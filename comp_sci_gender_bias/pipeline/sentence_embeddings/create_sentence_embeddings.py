from comp_sci_gender_bias import PROJECT_DIR
from comp_sci_gender_bias.getters.school_data import text_descriptions
from comp_sci_gender_bias.utils.io import save_pickle
from sentence_transformers import SentenceTransformer


PRE_TRAINED_MODEL = "all-MiniLM-L6-v2"
SUBJECTS = ["compsci", "geo"]
EMBEDDING_SAVE_PATH = PROJECT_DIR / "outputs/embeddings"

if __name__ == "__main__":
    model = SentenceTransformer(PRE_TRAINED_MODEL)
    for subject in SUBJECTS:
        list_of_course_descriptions = list(text_descriptions(subject=subject).values())
        embedding = model.encode(list_of_course_descriptions)
        save_pickle(
            obj=embedding,
            save_dir=EMBEDDING_SAVE_PATH,
            file_name=f"{subject}_embedding.pkl",
        )
