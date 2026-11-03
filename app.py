
# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO

# API_URL = "http://127.0.0.1:8000"

# st.set_page_config(
#     page_title="Ring Similarity Search",
#     page_icon="💍",
#     layout="wide"
# )

# st.title("💍 Ring Image Similarity Search")
# st.write("Upload a ring image and discover visually similar rings.")

# uploaded_file = st.file_uploader(
#     "Upload your ring image",
#     type=["jpg", "jpeg", "png", "webp"]
# )

# if uploaded_file is not None:

#     query_image = Image.open(
#         BytesIO(uploaded_file.getvalue())
#     ).convert("RGB")

#     st.subheader("Your Uploaded Image")
#     st.image(query_image, width=250)

#     if st.button("Find Similar Rings", type="primary"):

#         with st.spinner("Searching for similar rings..."):

#             try:
#                 response = requests.post(
#                     f"{API_URL}/search",
#                     files={
#                         "file": (
#                             uploaded_file.name,
#                             uploaded_file.getvalue(),
#                             uploaded_file.type
#                         )
#                     },
#                     timeout=120
#                 )

#                 response.raise_for_status()
#                 data = response.json()

#                 results = data["results"]

#                 st.success(
#                     f"Found {len(results)} similar rings!"
#                 )

#                 st.subheader("✨ Similar Rings")

#                 columns = st.columns(5)

#                 for i, result in enumerate(results):

#                     image_url = (
#                         API_URL + result["image_url"]
#                     )

#                     image_response = requests.get(
#                         image_url,
#                         timeout=30
#                     )
#                     image_response.raise_for_status()

#                     ring_image = Image.open(
#                         BytesIO(image_response.content)
#                     )

#                     with columns[i % 5]:

#                         st.image(
#                             ring_image,
#                             use_container_width=True
#                         )

#                         st.write(
#                             f"**Similarity: "
#                             f"{result['similarity_score']:.2%}**"
#                         )

#             except requests.exceptions.RequestException as e:
#                 st.error(f"API connection error: {e}")

#             except Exception as e:
#                 st.error(f"Something went wrong: {e}")




import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Ring Similarity Search",
    page_icon="💍",
    layout="wide"
)


# Load CLIP + FAISS service only once
@st.cache_resource
def load_search_service():
    from backend.clip_service import search_similar_images
    return search_similar_images


st.title("💍 Ring Image Similarity Search")
st.write("Upload a ring image and discover visually similar rings.")

uploaded_file = st.file_uploader(
    "Upload your ring image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    query_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Your Uploaded Image")
    st.image(query_image, width=250)

    if st.button("Find Similar Rings", type="primary"):

        try:
            with st.spinner("Loading CLIP model and searching..."):

                search_similar_images = load_search_service()

                results = search_similar_images(
                    query_image,
                    top_k=5
                )

            st.success(f"Found {len(results)} similar rings!")

            st.subheader("✨ Similar Rings")

            columns = st.columns(5)

            for i, result in enumerate(results):

                ring_image = Image.open(
                    result["image_path"]
                ).convert("RGB")

                with columns[i % 5]:

                    st.image(
                        ring_image,
                        use_container_width=True
                    )

                    st.write(
                        f"**Similarity: "
                        f"{result['similarity_score']:.2%}**"
                    )

        except Exception as e:
            st.error(f"Something went wrong: {e}")