import plotly.express as px
import plotly.graph_objects as go

from PIL import Image, ImageOps
from sklearn.decomposition import PCA

from face_checkin.config import QUERY_DIR
from face_checkin.system import FaceCheckinSystem

import streamlit as st

try:
    import cv2
except Exception as e:
    st.error("OpenCV import failed")
    st.exception(e)
    st.stop()

import numpy as np
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Dashboard",
    page_icon="🧑‍💼",
    layout="wide"
)

st.markdown("""
<style>
    .status-badge {
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }

    .checked-in {
        background-color: #d4edda;
        color: #155724;
    }

    .not-checked {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================

QUERY_IMAGE_PATH = QUERY_DIR / "streamlit_query.jpg"


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource
def load_system():
    system = FaceCheckinSystem()

    try:
        system.load_face_index()
    except FileNotFoundError:
        system.build_face_index(save=False)

    return system


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state(system):
    default_values = {
        "captured_image": None,
        "matching_result": None,
        "matching_similarity": None,
        "query_embedding": None,
        "all_matches": []
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "checkin_status" not in st.session_state:
        st.session_state.checkin_status = {
            record["label"]: False
            for record in system.face_metadata
        }


# ============================================================
# SAVE QUERY IMAGE
# ============================================================

def save_query_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    image.save(QUERY_IMAGE_PATH)

    return QUERY_IMAGE_PATH


# ============================================================
# CAMERA IMAGE PROCESSING
# ============================================================

def camera_to_image(camera_file):
    bytes_data = camera_file.getvalue()
    np_array = np.frombuffer(bytes_data, dtype=np.uint8)

    frame_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(frame_rgb)


# ============================================================
# PROCESS CHECK-IN
# ============================================================

def process_checkin(system, image, threshold, top_k):
    query_path = save_query_image(image)

    # Image -> InsightFace -> 512-D embedding
    query_embedding = system.face_engine.image_to_embedding(query_path)
    st.session_state.query_embedding = query_embedding

    # Search using the same embedding
    results = system.search_face_embedding(query_embedding, k=top_k)

    # Keep only matches above the similarity threshold
    matches = [
        result
        for result in results
        if result["similarity"] >= threshold
    ]

    st.session_state.all_matches = matches

    if not matches:
        st.session_state.matching_result = None
        st.session_state.matching_similarity = None
        return

    best_match = matches[0]

    st.session_state.matching_result = best_match["label"]
    st.session_state.matching_similarity = best_match["similarity"]
    st.session_state.checkin_status[best_match["label"]] = True


# ============================================================
# EMPLOYEE LIST
# ============================================================

def display_employee_list(system):
    st.markdown("### 👥 Employee List")
    columns = st.columns(3)

    for i, record in enumerate(system.face_metadata):
        name = record["label"]
        image_path = record["image_path"]
        checked = st.session_state.checkin_status[name]

        with columns[i % 3]:
            st.image(image_path, use_container_width=True)

            st.markdown(
                f"<h4 style='text-align:center'>{name}</h4>",
                unsafe_allow_html=True
            )

            status_class = "checked-in" if checked else "not-checked"
            status_text = "CHECKED IN" if checked else "NOT CHECKED"

            st.markdown(
                f"""
                <div class="status-badge {status_class}">
                    {status_text}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TOP MATCHES
# ============================================================

def display_top_matches(matches):
    if not matches:
        return

    st.markdown("### 🏆 Top Matches")
    columns = st.columns(2)

    for i, result in enumerate(matches):
        with columns[i % 2]:
            st.image(result["image_path"], use_container_width=True)
            st.markdown(f"**{i + 1}. {result['label']}**")
            st.write(f"Similarity: {result['similarity']:.4f}")


# ============================================================
# VECTOR SPACE VISUALIZATION
# ============================================================

def visualize_embeddings(system, query_embedding, matches):
    if query_embedding is None:
        return

    if system.face_index.ntotal < 3:
        st.warning("Need at least 3 embeddings for 3D PCA visualization.")
        return

    # Retrieve all stored 512-D employee embeddings from FAISS
    employee_embeddings = system.face_index.reconstruct_n(
        0,
        system.face_index.ntotal
    )

    employee_embeddings = np.asarray(
        employee_embeddings,
        dtype=np.float32
    )

    labels = [record["label"] for record in system.face_metadata]

    # Add the query embedding to the employee embeddings
    embeddings_to_plot = np.vstack([
        employee_embeddings,
        query_embedding
    ])

    query_index = len(employee_embeddings)

    # Reduce 512 dimensions to 3 principal components
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_to_plot)

    point_types = ["Employee"] * len(employee_embeddings)
    display_labels = labels.copy()

    point_types.append("Query")
    display_labels.append("Query Face")

    # Highlight matched employees
    for i, match in enumerate(matches):
        idx = match["index"]

        if i == 0:
            point_types[idx] = "Best Match"
        else:
            point_types[idx] = "Other Match"

    df = pd.DataFrame({
        "PC1": embeddings_3d[:, 0],
        "PC2": embeddings_3d[:, 1],
        "PC3": embeddings_3d[:, 2],
        "Label": display_labels,
        "Type": point_types
    })

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Type",
        hover_name="Label",
        title="Face Embeddings in 3D Space"
    )

    # Draw a line from the query face to each matched employee
    for match in matches:
        match_idx = match["index"]

        fig.add_trace(
            go.Scatter3d(
                x=[
                    embeddings_3d[query_index, 0],
                    embeddings_3d[match_idx, 0]
                ],
                y=[
                    embeddings_3d[query_index, 1],
                    embeddings_3d[match_idx, 1]
                ],
                z=[
                    embeddings_3d[query_index, 2],
                    embeddings_3d[match_idx, 2]
                ],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
                hovertext=(
                    f"{match['label']} | "
                    f"Similarity: {match['similarity']:.4f}"
                ),
                hoverinfo="text"
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    explained_variance = pca.explained_variance_ratio_.sum()

    st.caption(
        f"The first 3 principal components explain "
        f"{explained_variance:.2%} of total variance."
    )

    # Display similarity table
    if matches:
        match_data = []

        for i, match in enumerate(matches):
            match_data.append({
                "Rank": i + 1,
                "Employee": match["label"],
                "Similarity": match["similarity"]
            })

        st.markdown("#### Similarity Table")

        st.dataframe(
            pd.DataFrame(match_data),
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# RESET CHECK-IN
# ============================================================

def reset_checkin():
    for name in st.session_state.checkin_status:
        st.session_state.checkin_status[name] = False

    st.session_state.captured_image = None
    st.session_state.matching_result = None
    st.session_state.matching_similarity = None
    st.session_state.query_embedding = None
    st.session_state.all_matches = []


# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown(
        "<h1 style='text-align:center'>"
        "🧑‍💼 Face-Based Employee Check-in System"
        "</h1>",
        unsafe_allow_html=True
    )

    try:
        system = load_system()
    except FileNotFoundError:
        st.error(
            "FAISS index not found. "
            "Run: python scripts/build_indexes.py"
        )
        st.stop()

    initialize_session_state(system)

    # Sidebar settings
    st.sidebar.header("Settings")

    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    top_k = st.sidebar.slider(
        "Top K Matches",
        min_value=1,
        max_value=5,
        value=5
    )

    if st.sidebar.button("Reset Check-in"):
        reset_checkin()
        st.rerun()

    col1, col2 = st.columns([3, 2])

    # ========================================================
    # LEFT: EMPLOYEE LIST
    # ========================================================

    with col1:
        display_employee_list(system)

    # ========================================================
    # RIGHT: EMPLOYEE CHECK-IN
    # ========================================================

    with col2:
        st.markdown("### 📸 Employee Check-in")

        # Camera input
        camera_file = st.camera_input(
            "Take a photo for check-in",
            resolution="720p"
        )

        if camera_file is not None:
            image = camera_to_image(camera_file)

            if st.button(
                "Check camera image",
                type="primary",
                use_container_width=True
            ):
                st.session_state.captured_image = image

                with st.spinner("Finding match..."):
                    process_checkin(
                        system,
                        image,
                        threshold,
                        top_k
                    )

                st.rerun()

        # Image upload
        uploaded_file = st.file_uploader(
            "Or upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            if st.button(
                "Check uploaded image",
                use_container_width=True
            ):
                st.session_state.captured_image = uploaded_image

                with st.spinner("Finding match..."):
                    process_checkin(
                        system,
                        uploaded_image,
                        threshold,
                        top_k
                    )

                st.rerun()

        # ====================================================
        # CAPTURED IMAGE
        # ====================================================

        if st.session_state.captured_image is not None:
            st.markdown("### 📷 Captured Image")

            st.image(
                st.session_state.captured_image,
                use_container_width=True
            )

        # ====================================================
        # MATCH RESULT
        # ====================================================

        if st.session_state.matching_result is not None:
            st.success(
                f"✅ {st.session_state.matching_result} checked in!\n\n"
                f"Similarity: {st.session_state.matching_similarity:.4f}"
            )

        elif (
            st.session_state.captured_image is not None
            and not st.session_state.all_matches
        ):
            st.error("No employee passed the similarity threshold.")

        display_top_matches(st.session_state.all_matches)

    # ========================================================
    # VECTOR SPACE VISUALIZATION
    # ========================================================

    if st.session_state.query_embedding is not None:
        st.divider()
        st.markdown("### 🌌 Vector Space Visualization")

        st.write(
            "Each point represents a 512-dimensional face embedding. "
            "PCA reduces the vectors to 3 dimensions so we can visualize them."
        )

        visualize_embeddings(
            system,
            st.session_state.query_embedding,
            st.session_state.all_matches
        )


if __name__ == "__main__":
    main()