import streamlit as st
import sys
import os

def main():
    # Set page config

    # Dynamically add project folders to sys.path
    root_dir = os.path.dirname(__file__)
    for project in ["Project_1", "Project_2", "Project_3"]:
        full_path = os.path.join(root_dir, project)
        if full_path not in sys.path:
            sys.path.insert(0, full_path)

    # Import project modules safely
    try:
        import main1  # from Project_1
        import main as project2_main  # from Project_2
        import main3  # from Project_3
    except ImportError as e:
        st.error(f"❌ Import failed: {e}")
        st.stop()

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "menu"

    # Custom styling
    st.markdown("""
        <style>
        div.stButton > button {
            font-size: 20px;
            padding: 1em 2em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main menu UI
    def main_menu():
        st.title("MetaScan Optimization Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("SEO Audit Runner", key="audit_btn", use_container_width=True,
                      on_click=lambda: st.session_state.update(page="project1"))
        with col2:
            st.button("Keywords Generator", key="keyword_btn", use_container_width=True,
                      on_click=lambda: st.session_state.update(page="project2"))
        with col3:
            st.button("SEO Blog Generator", key="blog_btn", use_container_width=True,
                      on_click=lambda: st.session_state.update(page="project3"))

    # Sidebar back button
    if st.session_state.page != "menu":
        with st.sidebar:
            if st.button("Back to Main Menu"):
                st.session_state.page = "menu"
                st.rerun()  # 🔁 force UI refresh to hide the button


    # Page router
    if st.session_state.page == "menu":
        main_menu()
    elif st.session_state.page == "project1":
        main1.main()
    elif st.session_state.page == "project2":
        project2_main.main()
    elif st.session_state.page == "project3":
        main3.main()

# Optional for standalone running
if __name__ == "__main__":
    main()
