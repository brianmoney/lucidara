import arxiv
import client
import streamlit as st
from time import sleep

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://mintlify.s3-us-west-1.amazonaws.com/dagworksinc/logo/dark.png);
                background-repeat: no-repeat;
                background-size: 70%;
                background-position: 20px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def arxiv_search_container() -> None:
    """Container to query Arxiv using the Python `arxiv` library"""
    form = st.form(key="arxiv_search_form")
    query = form.text_area(
        "arXiv Search Query",
        value="LLM in production",
        help="[See docs](https://lukasschwab.me/arxiv.py/index.html#Search)",
    )

    with st.expander("arXiv Search Parameters"):
        max_results = st.number_input("Max results", value=5)
        sort_by = st.selectbox(
            "Sort by",
            [
                arxiv.SortCriterion.Relevance,
                arxiv.SortCriterion.LastUpdatedDate,
                arxiv.SortCriterion.SubmittedDate,
            ],
            format_func=lambda option: option.value[0].upper() + option.value[1:],
        )
        sort_order = st.selectbox(
            "Sort order",
            [arxiv.SortOrder.Ascending, arxiv.SortOrder.Descending],
            format_func=lambda option: option.value[0].upper() + option.value[1:],
        )

    if form.form_submit_button("Search"):
        st.session_state["arxiv_search"] = dict(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )


def article_selection_container(arxiv_form: dict) -> None:
    """Container to select arxiv search results and send /store_arxiv POST request"""
    results = list(arxiv.Search(**arxiv_form).results())
    form = st.form(key="article_selection_form")
    selection = form.multiselect("Select articles to store", results, format_func=lambda r: r.title)
    if form.form_submit_button("Store"):
        arxiv_ids = [entry.get_short_id() for entry in selection]
        with st.status("Storing arXiv articles"):
            client.post_store_arxiv(arxiv_ids)


def pdf_upload_container():
    """Container to uploader arbitrary PDF files and send /store_pdfs POST request"""
    uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload"):
        with st.status("Storing PDFs"):
            client.post_store_pdfs(uploaded_files)

def delete_documents():
    """Function to delete documents."""
    st.write('Deleting documents...')
    edited_rows = st.session_state["data_editor"]["edited_rows"]
    uuids_to_delete = [st.session_state.data[idx]["uuid"] for idx, value in edited_rows.items() if value["selected"]]
    st.write(uuids_to_delete)
    
    with st.spinner("Updating stored PDF documents"):
        client.delete_documents(uuids_to_delete)
        sleep(2)
        st.success("Files deleted")

def on_confirm_delete():
    """Callback for delete confirmation."""
    st.session_state['delete_confirmed'] = True
    delete_documents()
    st.session_state['delete_confirmed'] = False  # Reset the state


def stored_documents_update() -> None:
    dc = st.session_state['delete_confirmed']
    st.write(st.session_state.delete_confirmed)

    edited_rows = st.session_state["data_editor"]["edited_rows"]
    files_to_delete = []
    for idx, value in edited_rows.items():
        if value["selected"] is True:
            files_to_delete.append(st.session_state.data[idx]["name"])

    st.error(f"Really delete {files_to_delete} {dc}")
    if st.button("Yes, delete", on_click=on_confirm_delete):
        pass

def stored_documents_container():
    if 'delete_confirmed' not in st.session_state:
        st.session_state['delete_confirmed'] = False

    st.write(st.session_state.delete_confirmed)
    
    """Container showing stored PDF documents, results of /documents GET request"""
    response = client.get_all_documents_file_name().json()

    """Create a 'selected' column [0] """
    df_with_selections = [{'selected': False, **d} for d in response['documents']]
    st.session_state.data = df_with_selections
    st.data_editor(
        data=df_with_selections,
        key="data_editor",
        num_rows="dynamic",
        on_change=stored_documents_update
    )


def app() -> None:
    """Streamlit entrypoint for PDF Summarize frontend"""
    # config
    st.set_page_config(
        page_title="📥 ingestion",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )
    add_logo()

    st.title("📥 Ingestion")

    left, right = st.columns(2)

    with left:
        st.subheader("Download from arXiv")
        arxiv_search_container()
        if arxiv_form := st.session_state.get("arxiv_search"):
            article_selection_container(arxiv_form)

    with right:
        st.subheader("Upload PDF files")
        pdf_upload_container()

    st.header("Documents stored in Weaviate")
    stored_documents_container()


if __name__ == "__main__":
    # run as a script to test streamlit app locally
    app()
