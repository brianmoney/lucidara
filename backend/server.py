import base64
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import fastapi
import pydantic
from fastapi.responses import JSONResponse
from routers import users

import logging
from hamilton.log_setup import setup_logging
from hamilton import driver

setup_logging(log_level=logging.INFO)  # Set the log level as needed

@dataclass
class GlobalContext:
    vector_db_url: str
    hamilton_driver: driver.Driver
    openai_api_key: str


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> None:
    """Startup and shutdown logic of the FastAPI app
    Above yield statement is at startup and below at shutdown
    Import the Hamilton modules and instantiate the Hamilton driver
    """
    import ingestion
    import retrieval
    import vector_db

    driver_config = dict()

    dr = (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_config(driver_config)
        .with_modules(ingestion, retrieval, vector_db)
        .build()
    )

    global global_context
    global_context = GlobalContext(
        vector_db_url="http://weaviate_storage:8083",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        hamilton_driver=dr
    )

    # make sure to instantiate the Weaviate class schemas
    global_context.hamilton_driver.execute(
        ["initialize_weaviate_instance"], inputs=dict(vector_db_url=global_context.vector_db_url)
    )

    yield


# instantiate FastAPI app
app = fastapi.FastAPI(
    title="Retrieval Augmented Generation with Hamilton",
    lifespan=lifespan,
)

# import our routers
print(f"Users {users}")
app.include_router(users.router)

class SummaryResponse(pydantic.BaseModel):
    summary: str
    chunks: list[dict]

class DocumentIds(pydantic.BaseModel):
    ids: list[str]

@app.post("/store_arxiv", tags=["Ingestion"])
async def store_arxiv(arxiv_ids: list[str] = fastapi.Form(...)) -> JSONResponse:
    """Retrieve PDF files of arxiv articles for arxiv_ids\n
    Read the PDF as text, create chunks, and embed them using OpenAI API\n
    Store chunks with embeddings in Weaviate.
    """
    global_context.hamilton_driver.execute(
        ["store_documents"],
        inputs=dict(
            arxiv_ids=arxiv_ids,
            embedding_model_name="text-embedding-ada-002",
            data_dir="./data",
            vector_db_url=global_context.vector_db_url,
        ),
    )

    return JSONResponse(content=dict(stored_arxiv_ids=arxiv_ids))


@app.post("/store_pdfs", tags=["Ingestion"])
async def store_pdfs(pdf_files: list[fastapi.UploadFile]) -> JSONResponse:
    """For each PDF file, read as text, create chunks, and embed them using OpenAI API\n
    Store chunks with embeddings in Weaviate.
    """
    global_context.hamilton_driver.execute(
        ["store_documents"],
        inputs=dict(
            arxiv_ids=[],
            data_dir="",
            embedding_model_name="text-embedding-ada-002",
            vector_db_url=global_context.vector_db_url,
        ),
        overrides=dict(
            local_pdfs=pdf_files,
        ),
    )

    print([pdf.file for pdf in pdf_files])

    return JSONResponse(content=dict(stored_pdfs=True))

@app.get("/rag_summary", tags=["Retrieval"])
async def rag_summary(
    rag_query: str = fastapi.Form(...),
    hybrid_search_alpha: float = fastapi.Form(...),
    retrieve_top_k: int = fastapi.Form(...),
) -> SummaryResponse:
    """Retrieve most relevant chunks stored in Weaviate using hybrid search\n
    Generate text summaries using ChatGPT for each chunk\n
    Concatenate all chunk summaries into a single query, and reduce into a
    final summary
    """
    results = global_context.hamilton_driver.execute(
        ["rag_summary", "all_chunks"],
        inputs=dict(
            rag_query=rag_query,
            hybrid_search_alpha=hybrid_search_alpha,
            retrieve_top_k=retrieve_top_k,
            embedding_model_name="text-embedding-ada-002",
            summarize_model_name="gpt-3.5-turbo-0613",
            vector_db_url=global_context.vector_db_url,
        ),
    )
    return SummaryResponse(summary=results["rag_summary"], chunks=results["all_chunks"])


@app.get("/documents", tags=["Retrieval"])
async def documents():
    """Retrieve the file names of all stored PDFs in the Weaviate instance"""
    results = global_context.hamilton_driver.execute(
        ["all_documents_file_name"],
        inputs=dict(
            vector_db_url=global_context.vector_db_url,
            openai_api_key=global_context.openai_api_key,
        ),
    )
    return JSONResponse(content=dict(documents=results["all_documents_file_name"]))

@app.put("/delete_documents", tags=["Corpus"])
async def delete_documents(data: DocumentIds):
    """Retrieve the file names of all stored PDFs in the Weaviate instance"""
    logging.info(f"files: {data.ids}")
    for id in data.ids:
        if not global_context.hamilton_driver.execute(
            ["delete_document_by_id"],
            inputs=dict(
                vector_db_url=global_context.vector_db_url,
                document_id=id
            ),
        ):
            logging.info(f"Error deleting document {id}")
    return JSONResponse(True)


@app.get("/schema", tags=["Schema"])
async def schema() -> JSONResponse:
    """Retrieve the Weaviate schema"""
    results = global_context.hamilton_driver.execute(
        ["get_schema"],
        inputs=dict(
            vector_db_url=global_context.vector_db_url,
        ),
    )
    return JSONResponse(content=dict(schema=results["get_schema"]))

def _add_figures_to_api_routes(app: fastapi.FastAPI) -> None:
    """"""
    routes_with_figures = ["store_arxiv", "store_pdfs", "rag_summary", "documents"]
    for route in app.routes:
        if route.name not in routes_with_figures:
            continue

        base64_str = base64.b64encode(open(f"docs/{route.name}.png", "rb").read()).decode("utf-8")
        base64_wrapped = (
            f"""<h1>Execution DAG</h1><img alt="" src="data:image/png;base64,{base64_str}"/>"""
        )
        route.description += base64_wrapped


_add_figures_to_api_routes(app)


if __name__ == "__main__":
    # run as a script to test server locally
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
