from app.vector_search import SearchResult
from app.routes.schemas.conversation import type_model_name


def build_rag_prompt(
    search_results: list[SearchResult],
    model: type_model_name,
    display_citation: bool = True,
) -> str:
    context_prompt = ""
    for result in search_results:
        context_prompt += f"<search_result>\n<content>\n{result['content']}</content>\n<source>\n{result['rank']}\n</source>\n</search_result>"

    inserted_prompt = """Answer the user's question using only the search results below. If the results don't contain the answer, say so. Verify user assertions against the results.

<search_results>
{}
</search_results>

Do NOT directly quote the <search_results>. Answer concisely.
""".format(
        context_prompt,
    )

    if display_citation:
        inserted_prompt += """Cite sources inline using [^<source_id>] format. Do NOT list sources at the end.
Example: first answer [^3]. second answer [^1][^2].
"""

    else:
        inserted_prompt += """Do NOT include citations in [^<source_id>] format.
"""

    return inserted_prompt


def get_prompt_to_cite_tool_results(model: type_model_name) -> str:
    inserted_prompt = """Answer the user's question using only tool results. If tools don't provide the answer, say so. Verify user assertions against tool results.

Cite sources inline using [^source_id] format. Do NOT list sources at the end.
Example: first answer [^ccc]. second answer [^aaa][^bbb].
"""

    return inserted_prompt
