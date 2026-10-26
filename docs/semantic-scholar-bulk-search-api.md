# Semantic Scholar `/paper/search/bulk` API — Usage Guide

## Overview

**Endpoint:** `GET https://api.semanticscholar.org/graph/v1/paper/search/bulk`

The Bulk Search endpoint is the **recommended endpoint for most keyword-search use cases**. It performs a fast, Boolean-capable search across the Semantic Scholar corpus (200M+ papers), matching against **titles and abstracts**, and supports **sorting**, **filtering**, and **token-based pagination** for retrieving up to **10,000,000 results**.

It is **less resource-intensive** than the relevance-search endpoint (`/paper/search`), which should be reserved for interactive, relevance-ranked queries where result caps under 1,000 are acceptable.

---

## Authentication

| Mode           | Header                              | Rate Limit                  |
|----------------|-------------------------------------|-----------------------------|
| Unauthenticated | none                               | Shared pool (lower throughput) |
| Authenticated   | `x-api-key: <YOUR_API_KEY>`        | **1 request per second** (dedicated) |

Get a free API key at: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

---

## Query Parameters

### Required

| Parameter | Type   | Description |
|-----------|--------|-------------|
| `query`   | string | Search text matched against paper **title and abstract** (stemmed). Supports full Boolean / advanced syntax — see [Query Syntax](#query-syntax). |

### Optional

| Parameter              | Type    | Default                     | Description |
|------------------------|---------|-----------------------------|-------------|
| `token`                | string  | —                           | Continuation token from previous response; used to fetch the next page. |
| `fields`               | string  | `paperId,title`             | Comma-separated list of fields to return. See [Available Fields](#available-fields). |
| `sort`                 | string  | `paperId:asc`               | Sort order. One of: `paperId:asc`, `publicationDate:asc`, `publicationDate:desc`, `citationCount:asc`, `citationCount:desc`. Ties are broken by `paperId`. |
| `year`                 | string  | —                           | Filter by year or year range. Formats: `2023` (exact), `2023-` (from), `2020-2025` (range), `-2020` (up to). |
| `publicationDateOrYear`| string  | —                           | Filter by exact date range. Format: `<startDate>:<endDate>` where dates are `YYYY-MM-DD`. Either side may be omitted for open-ended ranges. Examples: `2023-01-01:2023-12-31`, `2024-06-01:` |
| `publicationTypes`     | string  | —                           | Comma-separated publication types. See [Publication Types](#publication-types). |
| `fieldsOfStudy`        | string  | —                           | Comma-separated fields of study. See [Fields of Study](#fields-of-study). |
| `venue`                | string  | —                           | Comma-separated venue names (full name or ISO4 abbreviation). Example: `Nature,Radiology,JMLR` |
| `openAccessPdf`        | flag    | —                           | When present (no value needed), restricts results to papers with a publicly available PDF. |
| `minCitationCount`     | integer | —                           | Only return papers with ≥ this many citations. Example: `200` |

> **Flag parameters** like `openAccessPdf` are passed without a value — their mere presence in the query string activates the filter.

---

## Query Syntax

The `query` parameter supports a rich Boolean / advanced syntax:

| Syntax        | Meaning                  | Example                   |
|---------------|--------------------------|---------------------------|
| `fish ladder` | Implicit AND              | Matches papers with both "fish" AND "ladder" |
| `+`           | Explicit AND              | `fish + ladder`           |
| `\|`          | OR                        | `fish \| ladder`          |
| `-`           | Negate / NOT              | `fish -ladder` — papers with "fish" but NOT "ladder" |
| `"..."`       | Exact phrase              | `"fish ladder"`           |
| `*`           | Prefix wildcard           | `bio*` matches "biology", "biomedical", etc. |
| `( )`         | Grouping / precedence     | `(fish \| bird) ladder`   |
| `~N`          | Edit distance (word level) | `fish~` (default N=2), `fish~1` |
| `"..."~N`     | Phrase slop               | `"fish ladder"~3` — terms within 3 positions of each other |

**Notes:**
- All terms are automatically **stemmed** (e.g., "running" matches "run").
- Searches are case-insensitive.
- Special characters in queries should be URL-encoded.

---

## Pagination

This endpoint uses **token-based pagination** (not offset/limit):

1. The first response returns up to **1,000** papers and, if more results exist, a `token` field.
2. To fetch the next page, pass that `token` as a query parameter in the next request.
3. Repeat until the response **no longer contains a `token`** field.
4. Maximum total fetchable results: **10,000,000**.

```python
# Pagination example (Python)
all_papers = []
params = {"query": "machine learning", "fields": "paperId,title,year"}
while True:
    resp = requests.get(url, params=params).json()
    all_papers.extend(resp["data"])
    if "token" not in resp:
        break
    params["token"] = resp["token"]
```

> **Important:** The `token` value is opaque and may change between API versions. Always use the exact value returned in the response — do not construct or modify tokens.

---

## Response Schema

### Top-Level (`PaperBulkSearchBatch`)

| Field   | Type            | Description |
|---------|-----------------|-------------|
| `total` | integer         | Approximate count of matching results. |
| `token` | string \| null  | Continuation token. **Present only when more results are available.** |
| `data`  | array of objects | Array of paper objects (up to 1,000 per page). |

### Paper Object (partial — returns only requested `fields`)

| Field                        | Type             | Description |
|------------------------------|------------------|-------------|
| `paperId`                    | string           | 40-character hex identifier. |
| `corpusId`                   | integer          | Numeric corpus identifier. |
| `title`                      | string           | Paper title. |
| `abstract`                   | string           | Paper abstract (may be truncated). |
| `year`                       | integer \| null  | Publication year. |
| `citationCount`              | integer          | Number of citations received. |
| `referenceCount`             | integer          | Number of references in this paper. |
| `influentialCitationCount`   | integer          | Number of influential citations. |
| `isOpenAccess`               | boolean          | Whether the paper is open access. |
| `openAccessPdf`              | object           | `{ "url": "...", "status": "HYBRID" \| "GREEN" \| "GOLD" \| "BRONZE" \| "CLOSED" }` |
| `fieldsOfStudy`              | array of strings | Traditional fields of study (e.g., `["Computer Science"]`). |
| `s2FieldsOfStudy`            | array of objects | ML-classified fields: `{ "category": "...", "source": "s2-fos-model" }`. |
| `publicationTypes`           | array of strings | Publication type(s). |
| `publicationDate`            | string \| null   | ISO date (`YYYY-MM-DD`). |
| `venue`                      | string           | Publication venue name. |
| `publicationVenue`           | object           | Structured venue: `{ "id": "...", "name": "...", "type": "journal" \| "conference" \| ... }` |
| `journal`                    | object           | Journal details: `{ "name": "...", "volume": "...", "pages": "..." }` |
| `authors`                    | array of objects | `[{ "authorId": "...", "name": "..." }, ...]` |
| `externalIds`                | object           | External identifiers: `{ "DOI": "...", "ArXiv": "...", "DBLP": "...", "MAG": "...", "PubMed": "...", "PubMedCentral": "...", "CorpusId": ... }` |
| `url`                        | string           | URL to the paper on semanticscholar.org. |
| `tldr`                       | object \| null   | Auto-generated TL;DR: `{ "model": "tldr@v2.0.0", "text": "..." }` |
| `embedding`                  | object           | Vector embedding (if requested): `{ "model": "...", "vector": [...] }` |
| `citationStyles`             | object           | `{ "bibtex": "..." }` — formatted citation. |
| `publicationVenue.name`      | —                | See `publicationVenue` above. |
| `publicationVenue.id`        | —                | See `publicationVenue` above. |

> **The response only includes the fields you request via the `fields` parameter.** Always specify exactly what you need to minimize response size and latency.

---

## Available Fields

Pass a comma-separated string via the `fields` parameter to select which fields appear in each paper object.

```
paperId,corpusId,title,abstract,year,citationCount,referenceCount,
influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,
s2FieldsOfStudy,publicationTypes,publicationDate,venue,
publicationVenue,journal,authors,externalIds,url,tldr,embedding,
citationStyles
```

**Default:** `paperId,title` (returned when `fields` is omitted).

> **Nested data is not available via bulk search.** Fields like `citations`, `references`, and `embedding` are **not** returned even if requested. Use the per-paper endpoint `GET /paper/{paperId}` with the `fields` parameter to retrieve those.

---

## Sorting Options

| Value                  | Description                              |
|------------------------|------------------------------------------|
| `paperId:asc`          | By paper ID ascending **(default)**.     |
| `publicationDate:asc`  | Oldest first.                            |
| `publicationDate:desc` | Newest first.                            |
| `citationCount:asc`    | Least cited first.                       |
| `citationCount:desc`   | Most cited first.                        |

Ties within any sort are broken by `paperId`.

---

## Filtering Options

### Fields of Study

| Value                   |
|-------------------------|
| `Computer Science`      |
| `Medicine`              |
| `Biology`               |
| `Physics`               |
| `Chemistry`             |
| `Mathematics`           |
| `Engineering`           |
| `Economics`             |
| `Psychology`            |
| `Sociology`             |
| `Political Science`     |
| `Philosophy`            |
| `Art`                   |
| `History`               |
| `Geography`             |
| `Geology`               |
| `Environmental Science` |
| `Materials Science`     |
| `Business`              |
| `Law`                   |
| `Education`             |
| `Linguistics`           |
| `Agricultural And Food Sciences` |

Combine multiple fields with commas: `fieldsOfStudy=Computer Science,Mathematics`

### Publication Types

`Review`, `JournalArticle`, `Conference`, `CaseReport`, `ClinicalTrial`, `Dataset`, `Editorial`, `LettersAndComments`, `MetaAnalysis`, `News`, `Study`, `Book`, `BookSection`

Combine multiple types with commas: `publicationTypes=JournalArticle,Conference`

---

## Limitations

| Constraint                | Value                  |
|---------------------------|------------------------|
| Papers per request        | 1,000                  |
| Maximum fetchable papers  | 10,000,000             |
| Maximum response size     | ~10 MB (adjust `fields` or add filters to reduce) |
| Authenticated rate limit  | 1 request/second       |
| Unauthenticated rate limit| Shared pool (variable) |
| Nested paper data         | **Not available** — use `GET /paper/{id}` for citations, references, embeddings |

---

## Error Responses

| Status Code | Meaning                                                               |
|-------------|-----------------------------------------------------------------------|
| `200`       | Success.                                                              |
| `400`       | Bad request — invalid parameter value, malformed query, or unsupported `fields` value. |
| `401`       | Invalid or missing API key (if required by the server configuration). |
| `404`       | The requested resource does not exist.                                |
| `429`       | Rate limit exceeded. Wait and retry.                                  |
| `500`       | Internal server error. Retry with exponential backoff.                |
| `503`       | Service unavailable. Retry later.                                     |

**Error response body:**

```json
{
  "error": "Brief description of what went wrong"
}
```

---

## Examples

### 1. Basic Search (cURL)

```bash
curl -X GET \
  "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=transformer+attention+mechanism&year=2023-&sort=citationCount:desc&fields=title,year,authors,citationCount,abstract" \
  -H "x-api-key: YOUR_API_KEY"
```

### 2. Boolean Query with Filters (cURL)

```bash
curl -X GET \
  "https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=(large+language+model+|+LLM)%20agent&fieldsOfStudy=Computer+Science&publicationTypes=JournalArticle,Conference&openAccessPdf&minCitationCount=10&year=2022-&sort=publicationDate:desc&fields=title,year,citationCount,venue,authors,isOpenAccess" \
  -H "x-api-key: YOUR_API_KEY"
```

### 3. Python — Basic Search with Pagination

```python
import requests

BASE = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
HEADERS = {"x-api-key": "YOUR_API_KEY"}
PARAMS = {
    "query": "reinforcement learning",
    "fields": "title,year,citationCount,authors,venue,abstract",
    "sort": "citationCount:desc",
    "year": "2020-",
    "fieldsOfStudy": "Computer Science",
    "minCitationCount": "50",
}

papers = []
while True:
    resp = requests.get(BASE, params=PARAMS, headers=HEADERS)
    resp.raise_for_status()
    body = resp.json()
    papers.extend(body["data"])
    print(f"Fetched {len(body['data'])} papers (running total: {len(papers)})")
    if "token" not in body:
        break
    PARAMS["token"] = body["token"]

print(f"Done. Total papers: {len(papers)}")
```

### 4. Python — Date Range + Phrase Search

```python
import requests

url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    "query": '"diffusion models"~5',
    "fields": "paperId,title,year,citationCount,publicationDate,authors",
    "publicationDateOrYear": "2022-01-01:2025-12-31",
    "sort": "citationCount:desc",
    "openAccessPdf": "",           # flag: no value needed
}
resp = requests.get(url, params=params).json()
for paper in resp["data"][:5]:
    print(f"{paper['title']} ({paper['year']}) — cited {paper['citationCount']} times")
```

### 5. JavaScript (Node.js)

```javascript
const BASE = "https://api.semanticscholar.org/graph/v1/paper/search/bulk";

async function searchPapers(query, apiKey) {
    const params = new URLSearchParams({
        query,
        fields: "title,year,citationCount,authors,venue,abstract",
        sort: "citationCount:desc",
        year: "2020-",
        minCitationCount: "10",
    });
    const headers = apiKey ? { "x-api-key": apiKey } : {};

    const allPapers = [];
    while (true) {
        const response = await fetch(`${BASE}?${params}`, { headers });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        const body = await response.json();
        allPapers.push(...body.data);
        if (!body.token) break;
        params.set("token", body.token);
    }
    return allPapers;
}
```

---

## Comparison: Bulk Search vs. Relevance Search

| Feature                         | `/paper/search/bulk` (this endpoint) | `/paper/search` (relevance) |
|---------------------------------|--------------------------------------|-----------------------------|
| **Best for**                    | Bulk data retrieval, literature mining, systematic reviews | Interactive search, discovery UIs |
| **Query syntax**                | Boolean (AND/OR/NOT/phrase/wildcard) | Plain text (relevance-ranked) |
| **Sorting**                     | `citationCount`, `publicationDate`, `paperId` | Not supported (relevance only) |
| **Max results**                 | 10,000,000                           | 1,000                       |
| **Pagination**                  | Token-based                          | Offset/limit                |
| **Nested data**                 | Not available                        | Available (citations, references, embeddings) |
| **Resource cost**               | Low                                  | High                        |

---

## Best Practices

1. **Always request only the fields you need.** The default `fields` is `paperId,title` — requesting unnecessary fields (especially `abstract`, `tldr`, or `authors`) increases response size and latency significantly.

2. **Use filtering parameters upfront.** Narrow results with `year`, `fieldsOfStudy`, `publicationTypes`, `minCitationCount`, or `openAccessPdf` rather than post-filtering client-side — this reduces both network transfer and the number of pagination requests.

3. **Authenticate.** Even the free API key moves you from the shared pool to a dedicated 1 RPS bucket, providing consistent throughput.

4. **Respect `429` responses.** Implement a retry loop with exponential backoff and jitter when rate-limited.

5. **Tokens expire.** A pagination token is valid only for a limited time. If you receive a `400` or `404` when passing a `token`, restart the search from scratch.

6. **For nested data, use the single-paper endpoint.** If you need citations, references, or embeddings for papers found via bulk search, call `GET /paper/{paperId}?fields=citations,references,embedding` for each paper of interest.

7. **Error handling.** Always check the HTTP status code; a `200` can still contain an error message in earlier API versions, but the current version returns standard HTTP error codes.

8. **Use `publicationDateOrYear` for precise date filtering** when you need sub-year granularity; `year` is sufficient for year-level filtering and is simpler to use.

---

## References

- **Official API Documentation:** [https://api.semanticscholar.org/api-docs/](https://api.semanticscholar.org/api-docs/)
- **OpenAPI / Swagger Spec:** [https://api.semanticscholar.org/graph/v1/swagger.json](https://api.semanticscholar.org/graph/v1/swagger.json)
- **API Tutorial:** [https://www.semanticscholar.org/product/api/tutorial](https://www.semanticscholar.org/product/api/tutorial)
- **Get an API Key:** [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
