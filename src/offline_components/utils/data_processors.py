import uuid
import re
from typing import List, Dict, Any, Optional


def get_section_depth(title: str) -> Optional[int]:
    """
    Returns hierarchy depth based on numbering.
    '1'     -> 1
    '1.2'   -> 2
    '1.2.3' -> 3
    None    -> Non-numbered header (title, author, etc.)
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\b", title)
    if not m:
        return None
    return m.group(1).count(".") + 1


def parse_docling_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = doc["texts"]

    # Map self_ref → text object
    text_map = {t["self_ref"]: t for t in texts}

    chunks = []
    section_stack = []  # [(depth, section_ref, section_title)]
    seen_real_section = False

    for ref in doc["body"]["children"]:
        if "$ref" not in ref:
            continue

        t = text_map.get(ref["$ref"])
        if not t:
            continue

        # Skip non-body content
        if t.get("content_layer") != "body":
            continue

        label = t.get("label", "")
        text = t.get("text", "").strip()
        if not text:
            continue

        # -----------------------------
        # SECTION HEADER HANDLING
        # -----------------------------
        if label == "section_header":
            depth = get_section_depth(text)

            # Non-numbered headers (title, author, etc.)
            if depth is None:
                # Keep only the document title (first header)
                if not seen_real_section and not section_stack:
                    section_stack.append((0, t["self_ref"], text))
                continue

            # First real section → drop fake headers
            if depth == 1 and not seen_real_section:
                section_stack = []
                seen_real_section = True

            # Pop same or deeper numeric levels
            while section_stack and section_stack[-1][0] >= depth:
                section_stack.pop()

            section_stack.append((depth, t["self_ref"], text))
            continue

        # -----------------------------
        # NORMAL TEXT NODE
        # -----------------------------
        prov = t.get("prov", [{}])[0]
        page_no = prov.get("page_no")

        chunk = {
            "uuid": str(uuid.uuid4()),
            "text": text,
            "label": label,
            "section_path": [s[2] for s in section_stack],
            "section_refs": [s[1] for s in section_stack],
            "page_no": page_no,
            "char_len": len(text),
        }

        chunks.append(chunk)

    return chunks
