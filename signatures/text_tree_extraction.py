import dspy


class ExtractPageTree(dspy.Signature):
    """
    Given a single page from a DSM‑5 book (as plain text), extract a hierarchical tree
    structure that reflects the page’s headings, subheadings, paragraphs, lists, case studies,
    and any other structural elements.

    **CRITICAL INSTRUCTION – VERBATIM CONTENT**
    The `content` field of every node (and the `heading` field, if present) MUST contain
    the **exact, unchanged, original substring** from the input `page`. Do not:
      - Summarise
      - Paraphrase
      - Rephrase
      - Correct grammar or spelling
      - Add or remove any characters (including spaces, line breaks, punctuation)
      - Omit any portion of the text that belongs to that node
    The only exception is that you may split the original text across multiple sibling nodes
    (e.g., separating two paragraphs under the same heading), but each piece must be a
    contiguous, verbatim excerpt from the input.
    Also, ignore the meaningless links and URLs.

    The output must be a dictionary (root node) with these keys:
    - "type" (str): one of "root", "heading", "paragraph", "list_item", "case_study", "criteria_block", etc.
    - "heading" (str or None): if the node is a heading, the EXACT heading text as it appears.
    - "content" (str): the EXACT substring of the original page that belongs to this node
                       (excluding children’s content). Use "" for nodes that purely serve as containers.
    - "children" (list): list of nodes following the same schema, preserving reading order.

    Preserve original line breaks and whitespace only if they are semantically meaningful
    (e.g., between paragraphs). When in doubt, keep them as in the input.
    """

    page: str = dspy.InputField(desc="The raw, unaltered text of a single DSM‑5 page.")
    tree: dict = dspy.OutputField(
        desc="A nested dictionary following the BookTreeNode schema, "
        "where every 'content' and 'heading' string is a **verbatim** excerpt from the input page."
    )
