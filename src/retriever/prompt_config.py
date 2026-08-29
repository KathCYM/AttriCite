"""Prompt fragments shared by CiteGuard inference and RL dataset preparation."""

CITATION_HUMAN_INTRO = (
    "You are now given an excerpt. Find me the paper cited in the excerpt, using the tools "
    "described above. Please make sure that the paper you select really corresponds to the "
    "excerpt: there will be details mentioned in the excerpt that should appear in the paper. "
    "If you read an abstract and it seems like it could be the paper we’re looking for, read "
    "the paper to make sure. Also: sometimes you’ll read a paper that cites the paper we’re "
    "looking for. In such cases, please go to the references in order to find the full name of "
    "the paper we’re looking for, and search for it, and then select it."
)
