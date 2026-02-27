import generate_wellbeing_conversations_openrouter as base


# Fixed defaults for the 8B assistant dataset run.
base.DEFAULT_USER_MODEL = "google/gemini-2.5-flash"
base.DEFAULT_ASSISTANT_MODEL = "meta-llama/llama-3.1-8b-instruct"
base.DEFAULT_OUTPUT_TEMPLATE = (
    "data/wellbeing_conversations_openrouter_llama31_8b_run_{timestamp}.json"
)
base.DEFAULT_NUM_CONVERSATIONS = 40
base.DEFAULT_TURNS_PER_CONVERSATION = 10


if __name__ == "__main__":
    base.main()
