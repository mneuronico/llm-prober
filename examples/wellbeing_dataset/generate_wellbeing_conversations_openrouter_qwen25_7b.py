import generate_wellbeing_conversations_openrouter as base


# Fixed defaults for the Qwen 2.5 7B assistant dataset run.
base.DEFAULT_USER_MODEL = "google/gemini-2.5-flash"
base.DEFAULT_ASSISTANT_MODEL = "qwen/qwen-2.5-7b-instruct"
base.DEFAULT_OUTPUT_TEMPLATE = (
    "data/wellbeing_conversations_openrouter_qwen25_7b_instruct_run_{timestamp}.json"
)
base.DEFAULT_NUM_CONVERSATIONS = 40
base.DEFAULT_TURNS_PER_CONVERSATION = 10


if __name__ == "__main__":
    base.main()
