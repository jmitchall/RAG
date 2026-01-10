from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

def get_revision_prompt(template:str):
     return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            # This is used to inject the actual content or message that the post will be based on.
            # The placeholder will be populated with the user's request at runtime.
            MessagesPlaceholder(variable_name="messages")
        ])