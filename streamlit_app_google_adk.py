import asyncio
import uuid

import streamlit as st
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

st.title("💬 Chatbot - Google ADK")
st.write(
    "This chatbot uses Google's Agent Development Kit (ADK) with Gemini. "
    "Provide your Google API key to get started. "
    "You can obtain one at https://aistudio.google.com/app/apikey."
)

google_api_key = st.text_input("Google API Key", type="password")

if not google_api_key:
    st.info("Please add your Google API key to continue.", icon="🗝️")
else:
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "adk_session_created" not in st.session_state:
        st.session_state.adk_session_created = False

    @st.cache_resource
    def get_runner(api_key: str) -> Runner:
        """Create and cache the ADK Runner."""
        import os
        os.environ["GOOGLE_API_KEY"] = api_key

        agent = Agent(
            name="assistant",
            model="gemini-2.0-flash",
            description="A helpful conversational assistant.",
            instruction="You are a helpful, friendly assistant. Answer questions clearly and concisely.",
        )

        session_service = InMemorySessionService()

        return Runner(
            agent=agent,
            app_name="streamlit_chatbot",
            session_service=session_service,
        )

    async def ensure_session(runner: Runner, session_id: str) -> None:
        """Create the ADK session if it doesn't exist yet."""
        if not st.session_state.adk_session_created:
            await runner.session_service.create_session(
                app_name="streamlit_chatbot",
                user_id="streamlit_user",
                session_id=session_id,
            )
            st.session_state.adk_session_created = True

    async def get_response(runner: Runner, session_id: str, user_message: str) -> str:
        """Send a message to the agent and return its response."""
        await ensure_session(runner, session_id)

        content = types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )

        response_text = ""
        async for event in runner.run_async(
            user_id="streamlit_user",
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text

        return response_text

    runner = get_runner(google_api_key)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = asyncio.run(
                    get_response(runner, st.session_state.session_id, prompt)
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
