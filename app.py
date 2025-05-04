import streamlit as st

from data_preprocessing import load_dataset, preprocess_name, build_bvd_index
from input_processing import get_user_input, match_company
from competitor_identification import find_potential_competitors
from agent_filtering import find_relevant_competitors
from agents import OpenAIStreamingAgent, call_perplexity_search, call_groq_chatbot, CompetitiveChatAgent

import streamlit as st
import os
from openai import OpenAI

# Ask for OpenAI API Key and save it
def get_and_store_api_key():
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = None

    # If no key in session, ask the user
    if not st.session_state["OPENAI_API_KEY"]:
        api_key_input = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-...",
        )
        if api_key_input:
            with open("OpenAI_API.txt", "w", encoding="utf-8") as f:
                f.write(api_key_input.strip())
            os.environ["OPENAI_API_KEY"] = api_key_input.strip()
            st.session_state["OPENAI_API_KEY"] = api_key_input.strip()
            st.success("API Key saved and loaded.")
            return True
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            return False
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
        return True

# Load and initialize OpenAI client
def init_openai_client():
    with open("OpenAI_API.txt", "r", encoding="utf-8") as f:
        os.environ["OPENAI_API_KEY"] = f.readline().strip()
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def main():
    st.title("Competitive Analysis")

    if not get_and_store_api_key():
        return  # Exit early if no API key provided
    client = init_openai_client()


    # 1) Load data
    df = load_dataset("Poland_desc_Industries_final.xlsx")
    df["company_name_clean"] = df["Company name Latin alphabet"].apply(preprocess_name)
    bvd_index = build_bvd_index(df)

    # 2) User input
    name, desc, bvd_id = get_user_input()

    # When a new search is made, reset previous selections & suggestions
    if st.button("Find Company"):
        st.session_state.pop("selected_company", None)
        st.session_state.pop("suggestions", None)
        st.session_state.pop("relevant_competitors", None)

        match, suggestions = match_company(name, desc, bvd_id, df, bvd_index)
        if match is not None:
            st.success(f"Unique match: **{match['Company name Latin alphabet']}**")
            st.session_state["selected_company"] = match
        else:
            st.warning("No unique match found. Please select one of the Top-5 suggestions:")
            st.session_state["suggestions"] = suggestions

    # 3) If suggestions are available (and no selected_company yet), show radio selection
    if "suggestions" in st.session_state and "selected_company" not in st.session_state:
        suggestions = st.session_state["suggestions"]
        placeholder = "🔍 Please select a company …"
        options = [placeholder] + [f"{c} ({s} %)" for c, s in suggestions]
        choice = st.radio("", options)

        if choice != placeholder:
            selected_name = choice.split(" (")[0]
            sel = df[df["Company name Latin alphabet"] == selected_name].iloc[0]
            st.session_state["selected_company"] = sel
            st.success(f"Selected: {selected_name}")
            st.session_state.pop("suggestions", None)

    # 4) If a company is selected, immediately determine competitors
    if "selected_company" in st.session_state:
        comp = st.session_state["selected_company"]

        # Only display the description
        st.markdown(f"### Description of **{comp['Company name Latin alphabet']}**")
        st.write(comp.get("Description", "No description available."))

        # Only once: search for relevant competitors
        if "relevant_competitors" not in st.session_state:
            with st.spinner("Searching for the relevant competitors for the company..."):
                # Step 3
                industry_cols = [f"Industry {i}" for i in range(1, 10)]
                nace_col = "NACE Rev.2 core code"

                potentials = find_potential_competitors(
                    input_row=comp, df=df,
                    industry_cols=industry_cols, nace_col=nace_col
                )
                candidate_names = potentials["Company name Latin alphabet"].tolist()

                # Step 4
                relevant = find_relevant_competitors(
                    candidates=candidate_names,
                    input_company=comp["Company name Latin alphabet"],
                    search_complexity="medium"
                )
                st.session_state["relevant_competitors"] = relevant

        # Display
        relevant = st.session_state["relevant_competitors"]
        if relevant:
            st.markdown("**Top 3–10 relevant competitors:**")
            for name in relevant:
                st.write(f"- {name}")

            st.markdown("#### What would you like to do next?")
            mode = st.radio(
                    "Selection", 
                    ["Start Chatbot", "Generate Report"],
                    label_visibility="collapsed",  # hides the “Selection” label but avoids the empty-label warning
                    key="mode_select")
        else:
            st.warning("No relevant competitors identified.")


        # after Step 4: Fetch and save background information
        if "background_info" not in st.session_state:
            # Repeat the prompts from agent_filtering
            from agent_filtering import call_perplexity_search
            candidates = st.session_state["relevant_competitors"]
            input_name = comp["Company name Latin alphabet"]
            # 1) Fetch background information silently
            sys1 = "You are a research bot, collecting brief reviews of companies."
            usr1 = (
                f"Input company: {input_name}\n"
                "Candidates:\n" +
                "\n".join(f"- {c}" for c in candidates)
            )
            info = call_perplexity_search(
                system_prompt=sys1,
                user_prompt=usr1,
                complexity="medium"
            )["content"]
            st.session_state["background_info"] = info

        if mode == "Start Chatbot":
            st.markdown("### Chat with the Competitive Analyst")

            # 1) Initialize Agent and chat_history (if not yet done)
            if "chat_agent" not in st.session_state:
                df = st.session_state["df"]
                comp_name = comp["Company name Latin alphabet"]
                comp_desc = comp.get("Description", "")
                competitors = st.session_state["relevant_competitors"]
                background_info = st.session_state["background_info"]

                agent = CompetitiveChatAgent(
                    df=df,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                agent.init_conversation(
                    input_company=comp_name,
                    comp_description=comp_desc,
                    competitors=competitors,
                    background_info=background_info
                )

                st.session_state["chat_agent"] = agent
                # NO more KeyError:
                st.session_state["chat_history"] = []

            agent: CompetitiveChatAgent = st.session_state["chat_agent"]

            # 2) Render old chat history
            for msg in st.session_state["chat_history"]:
                st.chat_message(msg["role"]).write(msg["content"])

            # 3) A single input field
            user_input = st.chat_input(
                "Ask your question to the competition chatbot …",
                key="chatbot_query"
            )

            if user_input:

                # User message
                st.session_state["chat_history"].append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                # Fetch complete answer
                full_answer = agent.handle_user_with_tools(user_input)

                # Assistant message display
                st.session_state["chat_history"].append({"role": "assistant", "content": full_answer})
                st.chat_message("assistant").write(full_answer)
            
                    
        else:
            st.info("Generating the report…")

if __name__ == "__main__":
    # Save df in the session state so that the agent can access it
    if "df" not in st.session_state:
        from data_preprocessing import load_dataset
        st.session_state["df"] = load_dataset("Poland_desc_Industries_final.xlsx")
    main()
