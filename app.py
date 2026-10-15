import streamlit as st
from rag import create_qa_chain   # ✅ NON CAMBIA MAI

# ⚠️ CAMBIA PER OGNI PROGETTO: titolo della pagina
st.set_page_config(page_title="AutoVeloce RAG", page_icon="🚗")
# ⚠️ CAMBIA PER OGNI PROGETTO: titolo dell'app
st.title("🚗 AutoVeloce Assistant")

@st.cache_resource
def get_qa_chain():
    return create_qa_chain()   # ✅ NON CAMBIA MAI

qa_chain = get_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ⚠️ CAMBIA PER OGNI PROGETTO: testo del campo di input
if prompt := st.chat_input("Ask a question about AutoVeloce rentals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result['result']
            st.markdown(answer)
            with st.expander("📚 Sources"):
                for doc in result['source_documents']:
                    st.write(f"- {doc.metadata.get('source', 'Unknown')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})