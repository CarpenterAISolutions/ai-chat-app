// script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const loadingSpinner = document.getElementById('loading-spinner');

    const BACKEND_URL = '/api/chat';

    // This is the "memory" for the conversation.
    let conversationHistory = [];

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        addMessage(userMessage, 'user');
        // Add the user's message to the history in the correct format.
        conversationHistory.push({ role: 'user', parts: [{ text: userMessage }] });

        userInput.value = '';
        loadingSpinner.classList.remove('hidden');

        try {
            // Send the entire history to the backend.
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // The body now sends the 'history' object, which matches the backend.
                body: JSON.stringify({ history: conversationHistory })
            });

            const data = await response.json();

            if (!response.ok) {
                // The backend now sends specific errors in a 'detail' key.
                throw new Error(data.detail || 'An unknown error occurred.');
            }

            addMessage(data.answer, 'ai');
            // Add the AI's response to the history to continue the conversation.
            conversationHistory.push({ role: 'model', parts: [{ text: data.answer }] });

        } catch (error) {
            console.error('Fetch Error:', error);
            // Display the specific error message from the backend.
            addMessage(`Error: ${error.message}`, 'ai');
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });

    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageElement.appendChild(paragraph);
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});